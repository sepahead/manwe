import Foundation
import Vision
import CoreML
import AVFoundation

// MARK: - Helpers

func getCurrentTime() -> Double {
    return Double(DispatchTime.now().uptimeNanoseconds) / 1_000_000_000
}

struct BenchmarkResult: Codable {
    let model: String
    let video: String
    let target_fps: Double
    let processed_fps: Double
    let drop_rate: Double
    let avg_latency_ms: Double
    let p99_latency_ms: Double
    let decode_avg_ms: Double
    let inference_avg_ms: Double
}

// MARK: - Detector

class YOLODetector {
    let model: VNCoreMLModel
    let request: VNCoreMLRequest
    
    init(modelPath: URL) throws {
        let config = MLModelConfiguration()
        config.computeUnits = .all // Use ANE/GPU
        
        let compiledUrl = modelPath
        let model = try MLModel(contentsOf: compiledUrl, configuration: config)
        self.model = try VNCoreMLModel(for: model)
        
        self.request = VNCoreMLRequest(model: self.model)
        self.request.imageCropAndScaleOption = .scaleFill
    }
    
    func predict(sampleBuffer: CMSampleBuffer) throws {
        let handler = VNImageRequestHandler(cmSampleBuffer: sampleBuffer, options: [:])
        try handler.perform([request])
        // Access results to ensure computation happens
        _ = request.results
    }
}

// MARK: - Main Benchmark Logic

func runBenchmark(videoPath: String, modelPath: String, targetFps: Double, runId: String) {
    let runDir = ProcessInfo.processInfo.environment["RUN_DIR"] ?? "video_results"
    try? FileManager.default.createDirectory(atPath: runDir, withIntermediateDirectories: true, attributes: nil)
    print("Loading CoreML Model from \(modelPath)...")
    guard let detector = try? YOLODetector(modelPath: URL(fileURLWithPath: modelPath)) else {
        print("Failed to load model")
        exit(1)
    }
    
    let url = URL(fileURLWithPath: videoPath)
    let asset = AVAsset(url: url)
    
    // Create reader
    guard let reader = try? AVAssetReader(asset: asset) else {
        print("Failed to create asset reader")
        exit(1)
    }
    
    guard let track = asset.tracks(withMediaType: .video).first else {
        print("No video track found")
        exit(1)
    }
    
    let outputSettings: [String: Any] = [
        kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
    ]
    
    let trackOutput = AVAssetReaderTrackOutput(track: track, outputSettings: outputSettings)
    reader.add(trackOutput)
    reader.startReading()
    
    // Timing setup
    let frameInterval = targetFps > 0 ? 1.0 / targetFps : 0.0
    var framesPresented = 0
    var framesProcessed = 0
    var latencies: [Double] = []
    var decodeTimes: [Double] = []
    var inferTimes: [Double] = []
    
    let startTime = getCurrentTime()
    var lastFrameTime = startTime
    
    print("Starting benchmark on \(videoPath) (Target: \(targetFps) FPS)...")
    
    // Queue setup for pipeline
    let decodeQueue = DispatchQueue(label: "com.benchmark.decode", qos: .userInitiated)
    let inferenceQueue = DispatchQueue(label: "com.benchmark.inference", qos: .userInteractive)
    let semaphore = DispatchSemaphore(value: 32) // Buffer size
    
    // Shared state
    class PipelineState {
        var isReading = true
        var framesPresented = 0
        var framesProcessed = 0
        var latencies: [Double] = []
        var decodeTimes: [Double] = []
        var inferTimes: [Double] = []
        var lock = NSLock()
    }
    let state = PipelineState()
    
    let group = DispatchGroup()
    
    // Producer (Decode)
    group.enter()
    decodeQueue.async {
        let frameInterval = targetFps > 0 ? 1.0 / targetFps : 0.0
        var loopStart = getCurrentTime()
        
        while reader.status == .reading {
            loopStart = getCurrentTime()
            
            semaphore.wait() // Wait for slot
            
            let t_decode_start = getCurrentTime()
            guard let sampleBuffer = trackOutput.copyNextSampleBuffer() else {
                semaphore.signal()
                break
            }
            let t0 = getCurrentTime() // Frame Arrival (post-decode)
            let decodeDuration = (t0 - t_decode_start) * 1000.0
            
            // Pass to Consumer
            inferenceQueue.async {
                let tInferStart = getCurrentTime()
                do {
                    try detector.predict(sampleBuffer: sampleBuffer)
                    let tInferEnd = getCurrentTime()
                    
                    state.lock.lock()
                    state.decodeTimes.append(decodeDuration)
                    state.inferTimes.append((tInferEnd - tInferStart) * 1000.0)
                    // System Latency: Arrival (t0) to Inference End (tInferEnd)
                    state.latencies.append((tInferEnd - t0) * 1000.0)
                    state.framesProcessed += 1
                    state.lock.unlock()
                } catch {
                    print("Inference error: \(error)")
                }
                semaphore.signal() // Release slot
            }
            
            state.lock.lock()
            state.framesPresented += 1
            state.lock.unlock()
            
            // Throttle (Decoder pacing)
            if targetFps > 0 {
                let elapsed = getCurrentTime() - loopStart
                let sleepTime = frameInterval - elapsed
                if sleepTime > 0 {
                    Thread.sleep(forTimeInterval: sleepTime)
                }
            }
        }
        
        state.lock.lock()
        state.isReading = false
        state.lock.unlock()
        group.leave()
    }
    
    group.wait()
    // Wait for queue to drain? 
    // Since we dispatch async to inferenceQueue, we need to wait for it to finish.
    // We can use the semaphore or simply sync on inferenceQueue.
    inferenceQueue.sync {} 
    
    let totalDuration = getCurrentTime() - startTime
    
    // Stats
    let processedFps = Double(state.framesProcessed) / totalDuration
    let dropRate = state.framesPresented > 0 ? 1.0 - (Double(state.framesProcessed) / Double(state.framesPresented)) : 0.0
    let avgLatency = state.latencies.isEmpty ? 0 : state.latencies.reduce(0, +) / Double(state.latencies.count)
    let avgDecode = state.decodeTimes.isEmpty ? 0 : state.decodeTimes.reduce(0, +) / Double(state.decodeTimes.count)
    let avgInfer = state.inferTimes.isEmpty ? 0 : state.inferTimes.reduce(0, +) / Double(state.inferTimes.count)
    let sortedLatencies = state.latencies.sorted()
    let p99 = sortedLatencies.count > 0 ? sortedLatencies[Int(0.99 * Double(sortedLatencies.count))] : 0
    
    let results = BenchmarkResult(
        model: "swift_coreml",
        video: videoPath,
        target_fps: targetFps,
        processed_fps: processedFps,
        drop_rate: dropRate,
        avg_latency_ms: avgLatency,
        p99_latency_ms: p99,
        decode_avg_ms: avgDecode,
        inference_avg_ms: avgInfer
    )
    
    // Save JSON
    let safeName = URL(fileURLWithPath: videoPath).lastPathComponent.replacingOccurrences(of: ".", with: "_")
    let fileName = "\(runDir)/res_swift_\(safeName)_\(runId).json"
    
    do {
        let encoder = JSONEncoder()
        encoder.outputFormatting = .prettyPrinted
        let data = try encoder.encode(results)
        try data.write(to: URL(fileURLWithPath: fileName))
        print("Saved results to \(fileName)")
    } catch {
        print("Failed to save JSON: \(error)")
    }
    
    print("[Swift CoreML] Processed: \(String(format: "%.2f", processedFps)) FPS | Infer: \(String(format: "%.1f", avgInfer))ms | Decode: \(String(format: "%.1f", avgDecode))ms")
}

// MARK: - CLI Entry

let args = CommandLine.arguments
guard args.count >= 4 else {
    print("Usage: BenchmarkCoreML <model_path_mlmodelc> <video_path> <target_fps> <run_id>")
    exit(1)
}

let modelPath = args[1]
let videoPath = args[2]
let targetFps = Double(args[3]) ?? 0.0
let runId = args.count > 4 ? args[4] : "0"

runBenchmark(videoPath: videoPath, modelPath: modelPath, targetFps: targetFps, runId: runId)
