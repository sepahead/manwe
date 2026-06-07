use std::process::Command;
use std::thread;
use std::time::Duration;

fn main() {
    println!("Building camera_view...");
    let status = Command::new("cargo")
        .args(["build", "--bin", "camera_view", "--release"])
        .status()
        .expect("Failed to build camera_view");

    if !status.success() {
        eprintln!("Build failed!");
        return;
    }

    let urls = vec![
        "rtsp://admin:sauronsauron1@192.168.10.172:554/h264",
        // Assuming other cameras might be on sequential IPs or same IP different channel
        // For now using the same one as requested/safe default
        "rtsp://admin:sauronsauron1@192.168.10.145:554/h264",
        "rtsp://admin:sauronsauron1@192.168.10.172:554/h264",
    ];

    let mut children = vec![];

    for (i, url) in urls.iter().enumerate() {
        println!("Spawning camera view {} for {}", i + 1, url);
        let child = Command::new("./target/release/camera_view")
            .args(["--url", url])
            // .stdout(Stdio::null()) // Optional: suppress output
            // .stderr(Stdio::null())
            .spawn()
            .expect("Failed to spawn camera_view");
        children.push(child);
        
        // Small delay to avoid window overlap issues or resource contention on startup
        thread::sleep(Duration::from_secs(1));
    }

    println!("All processes spawned. Press Ctrl+C to exit launcher (processes may persist).");
    
    // Wait for children? Or just exit? 
    // If we exit, children might become zombies or keep running (which is what we want probably).
    // But usually we want to keep the launcher alive.
    for mut child in children {
        let _ = child.wait();
    }
}
