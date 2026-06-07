import os
import glob
import random
import subprocess
import sys

def generate_video(fps, duration_sec, output_file):
    print(f"Generating {output_file} ({fps} FPS, {duration_sec}s)...")
    
    # Get images
    image_dir = "assets/kaggle1/train/images"
    images = sorted(glob.glob(os.path.join(image_dir, "*.jpg"))) # Assuming JPG
    if not images:
        # Try png
        images = sorted(glob.glob(os.path.join(image_dir, "*.png")))
    
    if not images:
        print("No images found!")
        return

    total_frames = fps * duration_sec
    
    # Create a file list for ffmpeg
    # ffmpeg concat demuxer format: "file 'path/to/file'"
    # We repeat images to fill the duration
    
    list_file = f"file_list_{fps}.txt"
    with open(list_file, "w") as f:
        for i in range(total_frames):
            # Pick image (loop or random)
            # Let's loop to keep it deterministic-ish, or random as requested "random order"
            # User said: "reuse the same ones but in a random order"
            img_path = random.choice(images)
            # Escape single quotes for ffmpeg
            img_path = os.path.abspath(img_path).replace("'", "'\\''")
            f.write(f"file '{img_path}'\n")
            # Duration of this frame
            f.write(f"duration {1.0/fps}\n")
            
    # Run ffmpeg
    # -f concat -safe 0 -i list.txt -vf scale=640:640 -c:v libx264 -pix_fmt yuv420p output.mp4
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", list_file,
        "-r", str(fps),
        "-vf", "scale=640:640",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        output_file
    ]
    
    # Run and capture output for debugging
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error generating {output_file}:")
        print(result.stderr)
    else:
        print(f"Done: {output_file}")
    
    # Clean up list
    if os.path.exists(list_file):
        os.remove(list_file)

def main():
    fps_list = [15, 30, 60, 90, 120]
    duration = 60 # seconds
    
    for fps in fps_list:
        output = os.path.join("video_benchmarks", f"bench_{fps}fps.mp4")
        generate_video(fps, duration, output)

if __name__ == "__main__":
    main()
