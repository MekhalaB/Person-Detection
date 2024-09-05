This project demonstrates how to perform object detection and tracking using YOLOv5 and the SORT (Simple Online and Realtime Tracking) algorithm on a YouTube video.

## Overview

The script performs the following steps:
1. Downloads a YouTube video.
2. Applies YOLOv5 for object detection.
3. Uses SORT to track detected objects across frames.
4. Labels objects as 'Child' or 'Therapist' based on their size.
5. Outputs the processed video with tracked objects.

## Requirements

To run this project, you need to have the following packages installed:
- `numpy`
- `opencv-python`
- `torch`
- `ultralytics` (for YOLOv5)
- `yt-dlp`
- `sort` (for tracking)

You can install the required packages using pip:
```
pip install numpy opencv-python torch ultralytics yt-dlp sort
```

## Usage

Clone the repository or download the script.

Ensure you have all the required packages installed.

### Modify the script:

Update the youtube_url variable with the URL of the YouTube video you want to process.
Adjust the download_path and output_path as needed.

### The script will:

Download the video from YouTube.
Process each frame to detect and track objects.
Label objects as 'Child' or 'Therapist' based on their size.
Save the processed video to the specified output path.
Notes
Thresholds for Labeling: The script uses a simple threshold to label objects based on their bounding box area. Adjust the threshold variable in the process_and_track function based on your specific requirements.
YouTube Video Format: The script downloads the video in MP4 format. Ensure that the video format is compatible with OpenCV.
