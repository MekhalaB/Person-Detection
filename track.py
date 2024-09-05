import os
import cv2
import torch
import numpy as np
import yt_dlp

from sort import Sort

# YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
tracker = Sort()

#download youtube video
def download_youtube_video(youtube_url, download_path):
    try:
        ydl_opts = {
            'format': 'mp4',
            'outtmpl': os.path.join(download_path, 'downloaded_video.mp4')
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])
            return os.path.join(download_path, 'downloaded_video.mp4')
    except Exception as e:
        print(f"Error downloading YouTube video: {e}")
        return None


# Input and Output paths
youtube_url = "https://www.youtube.com/watch?v=fEEelCgBkWA"  # Replace with YouTube URL
download_path = "./"

video_path = download_youtube_video(youtube_url, download_path)
output_path = "video4.mp4"

# Open video
cap = cv2.VideoCapture(video_path)

# Video writer setup
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))


def process_and_track(frame):
    # Perform detection
    results = model(frame)
    detections = results.xyxy[0].cpu().numpy()

    # Prepare detections for tracking
    detections_for_tracking = []
    labels = []
    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        # Only consider persons (usually class ID 0 for YOLOv5)
        if int(cls) == 0:
            # Calculate bounding box width and height
            width = x2 - x1
            height = y2 - y1
            print (width*height)

            # Example thresholds (in pixels, adjust as needed)
            threshold = 40000  # Example threshold for child(change acc to video specifications)


            if width * height < threshold:
                detections_for_tracking.append([x1, y1, x2, y2, conf])
                labels.append('Child')
            elif width * height > threshold:
                detections_for_tracking.append([x1, y1, x2, y2, conf])
                labels.append('Therapist')

    detections_for_tracking = np.array(detections_for_tracking)

    # Perform tracking
    tracked_objects = tracker.update(detections_for_tracking)

    # Draw results
    for det, label in zip(tracked_objects, labels):
        x1, y1, x2, y2, obj_id = det
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f'ID: {int(obj_id)} {label}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Write frame to output
    out.write(frame)

    # Optional: Display the frame
    cv2.imshow('Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        return False  # Indicate exit



# Main Loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Process and track the frame
    process_and_track(frame)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()



