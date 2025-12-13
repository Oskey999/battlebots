#!/usr/bin/env python3
"""
Process MP4 video with Roboflow object detection model and output highlighted video.
"""

import cv2
import sys
from roboflow import Roboflow
from pathlib import Path

def process_video_with_detection(input_video_path, output_video_path, api_key, model_id):
    """
    Apply Roboflow object detection to video frames and save highlighted output.
    
    Args:
        input_video_path: Path to input MP4 file
        output_video_path: Path to output MP4 file
        api_key: Roboflow API key
        model_id: Model identifier (e.g., "true-battlebots/1")
    """
    
    # Initialize Roboflow
    rf = Roboflow(api_key=api_key)
    project = rf.workspace().project("true-battlebots")
    model = project.version(1).model
    
    # Open input video
    cap = cv2.VideoCapture(input_video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {input_video_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            frame_count += 1
            print(f"Processing frame {frame_count}/{total_frames}", end='\r')
            
            # Save frame temporarily for inference
            temp_frame_path = "temp_frame.jpg"
            cv2.imwrite(temp_frame_path, frame)
            
            try:
                # Run inference
                results = model.predict(temp_frame_path, confidence=40)
                predictions = results.json()["predictions"]
                
                # Draw bounding boxes and labels
                for pred in predictions:
                    x = int(pred["x"])
                    y = int(pred["y"])
                    w = int(pred["width"])
                    h = int(pred["height"])
                    conf = pred.get("confidence", 0)
                    cls = pred.get("class", "Object")
                    
                    # Calculate box corners
                    x1 = max(0, x - w // 2)
                    y1 = max(0, y - h // 2)
                    x2 = min(width, x + w // 2)
                    y2 = min(height, y + h // 2)
                    
                    # Draw rectangle
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Draw label with confidence
                    label = f"{cls}: {conf:.2f}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                    cv2.rectangle(frame, (x1, y1 - label_size[1] - 4), 
                                (x1 + label_size[0], y1), (0, 255, 0), -1)
                    cv2.putText(frame, label, (x1, y1 - 2), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            except Exception as e:
                print(f"Inference error on frame {frame_count}: {e}")
            
            # Write frame to output video
            out.write(frame)
        
        print(f"\nProcessing complete! Processed {frame_count} frames")
        
    finally:
        cap.release()
        out.release()
        
        # Clean up temp file
        try:
            Path("temp_frame.jpg").unlink()
        except:
            pass
    
    print(f"Output saved to: {output_video_path}")


if __name__ == "__main__":
    # Configuration
    API_KEY = "1tawwnJjMmVq9BJnslF7"  # Get from https://app.roboflow.com/settings/api
    INPUT_VIDEO = "BZ-nhrl_dec25_3lb-pinevictus-silentspring-W-11-Cage-2-Overhead-High.mp4"           # Input video file
    OUTPUT_VIDEO = "output_detected.mp4" # Output video file
    
    if len(sys.argv) > 1:
        INPUT_VIDEO = sys.argv[1]
    if len(sys.argv) > 2:
        OUTPUT_VIDEO = sys.argv[2]
    if len(sys.argv) > 3:
        API_KEY = sys.argv[3]
    
    if not Path(INPUT_VIDEO).exists():
        print(f"Error: Input video file '{INPUT_VIDEO}' not found")
        sys.exit(1)
    
    if API_KEY == "YOUR_ROBOFLOW_API_KEY":
        print("Error: Please set your Roboflow API key")
        print("Get it from: https://app.roboflow.com/settings/api")
        sys.exit(1)
    
    process_video_with_detection(INPUT_VIDEO, OUTPUT_VIDEO, API_KEY, "true-battlebots/1")