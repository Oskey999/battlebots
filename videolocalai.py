#!/usr/bin/env python3
"""
Process MP4 video with locally-stored Roboflow object detection model.
Uses template matching to assign unique IDs to robots across frames.
"""

import cv2
import sys
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from collections import defaultdict
import pickle
import os
import time

def get_centroid(box_coords):
    """Calculate centroid of bounding box."""
    x1, y1, x2, y2 = box_coords
    return ((x1 + x2) / 2, (y1 + y2) / 2)

def euclidean_distance(p1, p2):
    """Calculate distance between two points."""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

class RobotTracker:
    """Track and identify robots across video frames using templates."""
    
    def __init__(self, max_distance=100, template_update_interval=30, template_dir="templates", scale_factor=0.5):
        self.robot_templates = {}  # {robot_id: template_image}
        self.robot_centroids = {}  # {robot_id: last_centroid}
        self.robot_ids = {}  # {current_detection_index: robot_id}
        self.next_id = 1
        self.max_distance = max_distance
        self.template_update_interval = template_update_interval
        self.frame_count = 0
        self.template_dir = template_dir
        self.scale_factor = scale_factor
        
        # Create templates directory if it doesn't exist
        os.makedirs(self.template_dir, exist_ok=True)
    
    def match_template(self, frame, detection_crop, detection_box):
        """Find best matching robot template using SIFT feature matching and weighted color density."""
        if not self.robot_templates:
            return None, 0.0
        
        best_match_id = None
        best_match_score = 0.0
        
        # Initialize SIFT detector
        sift = cv2.SIFT_create()
        
        # Detect and compute features for detection crop
        detection_kp, detection_desc = sift.detectAndCompute(detection_crop, None)
        
        if detection_desc is None or len(detection_kp) == 0:
            return None, 0.0
        
        # BFMatcher for feature matching
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        
        for robot_id, template in self.robot_templates.items():
            # Resize template to match detection crop size if needed
            if template.shape != detection_crop.shape:
                template_resized = cv2.resize(template, 
                    (detection_crop.shape[1], detection_crop.shape[0]))
            else:
                template_resized = template
            
            # Detect and compute features for template
            template_kp, template_desc = sift.detectAndCompute(template_resized, None)
            
            if template_desc is None or len(template_kp) == 0:
                continue
            
            # Match features using Lowe's ratio test
            matches = bf.knnMatch(detection_desc, template_desc, k=2)
            
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.7 * n.distance:
                        good_matches.append(m)
            
            # Calculate feature match score
            if len(good_matches) > 0:
                feature_match_score = len(good_matches) / max(len(detection_kp), len(template_kp))
            else:
                feature_match_score = 0.0
            
            # Calculate weighted color density (center-weighted)
            h, w = detection_crop.shape[:2]
            
            # Create Gaussian weight map (higher values at center, lower at edges)
            y = np.linspace(-1, 1, h)
            x = np.linspace(-1, 1, w)
            X, Y = np.meshgrid(x, y)
            weight_map = np.exp(-(X**2 + Y**2) / 0.5)  # Gaussian centered on middle
            
            # Normalize weight map
            weight_map = weight_map / np.max(weight_map)
            
            # Convert to grayscale for color density calculation
            detection_gray = cv2.cvtColor(detection_crop, cv2.COLOR_BGR2GRAY) if len(detection_crop.shape) == 3 else detection_crop
            template_gray = cv2.cvtColor(template_resized, cv2.COLOR_BGR2GRAY) if len(template_resized.shape) == 3 else template_resized
            
            # Calculate weighted average color density
            # detection_weighted_density = np.average(detection_gray, weights=weight_map)
            # template_weighted_density = np.average(template_gray, weights=weight_map)
            detection_weighted_densityb = np.average(detection_crop[:, :, 0], weights=weight_map)
            template_weighted_densityb = np.average(template_resized[:, :, 0], weights=weight_map)
            detection_weighted_densityg = np.average(detection_crop[:, :, 1], weights=weight_map)
            template_weighted_densityg = np.average(template_resized[:, :, 1], weights=weight_map)
            detection_weighted_densityr = np.average(detection_crop[:, :, 2], weights=weight_map)
            template_weighted_densityr = np.average(template_resized[:, :, 2], weights=weight_map)
            
            # Calculate color density similarity
            color_density_diffb = 1.0 - (abs(detection_weighted_densityb - template_weighted_densityb) / 255.0)
            color_density_diffg = 1.0 - (abs(detection_weighted_densityg - template_weighted_densityg) / 255.0)
            color_density_diffr = 1.0 - (abs(detection_weighted_densityr - template_weighted_densityr) / 255.0)
            color_density_diff = (color_density_diffb + color_density_diffg + color_density_diffr) / 3.0
            
            # Combine scores: 80% feature matching, 20% color density
            # Feature matching is heavily weighted
            combined_score = (0.6 * feature_match_score) + (0.4 * color_density_diff)
            
            if combined_score > best_match_score:
                best_match_score = combined_score
                best_match_id = robot_id
        
        # Only return match if score is above threshold
        print(f"Best match ID: {feature_match_score},{color_density_diff}")
        print(f"Template match score: {best_match_score:.3f} for detection box {detection_box}")
        match=0.4
        if len(self.robot_templates)>1:
            match=0.24
        if best_match_score > max(0.24, 0.38 - (self.frame_count / 200)) :
            return best_match_id,best_match_score
        
        return None,best_match_score
    
    def match_centroid(self, centroid):
        """Find closest robot by centroid distance."""
        if not self.robot_centroids:
            return None
        
        closest_id = None
        closest_distance = self.max_distance
        
        for robot_id, last_centroid in self.robot_centroids.items():
            dist = euclidean_distance(centroid, last_centroid)
            if dist < closest_distance:
                closest_distance = dist
                closest_id = robot_id
        
        return closest_id
    
    def update(self, detections, frame, model=None):
        """Update tracker with new detections."""
        self.frame_count += 1
        matched_ids = set()
        
        for box in detections.boxes:
            # Scale coordinates back to full resolution
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = int(x1 / self.scale_factor), int(y1 / self.scale_factor), \
                              int(x2 / self.scale_factor), int(y2 / self.scale_factor)
            
            # Get class name
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id] if model else None
            
            # Skip if not a robot detection
            if cls_name and cls_name.lower() != "robot":
                continue
            
            centroid = get_centroid([x1, y1, x2, y2])
            
            # Crop detection region (same as drawing rectangle)
            detection_crop = frame[y1:y2, x1:x2]
            
            if detection_crop.size == 0:
                continue
            
            # Try to match with existing templates - prioritize template match
            template_match_id,score = self.match_template(frame, detection_crop, [x1, y1, x2, y2])
            
            # Only use centroid matching if template matching failed
            if template_match_id is not None:
                robot_id = template_match_id
            else:
                centroid_match_id = self.match_centroid(centroid)
                if centroid_match_id is not None:
                    robot_id = centroid_match_id
                else:
                    # New robot detected
                    robot_id = self.next_id
                    self.next_id += 1
            
            matched_ids.add(robot_id)
            
            # Update centroid
            self.robot_centroids[robot_id] = centroid
            
            # Update template periodically
            if robot_id not in self.robot_templates or \
               (self.frame_count % self.template_update_interval == 0 and score>0.38):
                self.robot_templates[robot_id] = detection_crop.copy()
                # Save template to disk
                self.save_template(robot_id, detection_crop)
        
        return matched_ids
    
    def save_template(self, robot_id, template_image):
        """Save template image to templates folder."""
        filename = os.path.join(self.template_dir, f"robot_{robot_id}_template.jpg")
        cv2.imwrite(filename, template_image)
    
    def save_all_templates(self):
        """Save all final templates to disk."""
        for robot_id, template in self.robot_templates.items():
            self.save_template(robot_id, template)
        print(f"Saved {len(self.robot_templates)} robot templates to '{self.template_dir}' folder")


def process_video_with_detection(input_video_path, output_video_path, model_path, 
                                 frame_skip=2, scale_factor=0.5, confidence=0.4):
    """
    Apply YOLO object detection to video frames with robot tracking and save output.
    
    Args:
        input_video_path: Path to input MP4 file
        output_video_path: Path to output MP4 file
        model_path: Path to local YOLO model file (.pt)
        frame_skip: Process every nth frame (e.g., 2 = every 2nd frame)
        scale_factor: Scale frames to this factor (e.g., 0.5 = 50% resolution)
        confidence: Confidence threshold for detections (0-1)
    """
    
    # Load model locally
    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)
    
    # Initialize tracker
    tracker = RobotTracker(max_distance=150, template_update_interval=30, template_dir="templates", scale_factor=scale_factor)
    
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
    
    print(f"Video: {width}x{height}, {fps} FPS, {total_frames} frames")
    print(f"Processing every {frame_skip} frame(s) at {scale_factor*100:.0f}% resolution")
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    frame_count = 0
    last_detections = None
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            frame_count += 1
            print(f"Frame {frame_count}/{total_frames}", end='\r')
            
            # Process frame or use cached detections
            if (frame_count - 1) % frame_skip == 0:
                # Resize for faster inference
                h, w = frame.shape[:2]
                small_frame = cv2.resize(frame, (int(w * scale_factor), int(h * scale_factor)))
                
                try:
                    # Run inference
                    results = model(small_frame, conf=confidence, verbose=False)
                    last_detections = results[0]
                    
                    # Update tracker with detections from full-res frame
                    tracker.update(last_detections, frame, model)
                    
                except Exception as e:
                    print(f"Inference error on frame {frame_count}: {e}")
                    last_detections = None
            
            # Draw detections on full-resolution frame
            if last_detections is not None:
                detection_idx = 0
                for box in last_detections.boxes:
                    # Scale coordinates back to original resolution
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = int(x1 / scale_factor), int(y1 / scale_factor), \
                                      int(x2 / scale_factor), int(y2 / scale_factor)
                    
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    cls_name = model.names[cls_id]
                    
                    # Get robot ID from tracker
                    robot_id = None
                    for rid, centroid in tracker.robot_centroids.items():
                        box_centroid = get_centroid([x1, y1, x2, y2])
                        if euclidean_distance(centroid, box_centroid) < 50:
                            robot_id = rid
                            break
                    
                    # Draw rectangle
                    color = (0, 255, 0) if robot_id is None else (0, 255 - robot_id*30, robot_id*100)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw label with robot ID and confidence
                    if robot_id is not None:
                        label = f"Robot_{robot_id}: {cls_name} ({conf:.2f})"
                    else:
                        label = f"{cls_name}: {conf:.2f}"
                    
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(frame, (x1, y1 - label_size[1] - 8), 
                                (x1 + label_size[0], y1), color, -1)
                    cv2.putText(frame, label, (x1, y1 - 4), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                    
                    detection_idx += 1
            
            # Write frame to output video
            out.write(frame)
        
        print(f"\nProcessing complete! Processed {frame_count} frames")
        print(f"Detected {len(tracker.robot_templates)} unique robots")
        
        # Save all templates to disk
        tracker.save_all_templates()
        
    finally:
        cap.release()
        out.release()
    
    print(f"Output saved to: {output_video_path}")


if __name__ == "__main__":
    # Configuration
    MODEL_PATH = "runs/detect/train2/weights/best.pt"
    INPUT_VIDEO = "BZ-nhrl_dec25_3lb-projectliftoff3-jackalope-470f-Cage-6-Overhead-High.mp4"
    OUTPUT_VIDEO = INPUT_VIDEO + "_output_tracked0.1.mp4"
    FRAME_SKIP = 120
    SCALE_FACTOR = 0.3
    CONFIDENCE = 0.6
    
    if len(sys.argv) > 1:
        INPUT_VIDEO = sys.argv[1]
        OUTPUT_VIDEO = INPUT_VIDEO + "_output_tracked0.1.mp4"
    if len(sys.argv) > 2:
        OUTPUT_VIDEO = sys.argv[2]
    if len(sys.argv) > 3:
        MODEL_PATH = sys.argv[3]
    if len(sys.argv) > 4:
        FRAME_SKIP = int(sys.argv[4])
    if len(sys.argv) > 5:
        SCALE_FACTOR = float(sys.argv[5])

    
    
    if not Path(INPUT_VIDEO).exists():
        print(f"Error: Input video file '{INPUT_VIDEO}' not found")
        sys.exit(1)
    
    if not Path(MODEL_PATH).exists():
        print(f"Error: Model file '{MODEL_PATH}' not found")
        print("Download your Roboflow model as a YOLO format .pt file")
        sys.exit(1)
    
    process_video_with_detection(INPUT_VIDEO, OUTPUT_VIDEO, MODEL_PATH, 
                                FRAME_SKIP, SCALE_FACTOR, CONFIDENCE)
    
    start_time = time.perf_counter() # Record the start time
    process_video_with_detection(INPUT_VIDEO, OUTPUT_VIDEO, MODEL_PATH, 
                                FRAME_SKIP, SCALE_FACTOR, CONFIDENCE)
    end_time = time.perf_counter()   # Record the end time

    elapsed_time = end_time - start_time
    print(f"The method took {elapsed_time:.4f} seconds to run.")
    # print(f"Result: {elapsed_time}")