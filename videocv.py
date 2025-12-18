# import cv2
# import numpy as np
# from pathlib import Path

# def detect_robots_in_video(input_path, output_path, min_contour_area=500, 
#                           frame_skip=10, margin=50, scale=0.3):
#     """
#     Detects robots in a video with static background and outputs labeled video.
#     Uses background subtraction on downscaled frames for faster processing.
    
#     Args:
#         input_path (str): Path to input MP4 video
#         output_path (str): Path to output MP4 video with labels
#         min_contour_area (int): Minimum contour area to be considered a robot (on scaled frame)
#         frame_skip (int): Process every Nth frame (e.g., 10 = process every 10th frame)
#         margin (int): Pixel margin from frame edges to ignore detections
#         scale (float): Scale factor for processing (0.3 = 30% of original size)
#     """
    
#     cap = cv2.VideoCapture(input_path)    
#     if not cap.isOpened():
#         print(f"Error: Could not open video {input_path}")
#         return
    
#     # Get video properties
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
#     # Scaled dimensions for processing
#     scaled_width = int(width * scale)
#     scaled_height = int(height * scale)
#     scaled_margin = int(margin * scale)
    
#     # Create video writer
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
#     # Initialize background subtractor (MOG2 works well for static backgrounds)
#     bg_subtractor = cv2.createBackgroundSubtractorMOG2(
#         detectShadows=False,
#         varThreshold=50
#     )
    
#     # Morphological operations kernel
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
#     frame_count = 0
    
#     while True:
#         ret, frame = cap.read()
        
#         if not ret:
#             break
        
#         frame_count += 1
        
#         # Only process every Nth frame
#         if frame_count % frame_skip != 0:
#             out.write(frame)
#             continue
        
#         # Downscale frame for faster processing
#         small_frame = cv2.resize(frame, (scaled_width, scaled_height))
        
#         # Apply background subtraction
#         fg_mask = bg_subtractor.apply(small_frame)
        
#         # Apply morphological operations to clean up
#         fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=2)
#         fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
#         # Find contours
#         contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
#         # Filter contours by area and get bounding boxes
#         robots = []
#         for contour in contours:
#             area = cv2.contourArea(contour)
#             if area > min_contour_area:
#                 x, y, w, h = cv2.boundingRect(contour)
                
#                 # Check if object is within the valid region (not on outskirts)
#                 if (x > scaled_margin and y > scaled_margin and 
#                     x + w < scaled_width - scaled_margin and y + h < scaled_height - scaled_margin):
                    
#                     # Scale coordinates back to original resolution
#                     orig_x = int(x / scale)
#                     orig_y = int(y / scale)
#                     orig_w = int(w / scale)
#                     orig_h = int(h / scale)
                    
#                     robots.append({
#                         'bbox': (orig_x, orig_y, orig_w, orig_h),
#                         'center': (orig_x + orig_w // 2, orig_y + orig_h // 2),
#                         'area': area
#                     })
        
#         # Sort robots by x-coordinate for consistent labeling
#         robots.sort(key=lambda r: r['center'][0])
        
#         # Draw bounding boxes and labels on frame
#         for idx, robot in enumerate(robots):
#             x, y, w, h = robot['bbox']
#             cx, cy = robot['center']
            
#             # Draw bounding box
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
#             # Draw circle at center
#             cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)
            
#             # Add label
#             label = f"Robot {idx + 1}"
#             cv2.putText(frame, label, (x, y - 10),
#                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
#         # Add frame counter
#         cv2.putText(frame, f"Frame: {frame_count}", (10, 30),
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
#         # Add robot count
#         cv2.putText(frame, f"Robots Detected: {len(robots)}", (10, 70),
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
#         # Write frame to output video
#         out.write(frame)
    
#     # Release resources
#     cap.release()
#     out.release()
#     print(f"Video processing complete. Output saved to: {output_path}")

# if __name__ == "__main__":
#     # Example usage
#     input_video = "BZ-nhrl_dec25_3lb-pinevictus-silentspring-W-11-Cage-2-Overhead-High.mp4"
#     output_video = "output_robots_labeled3.mp4"
    
#     # Process every 10th frame at 30% resolution, ignore edges
#     detect_robots_in_video(input_video, output_video, min_contour_area=280, 
#                           frame_skip=1, margin=50, scale=0.3)



# Use BFMatcher for feature matching
                #     if use_sift and robot_data['desc'] is not None:
                #         bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
                #     else:
                #         bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                    
                #     matches = bf.match(robot_data['desc'], desc_current)
                    
                #     # Score based on number of good matches (normalized)
                #     feature_score = len(matches) / max(len(robot_data['kp']), len(kp_current) + 1e-6)                # Compute features for current detection
                # kp_current, desc_current = orb.detectAndCompute(roi, None)
                
                # # Handle case where descriptors are None
                # if kp_current is None:
                #     kp_current = []
                # if desc_current is None:
                #     desc_current = None
                
                # feature_score = 0
                # if robot_data['desc'] is not None and desc_current is not None and len(kp_current) > 0:                # Debug: uncomment to see feature matching scores
                # # print(f"Robot {robot_id}: {len(matches)} matches, feature_score={feature_score:.3f}, distance={distance:.1f}")
                
                # # Combine distance and feature matching
                # # If we have very few matches, rely more on distance
                # if len(matches) < 5:
                #     match_score = -distance / 50.0  # Use distance only if few matches
                # else:
                #     match_score = feature_score * 10 - (distance / 100.0)  # Boost feature score when we have good matchesimport cv2
import numpy as np
import cv2
from pathlib import Path
from scipy.spatial.distance import euclidean

def detect_robots_in_video(input_path, output_path, min_contour_area=500, max_contour_area=1000, 
                          frame_skip=10, margin=50, scale=0.3):
    """
    Detects robots in a video with static background and outputs labeled video.
    Uses background subtraction on downscaled frames for faster processing.
    Tracks robots by matching against previous detections using template matching.
    
    Args:
        input_path (str): Path to input MP4 video
        output_path (str): Path to output MP4 video with labels
        min_contour_area (int): Minimum contour area to be considered a robot (on scaled frame)
        frame_skip (int): Process every Nth frame (e.g., 10 = process every 10th frame)
        margin (int): Pixel margin from frame edges to ignore detections
        scale (float): Scale factor for processing (0.3 = 30% of original size)
    """
    
    cap = cv2.VideoCapture(input_path)   
    # backsub =cv2.createBackgroundSubtractorMOG2() 
    if not cap.isOpened():
        print(f"Error: Could not open video {input_path}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Scaled dimensions for processing
    scaled_width = int(width * scale)
    scaled_height = int(height * scale)
    scaled_margin = int(margin * scale)
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Initialize background subtractor (MOG2 works well for static backgrounds)
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        detectShadows=False,
        varThreshold=50
    )
    
    # Morphological operations kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    # Initialize ORB detector for rotationally invariant feature matching
    orb = cv2.ORB_create(nfeatures=500, scaleFactor=1.2, nlevels=8)
    
    # Robot tracking: store features and position of each robot
    robot_templates = {}  # {robot_id: {'desc': descriptors, 'kp': keypoints}}
    

    
    frame_count = 0
    lastframe= None
    frames=[]
    
    while True:
        ret, frame = cap.read()
        # frames.append(frame)
        # median_frame= np.median(frames, axis=0).astype(dtype=np.uint8)
        if lastframe is None:
            lastframe=frame
        
        if not ret:## or frame_count>1000:
            break
        
        frame_count += 1
        # fg_mask=backsub.apply(frame)
        
        # Only process every Nth frame
        if frame_count % frame_skip != 0:
            # out.write(frame)
            continue
        
        # frameg= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # dframeg= cv2.cvtColor(median_frame, cv2.COLOR_BGR2GRAY)
        # diff=cv2.absdiff(frameg,dframeg)

        # Downscale frame for faster processing
        small_frame = cv2.resize(frame, (scaled_width, scaled_height))
        
        # Apply background subtraction
        fg_mask = bg_subtractor.apply(small_frame)
        # fg_mask=backsub.apply(frame)
        
        # Apply morphological operations to clean up
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area and get bounding boxes
        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_contour_area and area<max_contour_area:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Check if object is within the valid region (not on outskirts)
                if (x > scaled_margin and y > scaled_margin and 
                    x + w < scaled_width - scaled_margin and y + h < scaled_height - scaled_margin):
                    
                    # Scale coordinates back to original resolution
                    orig_x = int(x / scale)
                    orig_y = int(y / scale)
                    orig_w = int(w / scale)
                    orig_h = int(h / scale)
                    
                    detections.append({
                        'bbox': (orig_x, orig_y, orig_w, orig_h),
                        'center': (orig_x + orig_w // 2, orig_y + orig_h // 2),
                        'area': area,
                        'small_bbox': (x, y, w, h),
                        'small_frame': small_frame,
                        'robot_id': None
                    })
        
        detections.sort(key=lambda d: d["area"])
        # Match detections to known robots
        matched_ids = set()
        print(f"on frame: {frame_count} found {len(detections)} robots", end='\r')
        # for detection in detections:
        #     best_match_id = None
        #     best_match_score = -1
            
        #     cx, cy = detection['center']
            
        #     # Compare with each known robot template using feature matching
        #     for robot_id, robot_data in robot_templates.items():
        #         if robot_id in matched_ids:
        #             continue  # Skip already matched robots
                
        #         # Calculate distance between current detection and last known position
        #         last_cx, last_cy = robot_data['center']
        #         distance = np.sqrt((cx - last_cx)**2 + (cy - last_cy)**2)
                
        #         # Use ORB feature matching for rotation-invariant comparison
        #         sx, sy, sw, sh = detection['small_bbox']
        #         roi = detection['small_frame'][sy:sy+sh, sx:sx+sw].copy()
                
        #         # Compute features for current detection
        #         kp_current, desc_current = orb.detectAndCompute(roi, None)
                
        #         feature_score = 0
        #         if robot_data['desc'] is not None and desc_current is not None and len(kp_current) > 0:
        #             # Use BFMatcher for feature matching
        #             bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        #             matches = bf.match(robot_data['desc'], desc_current)
                    
        #             # Score based on number of good matches (normalized)
        #             feature_score = len(matches) / max(len(robot_data['kp']), len(kp_current) + 1e-6)
                
        #         # Combine distance and feature matching
        #         # Prefer closer robots, but allow features to override if very different
        #         match_score = feature_score - (distance / 100.0)  # Distance less dominant
                
        #         if match_score > best_match_score:
        #             best_match_score = match_score
        #             best_match_id = robot_id
            
        #     # If no good match found, create new robot ID
        #     if best_match_id is None and len(robot_templates) < 10:  # Limit to 10 robots
        #         best_match_id = len(robot_templates) + 1
            
        #     # Always insert/update robot data if we have a valid ID
        #     if best_match_id is not None:
        #         detection['robot_id'] = best_match_id
        #         matched_ids.add(best_match_id)
                
        #         # Extract and store features and position for this robot
        #         sx, sy, sw, sh = detection['small_bbox']
        #         roi = detection['small_frame'][sy:sy+sh, sx:sx+sw].copy()
                
        #         # Enhance contrast to ensure feature detection
        #         clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        #         roi_enhanced = clahe.apply(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)) if len(roi.shape) == 3 else clahe.apply(roi)
                
        #         # Try SIFT first, fall back to ORB
        #         if use_sift:
        #             kp, desc = sift.detectAndCompute(roi_enhanced, None)
        #         else:
        #             kp, desc = orb.detectAndCompute(roi_enhanced, None)
                
        #         # If still None, create dummy descriptors to avoid crashes
        #         if kp is None:
        #             kp = []
        #         if desc is None:
        #             # Create a minimal descriptor array to avoid None issues
        #             desc = np.array([], dtype=np.float32).reshape(0, 128 if use_sift else 32)
                
        #         robot_templates[best_match_id] = {
        #             'kp': kp,
        #             'desc': desc,
        #             'center': detection['center']
        #         }
        
        # # Sort by robot ID for consistent ordering
        # detections.sort(key=lambda d: d['robot_id'] if d['robot_id'] is not None else float('inf'))
        
        # Draw bounding boxes and labels on frame
        for detection in detections:
            if detection['robot_id'] is None:
                x, y, w, h = detection['bbox']
                cx, cy = detection['center']
                robot_id = detection['area']
                
                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Draw circle at center
                cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)
                
                # Add label
                label = f"Robot {robot_id}"
                cv2.putText(frame, label, (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Add frame counter
        cv2.putText(frame, f"Frame: {frame_count}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add robot count
        detected_count = len(matched_ids)
        cv2.putText(frame, f"Robots Detected: {detected_count}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Write frame to output video
        out.write(frame)
        lastframe=frame
    
    # Release resources
    cap.release()
    out.release()
    print(f"Video processing complete. Output saved to: {output_path}")

if __name__ == "__main__":
    # Example usage
    input_video = "BZ-nhrl_dec25_3lb-projectliftoff3-jackalope-470f-Cage-6-Overhead-High.mp4"
    output_video = input_video+"output_robots_labeled3.mp4"
    
    # Process every 10th frame at 30% resolution, ignore edges
    detect_robots_in_video(input_video, output_video, min_contour_area=1220,max_contour_area=50000, 
                          frame_skip=10, margin=50, scale=0.3)