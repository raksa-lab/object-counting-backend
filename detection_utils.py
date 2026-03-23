import cv2
import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple, Any


def preprocess_image(image):
    """
    Enhance image for better object detection.
    
    Args:
        image: Input image (BGR format from OpenCV)
        
    Returns:
        Preprocessed image
    """
    try:
        # Resize if too large (faster processing)
        height, width = image.shape[:2]
        if width > 1920 or height > 1080:
            scale = min(1920/width, 1080/height)
            image = cv2.resize(image, None, fx=scale, fy=scale)
        
        # Convert to LAB color space for better contrast enhancement
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        image = cv2.merge([l, a, b])
        image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)
        
        # Denoise the image
        image = cv2.fastNlMeansDenoisingColored(
            image, 
            None, 
            h=10, 
            templateWindowSize=7, 
            searchWindowSize=21
        )
        
        return image
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return image


def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        box1, box2: [x1, y1, w, h] format
        
    Returns:
        IoU score (0-1)
    """
    try:
        x1_min, y1_min, w1, h1 = box1
        x2_min, y2_min, w2, h2 = box2
        
        x1_max = x1_min + w1
        y1_max = y1_min + h1
        x2_max = x2_min + w2
        y2_max = y2_min + h2
        
        # Calculate intersection
        x_left = max(x1_min, x2_min)
        y_top = max(y1_min, y2_min)
        x_right = min(x1_max, x2_max)
        y_bottom = min(y1_max, y2_max)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    except Exception as e:
        print(f"Error calculating IoU: {e}")
        return 0.0


def apply_nms(detections, iou_threshold=0.5):
    """
    Apply Non-Maximum Suppression to remove duplicate overlapping detections.
    
    Args:
        detections: List of detection dicts with 'bbox' and 'conf' keys
                   bbox format: [x, y, width, height]
        iou_threshold: IoU threshold for suppression (0-1)
        
    Returns:
        Filtered list of detections
    """
    if not detections:
        return []
    
    try:
        # Sort by confidence score (highest first)
        detections = sorted(detections, key=lambda x: x.get('conf', 0), reverse=True)
        kept = []
        
        for current in detections:
            # Check overlap with already kept detections
            is_duplicate = False
            for kept_det in kept:
                iou = calculate_iou(current['bbox'], kept_det['bbox'])
                if iou > iou_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                kept.append(current)
        
        return kept
    except Exception as e:
        print(f"Error applying NMS: {e}")
        return detections


class ObjectTracker:
    """
    Simple object tracker to maintain consistent IDs across frames.
    Reduces false positives by tracking objects over time.
    """
    
    def __init__(self, max_missing_frames=10, distance_threshold=50):
        self.tracks = {}  # {track_id: {bbox, class, frames_missing, detections}}
        self.next_id = 0
        self.max_missing_frames = max_missing_frames
        self.distance_threshold = distance_threshold
    
    @staticmethod
    def _calculate_centroid_distance(box1, box2):
        """Calculate distance between centroids of two bounding boxes."""
        try:
            x1, y1, w1, h1 = box1
            x2, y2, w2, h2 = box2
            
            cx1 = x1 + w1 / 2
            cy1 = y1 + h1 / 2
            cx2 = x2 + w2 / 2
            cy2 = y2 + h2 / 2
            
            distance = np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
            return distance
        except Exception as e:
            print(f"Error calculating centroid distance: {e}")
            return float('inf')
    
    def update(self, detections):
        """
        Update tracks with new detections.
        
        Args:
            detections: List of detection dicts with 'bbox' and 'class' keys
            
        Returns:
            Dict of active tracks: {track_id: detection}
        """
        try:
            matched_detections = set()
            
            # Try to match current detections with existing tracks
            for track_id, track_data in list(self.tracks.items()):
                best_match_idx = None
                best_distance = self.distance_threshold
                
                for idx, detection in enumerate(detections):
                    if idx in matched_detections:
                        continue
                    
                    # Calculate distance between detection and track
                    distance = self._calculate_centroid_distance(
                        track_data['bbox'], 
                        detection['bbox']
                    )
                    
                    # Also check if class matches (optional - comment out for cross-class tracking)
                    if detection.get('class') != track_data.get('class'):
                        distance *= 1.5  # Penalize class mismatch
                    
                    if distance < best_distance:
                        best_match_idx = idx
                        best_distance = distance
                
                if best_match_idx is not None:
                    # Update existing track
                    self.tracks[track_id]['bbox'] = detections[best_match_idx]['bbox']
                    self.tracks[track_id]['conf'] = detections[best_match_idx].get('conf', 0.5)
                    self.tracks[track_id]['frames_missing'] = 0
                    matched_detections.add(best_match_idx)
                else:
                    # Increment missing frame counter
                    self.tracks[track_id]['frames_missing'] += 1
            
            # Remove lost tracks (too many missing frames)
            self.tracks = {
                k: v for k, v in self.tracks.items() 
                if v['frames_missing'] < self.max_missing_frames
            }
            
            # Add new detections as new tracks
            for idx, detection in enumerate(detections):
                if idx not in matched_detections:
                    self.tracks[self.next_id] = {
                        'bbox': detection['bbox'],
                        'class': detection.get('class', 'unknown'),
                        'conf': detection.get('conf', 0.5),
                        'frames_missing': 0,
                        'detections': 1
                    }
                    self.next_id += 1
            
            return self.tracks
        except Exception as e:
            print(f"Error updating tracker: {e}")
            return self.tracks
    
    def get_active_tracks(self):
        """Get only the most confident/stable tracks."""
        return {k: v for k, v in self.tracks.items() 
                if v['frames_missing'] == 0}
    
    def reset(self):
        """Reset the tracker."""
        self.tracks = {}
        self.next_id = 0


def filter_detections_by_confidence(detections, confidence_threshold=0.5):
    """
    Filter detections by confidence score.
    
    Args:
        detections: List of detection dicts with 'conf' key
        confidence_threshold: Minimum confidence score
        
    Returns:
        Filtered detections list
    """
    return [d for d in detections if d.get('conf', 0) >= confidence_threshold]


def merge_overlapping_classes(detections, merge_map=None):
    """
    Merge detections of similar classes (e.g., 'person' and 'people' -> 'person').
    
    Args:
        detections: List of detection dicts with 'class' key
        merge_map: Dict mapping class names to merge targets
                  e.g., {'people': 'person', 'dog': 'cat'} (not recommended for accuracy)
        
    Returns:
        Detections with merged class names
    """
    if merge_map is None:
        merge_map = {}
    
    for detection in detections:
        original_class = detection.get('class', '')
        detection['class'] = merge_map.get(original_class, original_class)
    
    return detections


def count_objects_by_class(detections):
    """
    Count objects grouped by class.
    
    Args:
        detections: List of detection dicts with 'class' key
        
    Returns:
        Dict mapping class names to counts
    """
    counts = {}
    for detection in detections:
        class_name = detection.get('class', 'unknown')
        counts[class_name] = counts.get(class_name, 0) + 1
    return counts


def draw_detections(image, detections, color=(0, 255, 0), thickness=2):
    """
    Draw bounding boxes and labels on image.
    
    Args:
        image: Input image (BGR format)
        detections: List of detection dicts
        color: Box color in BGR format
        thickness: Box line thickness
        
    Returns:
        Annotated image
    """
    try:
        for detection in detections:
            x, y, w, h = detection['bbox']
            x, y, w, h = int(x), int(y), int(w), int(h)
            conf = detection.get('conf', 0)
            class_name = detection.get('class', 'object')
            track_id = detection.get('track_id', '')
            
            # Draw bounding box
            cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)
            
            # Draw label
            label = f"{class_name} {conf:.2f}"
            if track_id:
                label += f" (ID:{track_id})"
            
            cv2.putText(image, label, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return image
    except Exception as e:
        print(f"Error drawing detections: {e}")
        return image
