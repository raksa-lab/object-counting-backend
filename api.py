from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
from ultralytics import YOLO
import tempfile
import os
from detection_utils import (
    preprocess_image, 
    apply_nms, 
    filter_detections_by_confidence,
    count_objects_by_class,
    ObjectTracker,
    draw_detections
)

app = Flask(__name__)
CORS(app)

MAX_INFERENCE_SIDE = 960
YOLO_IMAGE_SIZE = 640

# Load YOLOv8 model - use larger model for better accuracy
try:
    model = YOLO("yolo26n.pt")  # Medium model for better accuracy
    print("✓ Loaded YOLOv8m model")
except:
    model = YOLO("yolov8n.pt")  # Fallback to nano
    print("✓ Loaded YOLOv8n model")

# Initialize object tracker for video processing
tracker = ObjectTracker(max_missing_frames=10, distance_threshold=50)

# Default optimized config for auto-application
DEFAULT_CONFIG = {
    'confidence': 0.55,        # Balanced confidence
    'iou_threshold': 0.45,     # Stricter NMS
    'preprocess': True,        # Always enhance
    'use_nms': True,           # Always remove duplicates
    'use_tracking': False,     # Set per-request
    'min_detection_area': 100, # Minimum bbox area (filters tiny false positives)
}

def decode_image(image_data):
    """Decode base64 image data."""
    try:
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"Error decoding image: {e}")
        return None

def filter_small_detections(detections, min_area=100):
    """Remove detections with very small bounding boxes (noise)."""
    filtered = []
    for detection in detections:
        x, y, w, h = detection['bbox']
        area = w * h
        if area >= min_area:
            filtered.append(detection)
    return filtered

def filter_redundant_detections(detections, class_overlap_threshold=0.7):
    """Remove detections that are likely redundant/duplicate same-class predictions."""
    if not detections:
        return detections
    
    filtered = []
    for i, det1 in enumerate(detections):
        is_duplicate = False
        for det2 in filtered:
            # Check if same class and heavily overlapping
            if det1['class'] == det2['class']:
                from detection_utils import calculate_iou
                iou = calculate_iou(det1['bbox'], det2['bbox'])
                if iou > class_overlap_threshold:
                    # Keep the one with higher confidence
                    if det1['conf'] > det2['conf']:
                        filtered.remove(det2)
                    else:
                        is_duplicate = True
                        break
        
        if not is_duplicate:
            filtered.append(det1)
    
    return filtered

def detect_objects(image, conf_threshold=0.55, iou_threshold=0.45, use_preprocessing=True, use_nms=True, min_area=100):
    """Perform object detection on image with advanced enhancements."""
    try:
        original_image = image.copy()

        # Resize oversized uploads to keep inference responsive on low-memory instances.
        height, width = image.shape[:2]
        longest_side = max(height, width)
        if longest_side > MAX_INFERENCE_SIDE:
            scale = MAX_INFERENCE_SIDE / float(longest_side)
            target_width = int(width * scale)
            target_height = int(height * scale)
            image = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)
            original_image = cv2.resize(original_image, (target_width, target_height), interpolation=cv2.INTER_AREA)
        
        # Preprocess image for better detection
        if use_preprocessing:
            image = preprocess_image(image)
        
        # Run YOLO detection
        results = model(image, conf=conf_threshold, iou=iou_threshold, imgsz=YOLO_IMAGE_SIZE, verbose=False)
        
        detections = []
        
        for r in results:
            # Convert YOLO results to standard format
            for box in r.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0]
                
                detection = {
                    'bbox': [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    'class': model.names[class_id],
                    'conf': confidence,
                    'class_id': class_id
                }
                detections.append(detection)
        
        # Apply multiple filtering stages
        if use_nms and detections:
            detections = apply_nms(detections, iou_threshold=iou_threshold)
        
        # Remove tiny false positives
        detections = filter_small_detections(detections, min_area=min_area)
        
        # Remove redundant same-class detections
        detections = filter_redundant_detections(detections, class_overlap_threshold=0.6)
        
        # Count objects by class
        object_counts = count_objects_by_class(detections)
        
        # Draw detections on original image
        annotated_frame = draw_detections(original_image, detections, color=(0, 255, 0), thickness=2)
        
        return annotated_frame, object_counts, detections
    except Exception as e:
        print(f"Error detecting objects: {e}")
        return None, {}, []

def encode_image(image):
    """Encode image to base64."""
    try:
        _, buffer = cv2.imencode('.jpg', image)
        base64_image = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/jpeg;base64,{base64_image}"
    except Exception as e:
        print(f"Error encoding image: {e}")
        return None

@app.route('/api/config', methods=['GET'])
def get_config():
    """Return optimal detection configuration for frontend to auto-apply."""
    return jsonify({
        'success': True,
        'config': DEFAULT_CONFIG,
        'description': 'Auto-optimized configuration for best detection accuracy'
    })

@app.route('/api/detect', methods=['POST'])
def detect():
    """Detect objects in uploaded image with automatic optimizations."""
    try:
        data = request.get_json()
        image_data = data.get('image')
        
        # Use provided values or fallback to optimal defaults
        conf_threshold = float(data.get('confidence', DEFAULT_CONFIG['confidence']))
        iou_threshold = float(data.get('iou_threshold', DEFAULT_CONFIG['iou_threshold']))
        use_preprocessing = data.get('preprocess', DEFAULT_CONFIG['preprocess'])
        use_nms = data.get('use_nms', DEFAULT_CONFIG['use_nms'])
        min_area = int(data.get('min_area', DEFAULT_CONFIG['min_detection_area']))
        
        if not image_data:
            return jsonify({'error': 'No image provided'}), 400
        
        image = decode_image(image_data)
        if image is None:
            return jsonify({'error': 'Invalid image data'}), 400
        
        annotated_image, object_counts, detections = detect_objects(
            image,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            use_preprocessing=use_preprocessing,
            use_nms=use_nms,
            min_area=min_area
        )
        
        if annotated_image is None:
            return jsonify({'error': 'Detection failed'}), 500
        
        result_image = encode_image(annotated_image)
        
        return jsonify({
            'success': True,
            'image': result_image,
            'counts': object_counts,
            'total': sum(object_counts.values()),
            'detections': [
                {
                    'class': d['class'],
                    'confidence': round(d['conf'], 3),
                    'bbox': [round(v, 2) for v in d['bbox']]
                }
                for d in detections
            ],
            'config_used': {
                'confidence': conf_threshold,
                'iou_threshold': iou_threshold,
                'preprocess': use_preprocessing,
                'use_nms': use_nms,
                'min_area': min_area
            }
        })
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    try:
        return jsonify({
            'status': 'ok',
            'model': 'yolov8m',
            'config': DEFAULT_CONFIG,
            'optimization': 'enabled'
        })
    except:
        return jsonify({'status': 'error'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
