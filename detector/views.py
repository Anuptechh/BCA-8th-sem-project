# ...existing code...
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import logging
import torch
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
import base64
from ultralytics import YOLO

from django.conf import settings
import os

logger = logging.getLogger(__name__)

# Global variables for model management
# (Forced reload to clear template cache)
model = None
current_model_name = 'pretrained'

def load_model(model_type='pretrained'):
    global model, current_model_name
    try:
        if model_type == 'pretrained':
            logger.info("Loading pretrained YOLOv5s model...")
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
            current_model_name = 'pretrained'
            logger.info("Pretrained model loaded successfully")

        elif model_type == 'vehicle':
            vehicle_model_path = os.path.join(settings.BASE_DIR, 'models', 'vehicle_model.pt')
            if os.path.exists(vehicle_model_path):
                logger.info(f"Loading vehicle model from {vehicle_model_path}...")
                try:
                    # Use Ultralytics (YOLOv8) for vehicle model
                    model = YOLO(vehicle_model_path)
                    current_model_name = 'vehicle'
                    logger.info("Vehicle model loaded successfully (YOLOv8)")
                except Exception as e:
                    logger.error(f"FATAL ERROR loading vehicle model: {e}")
                    print(f"FATAL ERROR loading vehicle model: {e}")
                    return False
            else:
                logger.error(f"Vehicle model not found at {vehicle_model_path}")
                return False
        
        # Set default confidence threshold
        if model:
            model.conf = 0.25  # Reasonable default
            logger.info(f"Model confidence set to {model.conf}")
        return True
    except Exception as e:
        logger.error(f"Error loading model: {e}", exc_info=True)
        return False
        

# Initialize with pretrained model
load_model('pretrained')

def index(request):
    # Get custom model info
    vehicle_model_path = os.path.join(settings.BASE_DIR, 'models', 'vehicle_model.pt')
    context = {
        'current_model': current_model_name,
        'vehicle_exists': os.path.exists(vehicle_model_path)
    }
    return render(request, 'detector/index.html', context)

# ...existing code...
def about_view(request):
    return render(request, 'about.html')

def anup_view(request):
    return render(request, 'anup.html')

def saurav_view(request):
    return render(request, 'saurav.html')
# ...existing code...

@csrf_exempt  # remove for production and send CSRF token from client instead
def detect(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST allowed'}, status=405)

    img_data = request.POST.get('image')
    if not img_data:
        return JsonResponse({'error': 'No image data provided under the key "image"'}, status=400)

    try:
        header, base64_data = img_data.split(';base64,')
        image_bytes = base64.b64decode(base64_data)
        image = Image.open(BytesIO(image_bytes)).convert('RGB')
        frame = np.array(image)

        # Hybrid Inference Handling
        detections = []
        
        if current_model_name == 'vehicle':
            # --- YOLOv8 (Ultralytics) Logic ---
            # Run inference
            results = model(frame, conf=0.25, iou=0.45, verbose=False)
            
            # Check for detections
            if len(results[0].boxes) == 0:
                 return JsonResponse({'no_detection': True, 'message': 'No vehicles detected.'})
            
            # Parse results
            for box in results[0].boxes:
                xmin, ymin, xmax, ymax = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                label = getattr(model.names, 'get', lambda x: str(x))(cls_id) if hasattr(model, 'names') else str(cls_id)
                if isinstance(model.names, dict):
                    label = model.names[cls_id]
                
                detections.append({
                    'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax,
                    'label': label,
                    'conf': conf,
                    'class_id': cls_id
                })

        else:
            # --- YOLOv5 (Torch Hub) Logic ---
            model.conf = 0.25  
            model.iou = 0.45   
            results = model(frame)
            
            if len(results.xyxy[0]) == 0:
                return JsonResponse({'no_detection': True, 'message': 'No objects detected.'})

            # Parse results
            df = results.pandas().xyxy[0]
            for _, row in df.iterrows():
                detections.append({
                    'xmin': int(row['xmin']), 'ymin': int(row['ymin']), 
                    'xmax': int(row['xmax']), 'ymax': int(row['ymax']),
                    'label': row['name'],
                    'conf': float(row['confidence']),
                    'class_id': int(row['class'])
                })

        # Convert RGB to BGR for drawing
        img_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Calculate adaptive thickness/font
        height, width = img_bgr.shape[:2]
        min_dim = min(height, width)
        box_thickness = max(1, int(min_dim * 0.002))
        font_scale = max(0.4, min_dim * 0.0008)
        text_thickness = max(1, int(min_dim * 0.001))

        # Colors
        COLORS = [
            (0, 0, 255), (255, 0, 0), (0, 255, 0), (0, 0, 0),
            (255, 255, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)
        ]

        # Draw Detections from standardized list
        final_detections_for_json = []

        for det in detections:
            xmin, ymin, xmax, ymax = det['xmin'], det['ymin'], det['xmax'], det['ymax']
            label = det['label']
            conf = det['conf']
            class_id = det['class_id']
            
            color = COLORS[class_id % len(COLORS)]

            # Draw Box
            cv2.rectangle(img_bgr, (xmin, ymin), (xmax, ymax), color, box_thickness)

            # Draw Label
            label_text = f"{label} {conf:.2f}"
            (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)
            label_ymin = max(ymin, h + 5)
            cv2.rectangle(img_bgr, (xmin, label_ymin - h - 5), (xmin + w, label_ymin + 5), color, -1)
            
            avg_color = sum(color) / 3
            text_color = (255, 255, 255) if avg_color < 128 else (0, 0, 0)
            cv2.putText(img_bgr, label_text, (xmin, label_ymin), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, text_thickness)

            final_detections_for_json.append({
                'name': label,
                'confidence': conf
            })

        _, img_encoded = cv2.imencode('.jpg', img_bgr)
        img_base64 = base64.b64encode(img_encoded).decode('utf-8')
        return JsonResponse({
            'result_image': f'data:image/jpeg;base64,{img_base64}',
            'detections': final_detections_for_json
        })
    except Exception as e:
        logger.exception("Detection failed")
        return JsonResponse({'error': str(e)}, status=500)

def admin_panel_view(request):
    vehicle_model_path = os.path.join(settings.BASE_DIR, 'models', 'vehicle_model.pt')
    context = {
        'current_model': current_model_name,
        'vehicle_exists': os.path.exists(vehicle_model_path)
    }
    return render(request, 'detector/admin_panel.html', context)

@csrf_exempt
def set_model(request):
    global model, current_model_name
    if request.method == 'POST':
        model_type = request.POST.get('model_type')

        # Accept all supported models exposed in the admin panel
        valid_models = ['pretrained', 'vehicle']
        if model_type not in valid_models:
            return JsonResponse({'success': False, 'error': 'Invalid model type'})

        # Load the model
        if load_model(model_type):
            logger.info(f"Successfully switched to {model_type} model")
            return JsonResponse({'success': True, 'current_model': current_model_name})
        else:
            return JsonResponse({'success': False, 'error': 'Failed to load model. Check if the selected model file exists.'})
    return JsonResponse({'success': False, 'error': 'Invalid request method'})
# ...existing code...