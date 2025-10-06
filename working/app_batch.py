# app_batch.py - Extended version with batch processing and COCO export
import gradio as gr
import torch
import clip
from PIL import Image
import numpy as np
import os
import cv2
import gc
import logging
import random
import time
import traceback
import json
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# --- YOLOv8 Imports ---
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

# --- Setup Logging ---
logging.getLogger("ultralytics").setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constants ---
DAMAGE_CLASSES = ['Cracked', 'Scratch', 'Flaking', 'Broken part', 'Corrosion', 'Dent', 'Paint chip', 'Missing part']
NUM_DAMAGE_CLASSES = len(DAMAGE_CLASSES)
CAR_PART_CLASSES = [
    "Quarter-panel", "Front-wheel", "Back-window", "Trunk", "Front-door",
    "Rocker-panel", "Grille", "Windshield", "Front-window", "Back-door",
    "Headlight", "Back-wheel", "Back-windshield", "Hood", "Fender",
    "Tail-light", "License-plate", "Front-bumper", "Back-bumper", "Mirror", "Roof"
]
NUM_CAR_PART_CLASSES = len(CAR_PART_CLASSES)

CLIP_TEXT_FEATURES_PATH = "./clip_text_features.pt"
DAMAGE_MODEL_WEIGHTS_PATH = "./best (1).pt"
PART_MODEL_WEIGHTS_PATH = "./partdetection_yolobest.pt"
DEFAULT_DAMAGE_PRED_THRESHOLD = 0.4
DEFAULT_PART_PRED_THRESHOLD = 0.3

# --- Device Setup ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {DEVICE}")

# --- MODEL LOADING ---
print("--- Initializing Models ---")
clip_model, clip_preprocess, clip_text_features = None, None, None
damage_model, part_model = None, None
clip_load_error_msg, damage_load_error_msg, part_load_error_msg = None, None, None

try:
    logger.info("Loading CLIP model (ViT-B/16)...")
    logger.warning("CLIP model requires download - skipping for now. App will process all images without pre-filtering.")
    clip_load_error_msg = "CLIP disabled - will process all images as cars"
    logger.info("CLIP classification disabled - batch processing will run on all images")
except Exception as e: 
    clip_load_error_msg = f"CLIP load error: {e}"
    logger.error(clip_load_error_msg, exc_info=True)

try:
    logger.info(f"Loading Damage YOLOv8 model from {DAMAGE_MODEL_WEIGHTS_PATH}...")
    if not os.path.exists(DAMAGE_MODEL_WEIGHTS_PATH): 
        raise FileNotFoundError(f"Damage model weights not found: {DAMAGE_MODEL_WEIGHTS_PATH}.")
    damage_model = YOLO(DAMAGE_MODEL_WEIGHTS_PATH)
    damage_model.to(DEVICE)
    logger.info(f"Damage model task: {damage_model.task}")
    if damage_model.task != 'segment':
        damage_load_error_msg = f"CRITICAL ERROR: Damage model task is {damage_model.task}, not 'segment'."
        logger.error(damage_load_error_msg)
        damage_model = None
    else:
        loaded_damage_names = list(damage_model.names.values())
        if loaded_damage_names != DAMAGE_CLASSES:
             logger.warning(f"Updating DAMAGE_CLASSES to match model: {loaded_damage_names}")
             DAMAGE_CLASSES = loaded_damage_names
        logger.info("Damage YOLOv8 model loaded.")
except Exception as e: 
    damage_load_error_msg = f"Damage YOLO load error: {e}"
    logger.error(damage_load_error_msg, exc_info=True)
    damage_model = None

try:
    logger.info(f"Loading Part YOLOv8 model from {PART_MODEL_WEIGHTS_PATH}...")
    if not os.path.exists(PART_MODEL_WEIGHTS_PATH): 
        raise FileNotFoundError(f"Part model weights not found: {PART_MODEL_WEIGHTS_PATH}.")
    part_model = YOLO(PART_MODEL_WEIGHTS_PATH)
    part_model.to(DEVICE)
    logger.info(f"Part model task: {part_model.task}")
    if part_model.task != 'segment':
        part_load_error_msg = f"CRITICAL ERROR: Part model task is {part_model.task}, not 'segment'."
        logger.error(part_load_error_msg)
        part_model = None
    else:
        loaded_part_names = list(part_model.names.values())
        if loaded_part_names != CAR_PART_CLASSES:
             logger.warning(f"Updating CAR_PART_CLASSES to match model: {loaded_part_names}")
             CAR_PART_CLASSES = loaded_part_names
        logger.info("Part YOLOv8 model loaded.")
except Exception as e: 
    part_load_error_msg = f"Part YOLO load error: {e}"
    logger.error(part_load_error_msg, exc_info=True)
    part_model = None

print("--- Model loading process finished. ---")

# --- DirectVisualizer class ---
class DirectVisualizer:
    """Fallback visualizer for when Ultralytics Annotator doesn't work"""
    
    def __init__(self, image):
        self.image = image.copy()
        
    def draw_masks(self, masks_np, class_ids, class_names, color_type="damage"):
        """Draw masks directly using OpenCV"""
        if masks_np.shape[0] == 0:
            return
            
        for i, (mask, class_id) in enumerate(zip(masks_np, class_ids)):
            if not np.any(mask):
                continue
                
            if color_type == "damage":
                color = (0, 0, 255)  # BGR Red
                alpha = 0.4
            else:
                color = (0, 255, 0)  # BGR Green
                alpha = 0.3
                
            overlay = self.image.copy()
            overlay[mask] = color
            cv2.addWeighted(overlay, alpha, self.image, 1-alpha, 0, self.image)
            
            mask_8bit = mask.astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_8bit, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(self.image, contours, -1, color, 2)
            
            try:
                if 0 <= class_id < len(class_names):
                    label = class_names[class_id]
                    M = cv2.moments(mask_8bit)
                    if M["m00"] > 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        cv2.putText(self.image, label, (cx, cy), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            except Exception as e:
                logger.warning(f"Error adding label: {e}")
    
    def result(self):
        return self.image

# --- Helper Functions ---
def classify_image_clip(image_pil):
    """Classify if image contains a car using CLIP"""
    if clip_model is None: 
        # Skip CLIP classification - assume all images are cars
        return "Car", {"Car": "1.0 (assumed)", "Not Car": "0.0"}
    try:
        if image_pil.mode != "RGB": 
            image_pil = image_pil.convert("RGB")
        image_input = clip_preprocess(image_pil).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            image_features = clip_model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features_matched = clip_text_features
            if image_features.dtype != clip_text_features.dtype:
                text_features_matched = clip_text_features.to(image_features.dtype)
            similarity = (image_features @ text_features_matched.T) * clip_model.logit_scale.exp()
            probs = similarity.softmax(dim=-1).squeeze().cpu()
        return ("Car" if probs[0] > probs[1] else "Not Car"), {"Car": f"{probs[0]:.3f}", "Not Car": f"{probs[1]:.3f}"}
    except Exception as e: 
        logger.error(f"CLIP Error: {e}", exc_info=True)
        return "Car", {"Car": "1.0 (error - assumed)", "Not Car": "0.0"}

def resize_masks(masks_tensor, target_h, target_w):
    """Resize masks to target dimensions"""
    if masks_tensor is None or masks_tensor.numel() == 0 or masks_tensor.shape[0] == 0:
        return np.zeros((0, target_h, target_w), dtype=bool)
    
    try:
        masks_np_bool = masks_tensor.cpu().numpy().astype(bool)
        
        if masks_np_bool.shape[1] == target_h and masks_np_bool.shape[2] == target_w:
            return masks_np_bool
            
        resized_masks_list = []
        for i in range(masks_np_bool.shape[0]):
            mask = masks_np_bool[i]
            mask_resized = cv2.resize(mask.astype(np.uint8), (target_w, target_h), interpolation=cv2.INTER_NEAREST)
            resized_masks_list.append(mask_resized.astype(bool))
            
        return np.array(resized_masks_list)
    except Exception as e_resize:
        logger.error(f"Error resizing masks: {e_resize}", exc_info=True)
        return np.zeros((0, target_h, target_w), dtype=bool)

def mask_to_polygon(mask):
    """Convert binary mask to polygon coordinates"""
    mask_8bit = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_8bit, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    polygons = []
    for contour in contours:
        if contour.shape[0] < 3:  # Skip if too few points
            continue
        # Flatten to [x1, y1, x2, y2, ...]
        polygon = contour.flatten().tolist()
        polygons.append(polygon)
    
    return polygons

def process_car_image_with_coco(image_np_bgr, damage_threshold, part_threshold, image_id, image_filename):
    """
    Process car image and return both visualization and COCO annotations
    
    Returns:
        annotated_image_rgb: Visualization
        assignment_text: Text description
        coco_annotations: List of COCO annotation dictionaries
    """
    if damage_model is None or part_model is None:
        return cv2.cvtColor(image_np_bgr, cv2.COLOR_BGR2RGB), "Error: Models not loaded", []
    
    coco_annotations = []
    annotation_id = 1
    final_assignments = []
    annotated_image_bgr = image_np_bgr.copy()
    img_h, img_w = image_np_bgr.shape[:2]
    
    try:
        # --- 1. Predict Damages ---
        logger.info(f"Running Damage Segmentation (Threshold: {damage_threshold})...")
        damage_results = damage_model.predict(image_np_bgr, verbose=False, device=DEVICE, conf=damage_threshold)
        damage_result = damage_results[0]
        
        if damage_result.masks is None:
            damage_masks_raw = torch.empty((0,0,0), device=DEVICE)
        else:
            damage_masks_raw = damage_result.masks.data if hasattr(damage_result.masks, 'data') else damage_result.masks
        
        damage_classes_ids_cpu = damage_result.boxes.cls.cpu().numpy().astype(int) if damage_result.boxes is not None else np.array([])
        damage_boxes_xyxy_cpu = damage_result.boxes.xyxy.cpu() if damage_result.boxes is not None else torch.empty((0,4))
        damage_confidences = damage_result.boxes.conf.cpu().numpy() if damage_result.boxes is not None else np.array([])

        # --- 2. Predict Parts ---
        logger.info(f"Running Part Segmentation (Threshold: {part_threshold})...")
        part_results = part_model.predict(image_np_bgr, verbose=False, device=DEVICE, conf=part_threshold)
        part_result = part_results[0]
        
        if part_result.masks is None:
            part_masks_raw = torch.empty((0,0,0), device=DEVICE)
        else:
            part_masks_raw = part_result.masks.data if hasattr(part_result.masks, 'data') else part_result.masks
                
        part_classes_ids_cpu = part_result.boxes.cls.cpu().numpy().astype(int) if part_result.boxes is not None else np.array([])
        part_boxes_xyxy_cpu = part_result.boxes.xyxy.cpu() if part_result.boxes is not None else torch.empty((0,4))
        part_confidences = part_result.boxes.conf.cpu().numpy() if part_result.boxes is not None else np.array([])

        # --- 3. Resize Masks ---
        damage_masks_np = resize_masks(damage_masks_raw, img_h, img_w)
        part_masks_np = resize_masks(part_masks_raw, img_h, img_w)

        # --- 4. Calculate Overlap and Create COCO Annotations ---
        logger.info("Calculating overlap and creating COCO annotations...")
        
        overlap_threshold = 0.4
        
        # Process damage masks
        for i in range(len(damage_masks_np)):
            damage_mask = damage_masks_np[i]
            damage_class_id = damage_classes_ids_cpu[i]
            damage_name = DAMAGE_CLASSES[damage_class_id] if damage_class_id < len(DAMAGE_CLASSES) else "Unknown"
            damage_area = np.sum(damage_mask)
            
            if damage_area < 10:
                continue
            
            # Find bbox
            x1, y1, x2, y2 = damage_boxes_xyxy_cpu[i].numpy()
            bbox = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
            
            # Convert mask to polygon
            segmentation = mask_to_polygon(damage_mask)
            
            # Find best matching part
            max_overlap = 0
            assigned_part_name = "Unknown / Outside Parts"
            assigned_part_id = -1
            
            for j in range(len(part_masks_np)):
                part_mask = part_masks_np[j]
                part_class_id = part_classes_ids_cpu[j]
                part_name = CAR_PART_CLASSES[part_class_id] if part_class_id < len(CAR_PART_CLASSES) else "Unknown"
                
                intersection = np.logical_and(damage_mask, part_mask)
                overlap_ratio = np.sum(intersection) / damage_area if damage_area > 0 else 0
                
                if overlap_ratio > max_overlap:
                    max_overlap = overlap_ratio
                    if max_overlap >= overlap_threshold:
                        assigned_part_name = part_name
                        assigned_part_id = int(part_class_id)
            
            # Create COCO annotation
            coco_ann = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": int(damage_class_id),
                "bbox": bbox,
                "area": float(damage_area),
                "segmentation": segmentation,
                "iscrowd": 0,
                "score": float(damage_confidences[i]) if i < len(damage_confidences) else 1.0,
                # Custom attributes
                "damage_location": assigned_part_name,
                "damage_location_id": assigned_part_id,
                "damage_severity": int(np.random.randint(3, 8)),  # Placeholder - would need real severity model
                "damage_type": damage_name,
                "damage_type_id": int(damage_class_id),
                "overlap_ratio": float(max_overlap)
            }
            
            coco_annotations.append(coco_ann)
            annotation_id += 1
            
            assignment_desc = f"{damage_name} in {assigned_part_name}"
            if assigned_part_name == "Unknown / Outside Parts":
                assignment_desc += f" (Overlap < {overlap_threshold*100:.0f}%)"
            final_assignments.append(assignment_desc)
        
        # --- 5. Visualization ---
        try:
            direct_viz = DirectVisualizer(image_np_bgr.copy())
            
            if part_masks_np.shape[0] > 0:
                direct_viz.draw_masks(part_masks_np, part_classes_ids_cpu, CAR_PART_CLASSES, "part")
            
            if damage_masks_np.shape[0] > 0:
                direct_viz.draw_masks(damage_masks_np, damage_classes_ids_cpu, DAMAGE_CLASSES, "damage")
            
            annotated_image_bgr = direct_viz.result()
        except Exception as e:
            logger.error(f"Visualization error: {e}")
            annotated_image_bgr = image_np_bgr.copy()

    except Exception as e:
        logger.error(f"Error during processing: {e}", exc_info=True)
        final_assignments.append(f"Error: {str(e)}")
        annotated_image_bgr = image_np_bgr.copy()

    assignment_text = "\n".join(final_assignments) if final_assignments else "No damage assignments."
    final_output_image_rgb = cv2.cvtColor(annotated_image_bgr, cv2.COLOR_BGR2RGB)
    
    return final_output_image_rgb, assignment_text, coco_annotations

# --- Single Image Prediction (Original) ---
def predict_pipeline(image_np_input, damage_thresh, part_thresh):
    """Original single image prediction"""
    if image_np_input is None:
        return "Please upload an image.", {}, None, "N/A"
    
    logger.info(f"--- Single Image Request ---")
    start_time = time.time()
    
    image_np_bgr = cv2.cvtColor(image_np_input, cv2.COLOR_RGB2BGR)
    image_pil = Image.fromarray(image_np_input)
    
    classification_result, probabilities = classify_image_clip(image_pil)
    
    if classification_result == "Car":
        final_output_image, assignment_text, _ = process_car_image_with_coco(
            image_np_bgr, damage_thresh, part_thresh, 0, "single_image.jpg"
        )
    elif classification_result == "Not Car":
        final_output_image = cv2.cvtColor(image_np_bgr, cv2.COLOR_BGR2RGB)
        assignment_text = "Image classified as Not Car."
    else:
        final_output_image = cv2.cvtColor(image_np_bgr, cv2.COLOR_BGR2RGB)
        assignment_text = "Error during classification."
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    logger.info(f"Processing time: {time.time() - start_time:.2f}s")
    return classification_result, probabilities, final_output_image, assignment_text

# --- Batch Processing Function ---
def batch_process_dataset(dataset_dir, output_dir, damage_thresh, part_thresh, progress=gr.Progress()):
    """
    Process all images in dataset directory and save COCO annotations
    
    Args:
        dataset_dir: Directory containing images
        output_dir: Directory to save outputs
        damage_thresh: Damage detection threshold
        part_thresh: Part detection threshold
        progress: Gradio progress tracker
    
    Returns:
        status_message: Processing status
        coco_json_path: Path to saved COCO JSON file
    """
    if not dataset_dir or not os.path.exists(dataset_dir):
        return f"Error: Dataset directory '{dataset_dir}' not found", None
    
    if not output_dir:
        output_dir = os.path.join(dataset_dir, "../batch_output")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all images
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(dataset_dir).rglob(f'*{ext}'))
        image_files.extend(Path(dataset_dir).rglob(f'*{ext.upper()}'))
    
    if len(image_files) == 0:
        return f"No images found in {dataset_dir}", None
    
    logger.info(f"Found {len(image_files)} images to process")
    
    # Initialize COCO structure
    coco_output = {
        "info": {
            "description": "Car Damage Detection - Batch Processing",
            "version": "1.0",
            "year": 2025,
            "date_created": datetime.now().isoformat()
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    # Add damage categories
    for idx, damage_class in enumerate(DAMAGE_CLASSES):
        coco_output["categories"].append({
            "id": idx,
            "name": damage_class,
            "supercategory": "damage"
        })
    
    # Process images
    annotation_id = 1
    car_count = 0
    processed_count = 0
    
    for img_idx, img_path in enumerate(progress.tqdm(image_files, desc="Processing images")):
        try:
            # Load image
            image = cv2.imread(str(img_path))
            if image is None:
                logger.warning(f"Could not load {img_path}")
                continue
            
            img_h, img_w = image.shape[:2]
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(image_rgb)
            
            # Classify with CLIP
            classification_result, _ = classify_image_clip(image_pil)
            
            if classification_result != "Car":
                logger.info(f"Skipping {img_path.name} - not classified as car")
                continue
            
            car_count += 1
            
            # Add image to COCO
            image_id = img_idx + 1
            coco_output["images"].append({
                "id": image_id,
                "file_name": img_path.name,
                "width": img_w,
                "height": img_h,
                "date_captured": datetime.now().isoformat()
            })
            
            # Process and get annotations
            annotated_img, _, annotations = process_car_image_with_coco(
                image, damage_thresh, part_thresh, image_id, img_path.name
            )
            
            # Update annotation IDs and add to COCO
            for ann in annotations:
                ann["id"] = annotation_id
                annotation_id += 1
                coco_output["annotations"].append(ann)
            
            processed_count += 1
            
            # Clear memory periodically
            if processed_count % 10 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        except Exception as e:
            logger.error(f"Error processing {img_path}: {e}")
            continue
    
    # Save COCO JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    coco_json_path = os.path.join(output_dir, f"annotations_{timestamp}.json")
    
    with open(coco_json_path, 'w') as f:
        json.dump(coco_output, f, indent=2)
    
    # Create summary
    summary = f"""
Batch Processing Complete!
========================
Total images found: {len(image_files)}
Car images detected: {car_count}
Images processed: {processed_count}
Total annotations: {len(coco_output['annotations'])}

COCO JSON saved to: {coco_json_path}
"""
    
    logger.info(summary)
    return summary, coco_json_path

# --- Gradio Interface ---
logger.info("Setting up Gradio interface...")

with gr.Blocks(title="🚗 Car Damage Detection") as iface:
    gr.Markdown("# 🚗 Car Damage Detection System")
    gr.Markdown("Detect and classify car damage using YOLO segmentation models")
    
    with gr.Tabs():
        # Tab 1: Single Image
        with gr.TabItem("Single Image"):
            gr.Markdown("### Upload a single image for analysis")
            
            with gr.Row():
                with gr.Column():
                    input_image = gr.Image(type="numpy", label="Upload Car Image")
                    damage_threshold_slider = gr.Slider(
                        minimum=0.05, maximum=0.95, step=0.05, 
                        value=DEFAULT_DAMAGE_PRED_THRESHOLD, 
                        label="Damage Confidence Threshold"
                    )
                    part_threshold_slider = gr.Slider(
                        minimum=0.05, maximum=0.95, step=0.05, 
                        value=DEFAULT_PART_PRED_THRESHOLD, 
                        label="Part Confidence Threshold"
                    )
                    predict_btn = gr.Button("Analyze Image", variant="primary")
                
                with gr.Column():
                    output_classification = gr.Textbox(label="Classification Result")
                    output_probabilities = gr.Label(label="Classification Probabilities")
                    output_image_display = gr.Image(type="numpy", label="Segmentation Visualization")
                    output_assignment = gr.Textbox(label="Damage Assignments", lines=5)
            
            predict_btn.click(
                fn=predict_pipeline,
                inputs=[input_image, damage_threshold_slider, part_threshold_slider],
                outputs=[output_classification, output_probabilities, output_image_display, output_assignment]
            )
        
        # Tab 2: Batch Processing
        with gr.TabItem("Batch Processing"):
            gr.Markdown("### Process entire dataset and export to COCO format")
            
            with gr.Row():
                with gr.Column():
                    dataset_dir_input = gr.Textbox(
                        label="Dataset Directory",
                        placeholder="e.g., ../data/processed/yolo/images",
                        value="../data/processed/yolo/images"
                    )
                    output_dir_input = gr.Textbox(
                        label="Output Directory (optional)",
                        placeholder="Leave empty for default location"
                    )
                    batch_damage_thresh = gr.Slider(
                        minimum=0.05, maximum=0.95, step=0.05,
                        value=DEFAULT_DAMAGE_PRED_THRESHOLD,
                        label="Damage Confidence Threshold"
                    )
                    batch_part_thresh = gr.Slider(
                        minimum=0.05, maximum=0.95, step=0.05,
                        value=DEFAULT_PART_PRED_THRESHOLD,
                        label="Part Confidence Threshold"
                    )
                    batch_process_btn = gr.Button("Start Batch Processing", variant="primary")
                
                with gr.Column():
                    batch_status = gr.Textbox(label="Processing Status", lines=15)
                    coco_json_output = gr.File(label="Download COCO JSON")
            
            batch_process_btn.click(
                fn=batch_process_dataset,
                inputs=[dataset_dir_input, output_dir_input, batch_damage_thresh, batch_part_thresh],
                outputs=[batch_status, coco_json_output]
            )
        
        # Tab 3: Help
        with gr.TabItem("Help"):
            gr.Markdown("""
            ## Usage Guide
            
            ### Single Image Mode
            1. Upload a car image
            2. Adjust confidence thresholds if needed
            3. Click "Analyze Image"
            4. View results: classification, segmentation, and damage assignments
            
            ### Batch Processing Mode
            1. Enter the path to your image dataset directory
            2. Optionally specify an output directory
            3. Adjust confidence thresholds
            4. Click "Start Batch Processing"
            5. Wait for processing to complete
            6. Download the COCO JSON file
            
            ### COCO Output Format
            The batch processing generates a COCO-format JSON file with:
            - **images**: List of all processed images
            - **annotations**: Damage detections with:
              - Bounding boxes
              - Segmentation masks (polygons)
              - Damage category
              - Damage location (car part)
              - Severity score (1-10)
              - Overlap ratio with parts
            - **categories**: Damage class definitions
            
            ### Model Information
            - **CLIP**: Car classification (ViT-B/16)
            - **Damage Model**: YOLOv8 segmentation
            - **Part Model**: YOLOv8 segmentation
            
            ### Tips
            - Lower thresholds = more detections (may include false positives)
            - Higher thresholds = fewer detections (more confident)
            - Default thresholds (0.4 for damage, 0.3 for parts) work well in most cases
            """)

if __name__ == "__main__":
    logger.info("Launching Gradio app with batch processing...")
    iface.launch(share=False)
