from ultralytics import YOLO
import cv2
import os
import numpy as np

# --- YOLOv8 Model Integration --- #
def load_yolov8_model(model_path):
    return YOLO(model_path)

def detect_beads_yolov8(model, image_path):
    results = model(image_path)
    bead_detections = []
    for r in results:
        # r.boxes contains detected bounding boxes
        for box in r.boxes:
            x1, y1, x2, y2 = [round(x) for x in box.xyxy[0].tolist()] # Pixel coordinates
            conf = round(box.conf[0].item(), 2)
            cls = int(box.cls[0].item())
            class_name = model.names[cls]
            
            if class_name == 'bead': # Ensure it's a bead detection
                bead_detections.append({
                    'bbox': (x1, y1, x2 - x1, y2 - y1), # (x, y, width, height)
                    'confidence': conf
                })
    return bead_detections, results # Return results object for plotting

def detect_gap_breaks_from_yolo(bead_detections, image_width, image_height):
    gap_breaks = []
    
    # Sort beads by x-coordinate to check for gaps in sequence
    bead_detections.sort(key=lambda b: b['bbox'][0])

    for i in range(len(bead_detections) - 1):
        bbox1 = bead_detections[i]['bbox']
        bbox2 = bead_detections[i+1]['bbox']
        
        # Calculate horizontal distance between end of bbox1 and start of bbox2
        horizontal_gap = bbox2[0] - (bbox1[0] + bbox1[2])
        
        # Define a threshold for what constitutes a significant gap
        if horizontal_gap > 50: # Example threshold in pixels
            # Approximate bounding box for the gap
            gap_x = bbox1[0] + bbox1[2]
            gap_y = min(bbox1[1], bbox2[1]) # Take the higher y-coordinate for the top of the gap
            gap_width = horizontal_gap
            gap_height = max(bbox1[3], bbox2[3]) # Take the larger height for the gap
            gap_breaks.append((gap_x, gap_y, gap_width, gap_height))
    return gap_breaks

def classify_quality_and_anomalies_yolov8(bead_detections, gap_breaks):
    overall_quality = "Good"
    anomalies = []

    if not bead_detections:
        overall_quality = "Poor - No beads detected"
        anomalies.append("No beads detected by YOLO")
    
    if gap_breaks:
        overall_quality = "Poor - Gaps detected"
        anomalies.append(f"Detected {len(gap_breaks)} potential gap breaks")

    # Further analysis based on bead_detections (e.g., count, size uniformity, distribution)
    # Example: Check for too few beads
    # This threshold should be based on expected number of beads in a 'good' image
    expected_min_beads = 8 # Example threshold
    if len(bead_detections) < expected_min_beads:
        overall_quality = "Needs Review - Too few beads"
        anomalies.append(f"Fewer beads ({len(bead_detections)}) than expected (min {expected_min_beads})")

    # Example: Check for low confidence detections
    low_conf_beads = [b for b in bead_detections if b["confidence"] < 0.7]
    if low_conf_beads:
        overall_quality = "Needs Review - Low confidence detections"
        anomalies.append(f"Detected {len(low_conf_beads)} beads with low confidence")

    # Example: Check for large variations in bead size (assuming beads should be uniform)
    if len(bead_detections) > 1:
        bead_areas = [b["bbox"][2] * b["bbox"][3] for b in bead_detections]
        if bead_areas:
            std_dev_area = np.std(bead_areas)
            mean_area = np.mean(bead_areas)
            # If standard deviation is high relative to mean, it indicates size variation
            if mean_area > 0 and std_dev_area / mean_area > 0.3: # Example threshold
                overall_quality = "Needs Review - Bead size variation"
                anomalies.append("Significant variation in bead sizes")

    return overall_quality, anomalies

# --- Main Execution for YOLOv8 Based Analysis --- #
if __name__ == '__main__':
    yolo_model_path = './runs/detect/train/weights/best.pt' # Replace with the actual path to your trained YOLOv8 model
    input_image_path = 'cat.png' # Example image for inference

    # Ensure ultralytics is installed for this script to run
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Error: 'ultralytics' package not found. Please install it using 'pip install ultralytics'.")
        exit()

    if not os.path.exists(yolo_model_path):
        print(f"Warning: YOLOv8 model file \'{yolo_model_path}\' not found. Attempting to load a pre-trained 'yolov8n.pt'.")
        # If you want to force using your trained model, remove this 'else' block
        # and ensure yolo_model_path points to your trained model.
        try:
            yolo_model = YOLO('yolov8n.pt') # Load a pre-trained nano model
            print("Loaded pre-trained yolov8n.pt for demonstration.")
        except Exception as e:
            print(f"Error loading pre-trained yolov8n.pt: {e}")
            print("Please ensure you have an internet connection or a local 'yolov8n.pt' file, or train your custom model.")
            exit()
    else:
        yolo_model = load_yolov8_model(yolo_model_path)
        print(f"Loaded custom trained model from {yolo_model_path}.")

    if not os.path.exists(input_image_path):
        print(f"Error: Input image \'{input_image_path}\' not found.")
        exit()
    
    # Perform bead detection
    bead_detections, results_obj = detect_beads_yolov8(yolo_model, input_image_path)
    print(f"Detected {len(bead_detections)} beads in {input_image_path}")

    # Get image dimensions for gap break detection
    img_original = cv2.imread(input_image_path)
    if img_original is None:
        raise FileNotFoundError(f"Original image not found at {input_image_path}")
    img_height, img_width, _ = img_original.shape

    # Perform gap break detection
    gap_breaks = detect_gap_breaks_from_yolo(bead_detections, img_width, img_height)

    # Classify quality and anomalies
    overall_quality, anomalies = classify_quality_and_anomalies_yolov8(bead_detections, gap_breaks)
    print(f"Overall Quality: {overall_quality}")
    if anomalies:
        print(f"Anomalies: {', '.join(anomalies)}")

    # Visualize results
    # YOLOv8 results object has a convenient plot() method
    # This will draw bounding boxes and labels from YOLO detections
    img_with_detections = results_obj[0].plot()

    # Optionally, draw gap breaks if you have a visualization for them
    for gb_bbox in gap_breaks:
        x, y, w, h = gb_bbox
        cv2.rectangle(img_with_detections, (x, y), (x + w, y + h), (0, 255, 255), 2) # Yellow for gaps

    output_filename = f"yolov8_detected_{os.path.basename(input_image_path)}"
    cv2.imwrite(output_filename, img_with_detections)
    print(f"Detection visualization saved to {output_filename}")


