import cv2
import numpy as np

# Load YOLO files
model_config = "./yolo/yolov4.cfg"
model_weights = "./yolo/yolov4.weights"
class_names_file = "./yolo/coco.names"

# Load class names
with open(class_names_file, "r") as f:
    class_names = [line.strip() for line in f.readlines()]
print("Loaded Classes:", class_names)

# Initialize YOLO
net = cv2.dnn.readNetFromDarknet(model_config, model_weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Access OBS Virtual Camera
cap = cv2.VideoCapture(1)  # Use the correct index for your OBS Virtual Camera

# YOLO Parameters
conf_threshold = 0.3  # Confidence threshold
nms_threshold = 0.4   # Non-Max Suppression threshold

output_layers = net.getUnconnectedOutLayersNames()

# Define HSV range for red color
lower_red1 = np.array([0, 70, 50])   # Lower hue range
upper_red1 = np.array([10, 255, 255])

lower_red2 = np.array([170, 70, 50]) # Higher hue range (for wrap-around)
upper_red2 = np.array([180, 255, 255])

# Main Loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to grab frame.")
        break

    # Prepare the frame for YOLO
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    h, w = frame.shape[:2]
    boxes = []
    confidences = []
    class_ids = []

    # Process YOLO outputs
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Only consider "car" detections
            if confidence > conf_threshold and class_names[class_id] == "car":
                center_x, center_y, bw, bh = (
                    int(detection[0] * w),
                    int(detection[1] * h),
                    int(detection[2] * w),
                    int(detection[3] * h),
                )
                x, y = int(center_x - bw / 2), int(center_y - bh / 2)
                boxes.append([x, y, bw, bh])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Max Suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    # Red car counter
    red_car_count = 0

    if len(indices) > 0:
        for i in indices.flatten():
            x, y, bw, bh = boxes[i]
            car_roi = frame[y:y + bh, x:x + bw]  # Extract the region of interest (car region)

            # Convert the ROI to HSV
            hsv_roi = cv2.cvtColor(car_roi, cv2.COLOR_BGR2HSV)

            # Create masks for red color
            mask1 = cv2.inRange(hsv_roi, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv_roi, lower_red2, upper_red2)
            red_mask = cv2.bitwise_or(mask1, mask2)

            # Calculate the percentage of red pixels
            red_ratio = cv2.countNonZero(red_mask) / (car_roi.size / 3)

            # If the red ratio exceeds a threshold, classify as a red car
            if red_ratio > 0.2:  # Adjust this threshold as needed
                cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 0, 255), 2)
                cv2.putText(frame, "Red Car", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                red_car_count += 1

    # Display the red car count on the video
    cv2.putText(frame, f"Red Cars: {red_car_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show the video with detections
    cv2.imshow("YOLOv4 Red Car Detection", frame)

    # Quit with 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
