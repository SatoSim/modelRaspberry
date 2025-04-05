import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter

# ====== CONFIGURATION ======
IMAGE_PATH = "image.jpg"  # Replace with your test image
MODEL_PATH = "best_float32.tflite"
CONFIDENCE_THRESHOLD = 0.1  # Lowered to catch low-confidence detections
IOU_THRESHOLD = 0.45
INPUT_SIZE = 640
DEBUG_MODE = True  # Set to True to print and draw all predictions

# ====== Load TFLite model ======
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ====== Load and preprocess image ======
image = cv2.imread(IMAGE_PATH)
if image is None:
    print(f"❌ Error: Could not load image '{IMAGE_PATH}'")
    exit()

original_h, original_w = image.shape[:2]
resized = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE))
input_tensor = np.expand_dims(resized, axis=0).astype(np.float32) / 255.0

# ====== Run inference ======
interpreter.set_tensor(input_details[0]['index'], input_tensor)
interpreter.invoke()
output = interpreter.get_tensor(output_details[0]['index'])[0]

print(f"✅ Model ran inference. Output shape: {output.shape}")
if DEBUG_MODE and len(output) == 0:
    print("⚠️ No predictions returned!")

# ====== Parse predictions ======
boxes, confidences, class_ids = [], [], []

for idx, pred in enumerate(output):
    if len(pred) < 6:
        print(f"⚠️ Skipping invalid prediction #{idx}: {pred}")
        continue

    x_center, y_center, w, h = pred[:4]
    obj_conf = pred[4]
    class_scores = pred[5:]

    cls = np.argmax(class_scores)
    class_conf = class_scores[cls]
    conf = obj_conf * class_conf

    if DEBUG_MODE:
        print(f"[{idx}] Raw box: {pred[:4]} | obj_conf: {obj_conf:.2f} | class_conf: {class_conf:.2f} | final_conf: {conf:.2f} | class: {cls}")

    # Optional: skip low-confidence predictions
    if not DEBUG_MODE and conf < CONFIDENCE_THRESHOLD:
        continue

    # Convert to box coordinates
    x = int((x_center - w / 2) * original_w / INPUT_SIZE)
    y = int((y_center - h / 2) * original_h / INPUT_SIZE)
    width = int(w * original_w / INPUT_SIZE)
    height = int(h * original_h / INPUT_SIZE)

    boxes.append([x, y, width, height])
    confidences.append(float(conf))
    class_ids.append(int(cls))

# ====== Apply Non-Max Suppression ======
indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, IOU_THRESHOLD)

# ====== Draw Boxes ======
if len(indices) == 0:
    print("⚠️ No boxes drawn after NMS.")

for i in indices:
    i = i[0] if isinstance(i, (list, tuple, np.ndarray)) else i
    x, y, w, h = boxes[i]
    cls_id = class_ids[i]
    conf = confidences[i]

    label = f"ID {cls_id} {conf:.2f}"
    color = (0, 255, 0)
    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
    cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# ====== Save and/or Show Output ======
cv2.imwrite("output.jpg", image)
print("✅ Saved output image to 'output.jpg'")

try:
    cv2.imshow("YOLOv8 Debug Output", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
except:
    print("⚠️ cv2.imshow() not supported. Check 'output.jpg' manually.")
