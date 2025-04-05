import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter

# ====== CONFIGURATION ======
IMAGE_PATH = "image.jpg"
MODEL_PATH = "best_float32.tflite"
CONFIDENCE_THRESHOLD = 0.1
DEBUG_MODE = True

# ====== Load model ======
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ====== Load image ======
image = cv2.imread(IMAGE_PATH)
if image is None:
    print(f"❌ Could not load image '{IMAGE_PATH}'")
    exit()

original_h, original_w = image.shape[:2]
resized = cv2.resize(image, (640, 640))
input_tensor = np.expand_dims(resized, axis=0).astype(np.float32) / 255.0

# ====== Inference ======
interpreter.set_tensor(input_details[0]['index'], input_tensor)
interpreter.invoke()
output = interpreter.get_tensor(output_details[0]['index'])[0]  # Shape: (n, 6)

print(f"✅ Model ran inference. Output shape: {output.shape}")

# ====== Parse predictions ======
boxes_drawn = 0
for i, det in enumerate(output):
    x1, y1, x2, y2, conf, cls_id = det
    if conf < CONFIDENCE_THRESHOLD:
        continue

    # Scale boxes to original image size
    x1 = int(x1 * original_w)
    y1 = int(y1 * original_h)
    x2 = int(x2 * original_w)
    y2 = int(y2 * original_h)

    if DEBUG_MODE:
        print(f"[{i}] x1: {x1}, y1: {y1}, x2: {x2}, y2: {y2}, conf: {conf:.2f}, class: {int(cls_id)}")

    label = f"ID {int(cls_id)} {conf:.2f}"
    color = (0, 255, 0)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    boxes_drawn += 1

# ====== Save / show result ======
if boxes_drawn == 0:
    print("⚠️ No boxes passed confidence threshold.")
else:
    print(f"✅ {boxes_drawn} box(es) drawn.")

cv2.imwrite("output.jpg", image)
print("🖼 Saved result to 'output.jpg'")

try:
    cv2.imshow("YOLOv8 Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
except:
    print("⚠️ GUI not supported — check 'output.jpg'")
