import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter
import time

# ====== CONFIG ======
MODEL_PATH = "best_float32.tflite"
IMAGE_PATH = "captured.jpg"
OUTPUT_PATH = "detection.jpg"
CONFIDENCE_THRESHOLD = 0.1
CLASS_NAMES = {0: "Satoshi", 1: "Alfredo"}

# ====== Load model ======
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']  # should be [1, 640, 640, 3]
INPUT_SIZE = input_shape[1]

# ====== Capture one image ======
print("📷 Opening camera to capture one image...")
cap = cv2.VideoCapture(0)
time.sleep(2)  # allow camera to warm up
ret, frame = cap.read()
cap.release()

if not ret:
    print("❌ Failed to capture image.")
    exit()

cv2.imwrite(IMAGE_PATH, frame)
print(f"✅ Image saved as '{IMAGE_PATH}'")

# ====== Preprocess ======
original_h, original_w = frame.shape[:2]
resized = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
input_tensor = np.expand_dims(resized, axis=0).astype(np.float32) / 255.0

interpreter.set_tensor(input_details[0]['index'], input_tensor)
interpreter.invoke()
output = interpreter.get_tensor(output_details[0]['index'])[0]

# ====== Parse + Draw ======
for det in output:
    x1, y1, x2, y2, conf, cls_id = det
    if conf < CONFIDENCE_THRESHOLD:
        continue

    x1 = int(x1 * original_w)
    y1 = int(y1 * original_h)
    x2 = int(x2 * original_w)
    y2 = int(y2 * original_h)

    label = f"{CLASS_NAMES.get(int(cls_id), f'ID {int(cls_id)}')} {conf:.2f}"
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

cv2.imwrite(OUTPUT_PATH, frame)
print(f"✅ Detection result saved as '{OUTPUT_PATH}'")

try:
    cv2.imshow("Detection Result", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
except:
    print("⚠️ GUI not supported — open 'detection.jpg'")
