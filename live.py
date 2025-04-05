import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter
import time

# ====== CONFIGURATION ======
MODEL_PATH = "best_float32.tflite"
CONFIDENCE_THRESHOLD = 0.1
CLASS_NAMES = {
    0: "Satoshi",
    1: "Alfredo"
}
DEBUG_MODE = True
INPUT_SIZE = 640

# ====== Load model ======
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ====== Open Pi Camera ======
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Could not open camera. Try using legacy camera stack.")
    exit()

print("📷 Camera opened. Starting detection...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    original_h, original_w = frame.shape[:2]
    resized = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
    input_tensor = np.expand_dims(resized, axis=0).astype(np.float32) / 255.0

    # Inference
    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])[0]

    boxes_drawn = 0
    for i, det in enumerate(output):
        x1, y1, x2, y2, conf, cls_id = det
        if conf < CONFIDENCE_THRESHOLD:
            continue

        # Scale coords
        x1 = int(x1 * original_w)
        y1 = int(y1 * original_h)
        x2 = int(x2 * original_w)
        y2 = int(y2 * original_h)

        name = CLASS_NAMES.get(int(cls_id), f"ID {int(cls_id)}")
        label = f"{name} {conf:.2f}"

        if DEBUG_MODE:
            print(f"[{i}] {label} at ({x1},{y1},{x2},{y2})")

        color = (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        boxes_drawn += 1

    cv2.imshow("Live YOLOv8 Detection", frame)

    # Press 'q' to quit and save last frame
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        cv2.imwrite("last_frame.jpg", frame)
        print("🖼 Saved 'last_frame.jpg' and exiting...")
        break

cap.release()
cv2.destroyAllWindows()
