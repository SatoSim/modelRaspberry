import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter
import time

# ====== CONFIG ======
MODEL_PATH = "best_float32.tflite"
CONFIDENCE_THRESHOLD = 0.1
CLASS_NAMES = {0: "Satoshi", 1: "Alfredo"}

# ====== Load model ======
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
INPUT_SIZE = input_details[0]['shape'][1]

# ====== Start camera ======
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Could not open camera.")
    exit()

print("📷 Camera opened. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    # ✅ Ensure 3-channel input
    if len(frame.shape) == 2 or frame.shape[2] == 1:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    original_h, original_w = frame.shape[:2]
    resized = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
    input_tensor = np.expand_dims(resized.astype(np.float32) / 255.0, axis=0)

    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])[0]

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

    cv2.imshow("Live YOLOv8 Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.imwrite("last_frame.jpg", frame)
        print("🖼 Saved 'last_frame.jpg'. Exiting.")
        break

cap.release()
cv2.destroyAllWindows()
