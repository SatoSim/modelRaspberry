import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter

# ====== CONFIGURATION ======
IMAGE_PATH = "image.jpg"
MODEL_PATH = "best_float32.tflite"
CONFIDENCE_THRESHOLD = 0.3
IOU_THRESHOLD = 0.45
INPUT_SIZE = 640

# ====== Load TFLite model ======
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ====== Load and preprocess image ======
image = cv2.imread(IMAGE_PATH)
original_h, original_w = image.shape[:2]
resized = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE))
input_tensor = np.expand_dims(resized, axis=0).astype(np.float32) / 255.0

# ====== Run inference ======
interpreter.set_tensor(input_details[0]['index'], input_tensor)
interpreter.invoke()
output = interpreter.get_tensor(output_details[0]['index'])[0]

# ====== Parse results and draw boxes ======
boxes, class_ids, confidences = [], [], []
for pred in output:
    x_center, y_center, w, h, conf, cls = pred

    if conf < CONFIDENCE_THRESHOLD:
        continue

    x = int((x_center - w / 2) * original_w / INPUT_SIZE)
    y = int((y_center - h / 2) * original_h / INPUT_SIZE)
    width = int(w * original_w / INPUT_SIZE)
    height = int(h * original_h / INPUT_SIZE)

    boxes.append([x, y, width, height])
    confidences.append(float(conf))
    class_ids.append(int(cls))

indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, IOU_THRESHOLD)

for i in indices.flatten():
    x, y, w, h = boxes[i]
    label = f"ID {class_ids[i]} {confidences[i]:.2f}"
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

cv2.imshow("YOLOv8 TFLite Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
