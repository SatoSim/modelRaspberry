import cv2
import numpy as np
import onnxruntime as ort

# ====== CONFIGURATION ======
IMAGE_PATH = "image.jpg"
MODEL_PATH = "best.onnx"
CONFIDENCE_THRESHOLD = 0.3
IOU_THRESHOLD = 0.45  # For NMS
INPUT_SIZE = (640, 640)

# ====== HELPER: Non-max suppression ======
def nms(boxes, scores, iou_threshold):
    indices = cv2.dnn.NMSBoxes(
        bboxes=boxes,
        scores=scores,
        score_threshold=CONFIDENCE_THRESHOLD,
        nms_threshold=iou_threshold
    )
    return indices.flatten() if len(indices) > 0 else []

# ====== LOAD IMAGE ======
image = cv2.imread(IMAGE_PATH)
original_h, original_w = image.shape[:2]
resized = cv2.resize(image, INPUT_SIZE)
input_img = resized.transpose(2, 0, 1)[np.newaxis, :, :, :].astype(np.float32) / 255.0

# ====== LOAD MODEL AND RUN INFERENCE ======
session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
outputs = session.run(None, {"images": input_img})  # 'images' is the input name for YOLOv8 ONNX

# ====== PARSE OUTPUTS ======
predictions = outputs[0][0]  # Shape: (num_detections, 6)
boxes, confidences, class_ids = [], [], []

for pred in predictions:
    x_center, y_center, w, h, conf, cls = pred

    if conf < CONFIDENCE_THRESHOLD:
        continue

    x = int((x_center - w / 2) * original_w / INPUT_SIZE[0])
    y = int((y_center - h / 2) * original_h / INPUT_SIZE[1])
    width = int(w * original_w / INPUT_SIZE[0])
    height = int(h * original_h / INPUT_SIZE[1])

    boxes.append([x, y, width, height])
    confidences.append(float(conf))
    class_ids.append(int(cls))

# ====== APPLY NON-MAX SUPPRESSION ======
indices = nms(boxes, confidences, IOU_THRESHOLD)

# ====== DRAW BOXES ======
for i in indices:
    x, y, w, h = boxes[i]
    cls = class_ids[i]
    conf = confidences[i]

    label = f"ID {cls} {conf:.2f}"
    color = (0, 255, 0)
    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
    cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# ====== SHOW IMAGE ======
cv2.imshow("YOLOv8 ONNX Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
