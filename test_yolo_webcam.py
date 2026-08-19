from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")  # Tự tải ~6MB lần đầu
cap = cv2.VideoCapture(0)   # 0 = webcam laptop

while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame, conf=0.5, verbose=False)[0]
    frame = results.plot()
    cv2.imshow("Test YOLOv8 - Nhan Q de thoat", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()