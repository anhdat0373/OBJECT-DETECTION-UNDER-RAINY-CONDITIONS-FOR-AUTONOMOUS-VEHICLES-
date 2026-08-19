"""
ESP32-CAM + YOLOv8 - Nhận diện phương tiện giao thông
=====================================================
Yêu cầu: pip install ultralytics opencv-python pyserial
Tác giả: Dự án xe tự hành DIY
"""

import cv2
import serial
import time
import urllib.request
import numpy as np
from ultralytics import YOLO

# ============================================================
# CẤU HÌNH - Chỉnh sửa phần này trước khi chạy
# ============================================================

# Địa chỉ IP của ESP32-CAM (xem trong Serial Monitor Arduino IDE)
ESP32_IP = "192.168.100.245/"
STREAM_URL = f"http://192.168.100.245/:81/stream"

# Cổng COM của Arduino (Windows: "COM3", Mac/Linux: "/dev/ttyUSB0")
ARDUINO_PORT = "COM4"
ARDUINO_BAUD = 9600

# Ngưỡng tin cậy nhận diện (0.0 - 1.0), tăng lên nếu nhiều false positive
CONFIDENCE_THRESHOLD = 0.5

# Tốc độ xe (gửi kèm lệnh): 0-255
SPEED = 150

# Bật/tắt gửi lệnh xuống Arduino (False = chỉ xem camera, không điều khiển xe)
ENABLE_ARDUINO = True

# ============================================================
# CÁC CLASS CẦN NHẬN DIỆN (từ COCO dataset - 80 class)
# ============================================================
VEHICLE_CLASSES = {
    0:  "nguoi",
    1:  "xe_dap",
    2:  "o_to",
    3:  "xe_may",
    5:  "xe_buyt",
    7:  "xe_tai",
    9:  "den_giao_thong",
    11: "bien_bao_dung",
}

# Màu bounding box cho từng loại (BGR)
CLASS_COLORS = {
    0:  (0, 255, 255),    # người - vàng
    1:  (0, 255, 0),      # xe đạp - xanh lá
    2:  (255, 0, 0),      # ô tô - xanh dương
    3:  (0, 128, 255),    # xe máy - cam
    5:  (128, 0, 128),    # xe buýt - tím
    7:  (0, 0, 255),      # xe tải - đỏ
    9:  (0, 200, 200),    # đèn giao thông - vàng đậm
    11: (50, 50, 255),    # biển dừng - đỏ nhạt
}


# ============================================================
# KẾT NỐI ARDUINO
# ============================================================
def ket_noi_arduino():
    """Kết nối Serial với Arduino, trả về None nếu thất bại."""
    if not ENABLE_ARDUINO:
        print("[INFO] Chế độ xem camera - không kết nối Arduino")
        return None
    try:
        ser = serial.Serial(ARDUINO_PORT, ARDUINO_BAUD, timeout=1)
        time.sleep(2)  # Chờ Arduino reset
        print(f"[OK] Kết nối Arduino tại {ARDUINO_PORT}")
        return ser
    except Exception as e:
        print(f"[LỖI] Không kết nối được Arduino: {e}")
        print("      Kiểm tra lại cổng COM hoặc cắm dây USB")
        return None


def gui_lenh(ser, lenh: str):
    """
    Gửi lệnh 1 ký tự xuống Arduino.
    F = tiến | B = lùi | L = rẽ trái | R = rẽ phải | S = dừng
    """
    if ser and ser.is_open:
        ser.write(lenh.encode())


# ============================================================
# ĐỌC STREAM TỪ ESP32-CAM
# ============================================================
class ESP32Stream:
    """Đọc MJPEG stream từ ESP32-CAM qua HTTP."""

    def __init__(self, url: str):
        self.url = url
        self.stream = None
        self.bytes_buffer = b""

    def ket_noi(self) -> bool:
        try:
            self.stream = urllib.request.urlopen(self.url, timeout=5)
            print(f"[OK] Kết nối stream ESP32-CAM: {self.url}")
            return True
        except Exception as e:
            print(f"[LỖI] Không kết nối được ESP32-CAM: {e}")
            print("      Kiểm tra:")
            print("      1. ESP32-CAM và laptop cùng mạng WiFi")
            print(f"     2. Mở trình duyệt thử: {self.url}")
            return False

    def doc_frame(self):
        """Đọc 1 frame từ MJPEG stream, trả về numpy array hoặc None."""
        try:
            self.bytes_buffer += self.stream.read(4096)

            # Tìm ranh giới JPEG frame trong MJPEG stream
            start = self.bytes_buffer.find(b'\xff\xd8')  # JPEG start
            end = self.bytes_buffer.find(b'\xff\xd9')    # JPEG end

            if start != -1 and end != -1:
                jpg_data = self.bytes_buffer[start:end + 2]
                self.bytes_buffer = self.bytes_buffer[end + 2:]

                # Giải mã JPEG → numpy array
                frame = cv2.imdecode(
                    np.frombuffer(jpg_data, dtype=np.uint8),
                    cv2.IMREAD_COLOR
                )
                return frame
        except Exception:
            pass
        return None


# ============================================================
# LOGIC QUYẾT ĐỊNH: Từ kết quả nhận diện → lệnh điều khiển
# ============================================================
def quyet_dinh_hanh_dong(detections: list, frame_width: int) -> str:
    """
    Dựa vào vật thể phát hiện được, quyết định xe làm gì.
    
    Logic đơn giản:
    - Có người / xe gần (box lớn) → DỪNG
    - Có đèn đỏ → DỪNG  
    - Không có gì nguy hiểm → TIẾN
    
    Trả về: 'F' tiến | 'S' dừng | 'L' trái | 'R' phải
    """
    NGUONG_DUNG = 0.35  # Nếu vật chiếm >35% chiều rộng frame → dừng

    for det in detections:
        class_id = int(det['class_id'])
        x1, y1, x2, y2 = det['box']
        do_rong_vat = (x2 - x1) / frame_width

        # Người hoặc phương tiện đang ở gần → dừng khẩn
        if class_id in [0, 2, 3, 5, 7] and do_rong_vat > NGUONG_DUNG:
            return 'S'

        # Đèn giao thông hoặc biển dừng → dừng
        if class_id in [9, 11]:
            return 'S'

    # Không có vật cản → tiến
    return 'F'


# ============================================================
# HÀM VẼ THÔNG TIN LÊN FRAME
# ============================================================
def ve_thong_tin(frame, detections: list, lenh: str):
    """Vẽ bounding box, nhãn, và trạng thái xe lên frame."""

    for det in detections:
        class_id = int(det['class_id'])
        if class_id not in VEHICLE_CLASSES:
            continue

        x1, y1, x2, y2 = det['box']
        conf = det['confidence']
        ten = VEHICLE_CLASSES[class_id]
        mau = CLASS_COLORS.get(class_id, (255, 255, 255))

        # Vẽ bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), mau, 2)

        # Nhãn với nền mờ
        nhan = f"{ten} {conf:.0%}"
        (w, h), _ = cv2.getTextSize(nhan, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(frame, (x1, y1 - h - 8), (x1 + w + 4, y1), mau, -1)
        cv2.putText(frame, nhan, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    # Hiển thị lệnh xe ở góc trên trái
    trang_thai = {
        'F': ("TIEN", (0, 200, 0)),
        'S': ("DUNG", (0, 0, 255)),
        'L': ("RE TRAI", (0, 165, 255)),
        'R': ("RE PHAI", (0, 165, 255)),
    }
    ten_lenh, mau_lenh = trang_thai.get(lenh, ("...", (200, 200, 200)))
    cv2.rectangle(frame, (10, 10), (200, 50), (0, 0, 0), -1)
    cv2.putText(frame, f"XE: {ten_lenh}", (15, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, mau_lenh, 2)

    return frame


# ============================================================
# VÒNG LẶP CHÍNH
# ============================================================
def chay_chinh():
    print("=" * 50)
    print("  ESP32-CAM + YOLOv8 - Xe tự hành DIY")
    print("=" * 50)
    print("[INFO] Đang tải model YOLOv8n... (lần đầu ~vài phút)")

    # Tải model YOLOv8 nano (nhỏ nhất, nhanh nhất)
    # Lần đầu tự động tải ~6MB về máy
    model = YOLO("yolov8n.pt")
    print("[OK] Model YOLOv8n sẵn sàng")

    # Kết nối các thiết bị
    arduino = ket_noi_arduino()
    esp32 = ESP32Stream(STREAM_URL)

    if not esp32.ket_noi():
        print("\n[HƯỚNG DẪN] Nếu chưa có ESP32-CAM, test bằng webcam laptop:")
        print("            Đổi STREAM_URL = 0  (webcam index)")
        return

    print("\n[INFO] Nhấn Q để thoát | Nhấn P để pause\n")

    lenh_hien_tai = 'S'
    pause = False
    fps_list = []

    while True:
        t_start = time.time()

        frame = esp32.doc_frame()
        if frame is None:
            continue

        if not pause:
            # Chỉ nhận diện các class cần thiết để tăng tốc
            ket_qua = model(
                frame,
                conf=CONFIDENCE_THRESHOLD,
                classes=list(VEHICLE_CLASSES.keys()),
                verbose=False
            )[0]

            # Chuyển kết quả thành list dict dễ dùng
            detections = []
            for box in ket_qua.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append({
                    'class_id': box.cls[0].item(),
                    'confidence': box.conf[0].item(),
                    'box': (x1, y1, x2, y2)
                })

            # Quyết định hành động
            lenh_hien_tai = quyet_dinh_hanh_dong(detections, frame.shape[1])
            gui_lenh(arduino, lenh_hien_tai)

            # Vẽ lên frame
            frame = ve_thong_tin(frame, detections, lenh_hien_tai)

        # Tính và hiển thị FPS
        fps = 1 / (time.time() - t_start + 1e-6)
        fps_list.append(fps)
        if len(fps_list) > 30:
            fps_list.pop(0)
        fps_avg = sum(fps_list) / len(fps_list)
        cv2.putText(frame, f"FPS: {fps_avg:.1f}", (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow("ESP32-CAM + YOLOv8 | Nhan 'Q' de thoat", frame)

        phim = cv2.waitKey(1) & 0xFF
        if phim == ord('q'):
            break
        elif phim == ord('p'):
            pause = not pause
            print(f"[INFO] {'PAUSE' if pause else 'RESUME'}")

    # Dọn dẹp
    gui_lenh(arduino, 'S')  # Dừng xe trước khi thoát
    if arduino:
        arduino.close()
    cv2.destroyAllWindows()
    print("[INFO] Đã thoát chương trình")


# ============================================================
if __name__ == "__main__":
    chay_chinh()
