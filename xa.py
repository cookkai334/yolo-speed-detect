import cv2
import time
import math
from ultralytics import YOLO

# 1 pixel ~ 0.05 m (สมมุติ) → ปรับตาม calibration จริง
PIXEL_TO_METER = 0.05  

def main():
    # โหลดโมเดล YOLO (ใช้โมเดลเล็ก yolo v8n.pt จะได้เร็ว)
    model = YOLO("yolov8n.pt")

    # เปิดวิดีโอ mp4
    cap = cv2.VideoCapture("zzzz.mp4")

    prev_time = time.time()
    prev_centers = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # ตรวจจับเฉพาะรถ (class: car=2, motorbike=3, bus=5, truck=7)
        results = model.track(frame, persist=True, classes=[2, 3, 5, 7], verbose=False)

        current_time = time.time()
        dt = current_time - prev_time
        prev_time = current_time

        annotated_frame = results[0].plot()

        if results[0].boxes.id is not None:
            for box, track_id in zip(results[0].boxes.xyxy, results[0].boxes.id):
                x1, y1, x2, y2 = map(int, box[:4])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                if int(track_id) in prev_centers:
                    prev_cx, prev_cy = prev_centers[int(track_id)]
                    dx, dy = cx - prev_cx, cy - prev_cy
                    distance_pixels = math.sqrt(dx**2 + dy**2)

                    # คำนวณความเร็ว
                    speed_mps = (distance_pixels * PIXEL_TO_METER) / dt if dt > 0 else 0
                    speed_kmh = speed_mps * 3.6

                    cv2.putText(annotated_frame,
                                f"ID {int(track_id)}: {speed_kmh:.1f} km/h",
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                prev_centers[int(track_id)] = (cx, cy)

        # แสดงวิดีโอพร้อมความเร็ว
        cv2.imshow("YOLO Speed Estimation", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

#โหลด pip install numpy==1.26.4 
#โหลด pip install opencv-contrib-python==4.9.0.80
#โหลด pip install ultralytics
