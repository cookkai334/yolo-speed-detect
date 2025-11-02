import cv2
import time
import math
import numpy as np
import statistics
from ultralytics import YOLO

#โหลด pip install numpy==1.26.4 
#โหลด pip install opencv-contrib-python==4.9.0.80
#โหลด pip install ultralytics
LANE_WIDTH_METERS = 3.5  

src_points = np.float32([[100,720],[1200,720],[750,450],[550,450]])
dst_points = np.float32([[0,500],[500,500],[500,0],[0,0]])
M = cv2.getPerspectiveTransform(src_points, dst_points)

def warp_point(x, y):
    pts = np.array([[[x, y]]], dtype="float32")
    dst = cv2.perspectiveTransform(pts, M)[0][0]
    return dst[0], dst[1]

def main():
    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture("zzzz.mp4")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30
    dt = 1.0 / fps

    prev_positions = {}
    speeds = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(frame, persist=True, classes=[2,3,5,7], verbose=False)
        annotated = frame.copy()

        if results and results[0].boxes.id is not None:
            for box, tid in zip(results[0].boxes.xyxy, results[0].boxes.id):
                if tid is None: 
                    continue
                x1,y1,x2,y2 = map(int, box[:4])
                cx, cy = (x1+x2)//2, (y1+y2)//2

                wx, wy = warp_point(cx, cy)

                if int(tid) in prev_positions:
                    px, py = prev_positions[int(tid)]
                    dist = math.hypot(wx-px, wy-py)

                    meters_per_pixel = LANE_WIDTH_METERS / (dst_points[1][0]-dst_points[0][0])
                    dist_m = dist * meters_per_pixel

                    # ถ้าเด้งเกินจริง → ข้าม
                    if dist_m > 20:
                        continue

                    speed_mps = dist_m / dt
                    speed_kmh = min(speed_mps * 3.6, 200)  # จำกัดไม่เกิน 200 km/h

                    if int(tid) not in speeds:
                        speeds[int(tid)] = []
                    speeds[int(tid)].append(speed_kmh)
                    if len(speeds[int(tid)]) > 10:
                        speeds[int(tid)].pop(0)

                    avg_speed = statistics.median(speeds[int(tid)])

                    cv2.putText(annotated, f"{avg_speed:.1f} km/h",
                                (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.9, (0,0,255), 3)

                prev_positions[int(tid)] = (wx, wy)
                cv2.rectangle(annotated,(x1,y1),(x2,y2),(0,255,0),2)

        cv2.imshow("Speed Estimation", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()

