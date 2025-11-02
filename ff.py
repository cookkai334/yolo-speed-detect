"""
YOLO vehicle speed overlay (labels avoid overlap & are clear)

Requirements:
  pip install ultralytics opencv-contrib-python numpy torch torchvision torchaudio
  (use numpy==1.26.4 if you had compatibility problems)
"""

import cv2
import math
import time
import numpy as np
from ultralytics import YOLO
from collections import deque

# ---------- SETTINGS ----------
VIDEO_PATH = "zzzz.mp4"
MODEL = "yolov8n.pt"         # หรือ yolov8s.pt / yolov8m.pt
PIXEL_TO_METER = 0.02        # ปรับตาม calibration: 1 px = 0.02 m (ตัวอย่าง)
SMOOTH_FRAMES = 5            # ใช้เฉลี่ยความเร็วจาก N เฟรมล่าสุด
CLASSES_TO_TRACK = [2,5,7]   # COCO ids: 2=car,5=bus,7=truck (เพิ่ม 3=motorbike ถ้าต้องการ)
FONT = cv2.FONT_HERSHEY_SIMPLEX

# ---------- helper utilities ----------
def distinct_color(idx):
    """ deterministic color per id """
    np.random.seed(idx+12345)
    c = tuple(int(x) for x in np.random.randint(60, 230, size=3))
    return c

def box_to_centroid(box):
    x1,y1,x2,y2 = box
    return ((x1 + x2)/2.0, (y1 + y2)/2.0)

def rect_overlap(r1, r2):
    # r = (x1,y1,x2,y2)
    return not (r2[0] > r1[2] or r2[2] < r1[0] or r2[1] > r1[3] or r2[3] < r1[1])

def place_label_no_overlap(desired_rect, occupied_rects, step=10, max_tries=10):
    # desired_rect: (x1,y1,x2,y2). Try shifting upwards to avoid overlap.
    x1,y1,x2,y2 = desired_rect
    for t in range(max_tries):
        candidate = (x1, y1 - t*step, x2, y2 - t*step)
        collision = False
        for occ in occupied_rects:
            if rect_overlap(candidate, occ):
                collision = True
                break
        if not collision:
            return candidate
    # fallback: return original
    return desired_rect

# ---------- Tracker & smoothing store ----------
class TrackStore:
    def __init__(self):
        # id -> deque of (frame_idx, cx, cy)
        self.positions = {}
        # id -> deque of computed speeds (m/s) for smoothing if desired
        self.speeds = {}

    def update(self, obj_id, cx, cy, frame_idx):
        if obj_id not in self.positions:
            self.positions[obj_id] = deque(maxlen=SMOOTH_FRAMES)
            self.speeds[obj_id] = deque(maxlen=SMOOTH_FRAMES)
        self.positions[obj_id].append((frame_idx, cx, cy))

    def compute_speed(self, obj_id, fps):
        """ compute smoothed speed in m/s using delta over available history """
        pos = self.positions.get(obj_id, None)
        if not pos or len(pos) < 2:
            return None
        # compute speed between oldest and newest in deque for stability
        f1, x1, y1 = pos[0]
        f2, x2, y2 = pos[-1]
        dt = (f2 - f1) / fps if fps > 0 else None
        if not dt or dt <= 0:
            return None
        dist_px = math.hypot(x2 - x1, y2 - y1)
        speed_m_s = (dist_px * PIXEL_TO_METER) / dt
        # append to speeds for moving-average smoothing
        self.speeds[obj_id].append(speed_m_s)
        # return average of stored speeds
        return float(np.mean(self.speeds[obj_id]))

# ---------- MAIN ----------
def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("ERROR: cannot open", VIDEO_PATH)
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    model = YOLO(MODEL)

    store = TrackStore()
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # run detection+tracking (ultralytics track API)
        # persist=True keeps track ids; classes filter for vehicles
        results = model.track(frame, persist=True, classes=CLASSES_TO_TRACK, verbose=False)
        r = results[0]

        # list of bbox (x1,y1,x2,y2), id
        bboxes = []
        ids = []
        if r.boxes is not None and len(r.boxes) > 0:
            # บางครั้ง r.boxes.id อาจเป็น None ถ้ายัง track ไม่ได้
            id_list = r.boxes.id
            if id_list is None:
                continue  # ไม่มี ID -> ข้ามเฟรมนี้ไปเลย
            for box, tid in zip(r.boxes.xyxy.tolist(), id_list.tolist()):
                if tid is None:
                    continue
                x1, y1, x2, y2 = box[:4]
                bboxes.append((int(x1), int(y1), int(x2), int(y2)))
                ids.append(int(tid))


        # update trackstore with centroids
        for (x1,y1,x2,y2), tid in zip(bboxes, ids):
            cx, cy = ( (x1+x2)/2.0, (y1+y2)/2.0 )
            store.update(tid, cx, cy, frame_idx)

        # draw bboxes and prepare label placement avoiding overlap
        occupied_label_rects = []   # list of (x1,y1,x2,y2) occupied by labels this frame
        for (x1,y1,x2,y2), tid in zip(bboxes, ids):
            color = distinct_color(tid)
            # draw bbox thicker for visibility
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)

            # compute smoothed speed (m/s) and km/h
            speed_m_s = store.compute_speed(tid, fps)
            speed_text = "calculating..."
            if speed_m_s is not None:
                speed_kmh = speed_m_s * 3.6
                speed_text = f"{speed_kmh:.1f} km/h"

            id_text = f"ID {tid}"
            label_text = f"{id_text} | {speed_text}"

            # measure text size
            (tw, th), baseline = cv2.getTextSize(label_text, FONT, 0.6, 2)
            pad = 6
            desired_x1 = x1
            desired_y1 = y1 - (th + pad + 4)  # try above bbox
            desired_x2 = desired_x1 + tw + pad*2
            desired_y2 = desired_y1 + th + pad*2

            # if label would go above image top, move inside bbox top
            if desired_y1 < 0:
                desired_y1 = y1 + 4
                desired_y2 = desired_y1 + th + pad*2

            desired_rect = (desired_x1, desired_y1, desired_x2, desired_y2)

            # place label avoiding overlap with previously placed labels
            placed_rect = place_label_no_overlap(desired_rect, occupied_label_rects, step= (th+8), max_tries=8)
            occupied_label_rects.append(placed_rect)

            lx1, ly1, lx2, ly2 = map(int, placed_rect)
            # draw filled rectangle background (semi-opaque look)
            # to get semi opacity, draw on overlay and blend (we'll use solid for simplicity)
            cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), color, -1)  # filled background
            # put text (use black text for contrast)
            text_x = lx1 + pad
            text_y = ly1 + th + (pad//2)
            cv2.putText(frame, label_text, (text_x, text_y), FONT, 0.6, (0,0,0), 2, cv2.LINE_AA)

            # optional small leader line from label to bbox centroid if label moved
            lab_cx = (lx1 + lx2)//2
            lab_cy = (ly1 + ly2)//2
            box_cx = int((x1+x2)/2)
            box_cy = int((y1+y2)/2)
            # draw thin line if labels are offset above
            if lab_cy < box_cy - 10:
                cv2.line(frame, (lab_cx, lab_cy+ (th//2)), (box_cx, box_cy), color, 1, cv2.LINE_AA)

        # small info overlay - FPS
        cv2.putText(frame, f"Frame {frame_idx}  FPS {fps:.1f}", (10,30), FONT, 0.7, (255,255,255), 2, cv2.LINE_AA)

        # show window (requires opencv-contrib-python)
        cv2.imshow("Vehicle speed overlay", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('p'):
            # pause toggle
            cv2.waitKey(0)  # wait until any key

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
