import http.client

import numpy as np

import time

import cv2

from ultralytics import YOLO

MAX_FPS = 10
FRAME_TIME = 1.0 / MAX_FPS
CLR = (0,0,255)

cur_id = 1

model = YOLO("yolo26n_ncnn_model")

SERVER_URL = "192.168.4.1"

connection = http.client.HTTPConnection(SERVER_URL, 80, timeout=10)

def capture_frame():

    global connection
    frame = None
    
    try:
        connection.request("GET", "/frame", headers={"Connection": "keep-alive"})
        response = connection.getresponse()

        if response.status == 200:
            frame_buf = response.read()
            nparr = np.frombuffer(frame_buf, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    except Exception as e:
        print(f"Błąd połączenia: {e}")
        connection.close()
        connection = http.client.HTTPConnection(SERVER_URL, 80, timeout=10)

    return frame

while True:
    start = time.perf_counter()

    frame = capture_frame()

    if frame is None:
        time.sleep(0.1)
        continue

    results = model.track(
        frame,
        classes=[0, 2],
        persist=True,
        tracker="bytetrack.yaml"
    )
    if results[0].boxes is not None and results[0].boxes.id is not None:

        annotated = results[0].plot()
        max_id = len(results[0].boxes)

        boxes = results[0].boxes.xyxy.int().cpu().tolist()
        box_ids = results[0].boxes.id.int().cpu().tolist()

        for (box, id) in zip(boxes, box_ids):
            if(id == cur_id):
                x1, y1, x2, y2 = box
                cv2.rectangle(annotated, (x1, y1), (x2, y2), CLR, 2)
                break

        cv2.imshow("Tracking", annotated)
    else:
        cv2.imshow("Tracking", frame)

    if cv2.waitKey(1) == 27:
        break

    elapsed = time.perf_counter() - start
    remaining = FRAME_TIME - elapsed

    if remaining > 0:
        time.sleep(remaining)

cv2.destroyAllWindows()
