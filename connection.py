import http.client

import numpy as np

import time

import cv2

from ultralytics import YOLO

MAX_FPS = 10
FRAME_TIME = 1.0 / MAX_FPS
CLR = (0,0,255)

cur_id = 0

model = YOLO("yolo11x.pt")

SERVER_URL = "192.168.4.1"

def capture_frame():
    try:
        connection = http.client.HTTPConnection(SERVER_URL, 80, timeout=10)
        connection.request("GET", "/frame")
        response = connection.getresponse()

        if(response.status == 200):
            frame_buf = response.read()
            nparr = np.frombuffer(frame_buf, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            #with open("image.jpeg", "wb") as file:
                #file.write(frame)
            #print("Image saved")
        #else:
            #print(f"Connection failed, reason: {response.status} {response.reason}")

    except Exception as e:
        print(f"Error: {e}")

    finally:
        connection.close()
        return frame

while True:
    start = time.perf_counter()

    frame = capture_frame()

    results = model.track(
        frame,
        classes=[0, 2],
        persist=True,
        tracker="bytetrack.yaml"
    )
    if results.boxes is not None and results.boxes.id is not None:

        annotated = results[0].plot()
        max_id = len(results[0].boxes)

        boxes = results.boxes.xyxy.int().cpu().toList()
        box_ids = results.boxes.id.int().cpu().toList()

        for (box, id) in zip(boxes, box_ids):
            if(id == cur_id):
                x1, y1, x2, y2 = box
                cv2.rectangle(annotated, (x1, y1), (x2, y2), CLR, 2)
                break

    cv2.imshow("Tracking", annotated)

    if cv2.waitKey(1) == 27:
        break

    elapsed = time.perf_counter() - start
    remaining = FRAME_TIME - elapsed

    if remaining > 0:
        time.sleep(remaining)
