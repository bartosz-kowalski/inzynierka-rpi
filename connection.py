import sys #Kuba
import os #Kuba

import http.client

import numpy as np

import time

import cv2

import socket
import struct
import threading #wielowątkowość żeby wysyłąnie ramek działało niezależnie od odczytów z pada ;Kuba
import pygame #zczytwanie pada ;Kuba

from ultralytics import YOLO

# Konfiguracaja wideo i yolo vvvvvvvv
MAX_FPS = 24
FRAME_TIME = 1.0 / MAX_FPS
CLR = (0,0,255)

cur_id = 1

model = YOLO("yolo26n_ncnn_model")

SERVER_URL = "192.168.4.1"
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

#Konfiguracja sterowaniavvvvvvvvvvvvvvvv ; Kuba
DRONE_UDP_PORT = 5005
DEADZONE = 0.15
LOOP_RATE_HZ = 50

#flaga odpowiedzalna za działanie programu; Kuba
is_running = True
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

#Połączenie HTTP vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
connection = http.client.HTTPConnection(SERVER_URL, 80, timeout=10)

#Pad strefa martwa dla gałek
def apply_deadzone(val: float, threshold: float = DEADZONE) -> float:
    return 0.0 if abs(val) < threshold else val

#Pad zamiana wartości na takie do sterowania silnika (1000,2000) pygame zwraca wartości (-1;1) dlatego przemnażanie; Kuba
def scale_to_pwm(axis_val: float, reverse: bool = False) -> int:
    if reverse:
        axis_val = -axis_val
    pwm = int(1500 + (axis_val*500))
    return max(1000,min(2000,pwm))

#wątek działający w tle wysyłanie sterowania z pada; Kuba
def gamepad_thread():
    pygame.init()
    pygame.joystick.init()

    if pygame.joystick.get_count() == 0:
        print("Nie wykryto pada, sterowania wyłączone")
        return

    pad = pygame.joystick.Joystick(0)
    pad.init()
    print(f"Podłączono z padem : {pad.get_name()}")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    interval = 1.0 / LOOP_RATE_HZ

    while is_running:
        t_start = time.perf_counter()
        pygame.event.pump()

        #Odczyt osi; Kuba
        raw_yaw = apply_deadzone(pad.get_axis(0))
        raw_thr = apply_deadzone(pad.get_axis(1))
        raw_roll = apply_deadzone(pad.get_axis(2))
        raw_pitch = apply_deadzone(pad.get_axis(3))

        #Zamiana na 1000-2000; Kuba
        yaw = scale_to_pwm(raw_yaw)
        throttle = scale_to_pwm(raw_thr, reverse = True)
        roll = scale_to_pwm(raw_roll)
        pitch = scale_to_pwm(raw_pitch, reverse = True)

        #odczyt przycisków; Kuba
        btn_a = pad.get_button(0)
        btn_b = pad.get_button(1)
        btn_lb = pad.get_button(4)
        btn_rb = pad.get_button(5)

        # Wypisywanie wartości na żywo w jednej linijce konsoli
        sys.stdout.write(f"\r[PAD] THR:{throttle} YAW:{yaw} PITCH:{pitch} ROLL:{roll} A:{btn_a} B:{btn_b}  lb:{btn_lb} rb{btn_rb}")
        sys.stdout.flush()

        #Pakowanie i wysykłka UDP; Kuba
        packet = struct.pack("!4H4B", roll, pitch, throttle, yaw, btn_a, btn_b, btn_lb, btn_rb)
        try:
            sock.sendto(packet, (SERVER_URL,DRONE_UDP_PORT))
        except Exception as e:
            pass

        elapsed = time.perf_counter() - t_start
        time.sleep(max(0.0, interval - elapsed))
    
    sock.close()
    pygame.quit()
    print("Zakończony wątek UDP sterowania")



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

#Dodałem def main w który wkleiłem stary while; Kuba
def main():
    global is_running
    #uruchomienie wątku sterowania; Kuba
    ctrl_thread = threading.Thread(target = gamepad_thread, daemon = True)
    ctrl_thread.start()

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
            tracker="bytetrack.yaml",
            verbose=False #wyłączenie spamu w kontroli
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
            is_running = False #zamykanie wątku sterowania; Kuba
            break

        elapsed = time.perf_counter() - start
        remaining = FRAME_TIME - elapsed

        if remaining > 0:
            print("waiting")
            time.sleep(remaining)

    cv2.destroyAllWindows()
    ctrl_thread.join(timeout=1.0)

if __name__ == "__main__":
    main()
