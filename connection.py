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

model = YOLO("yolo26n_ncnn_model")
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

BBOX_SCALE_H = 200 # 200 px z odległości jednego metra
BBOX_SCALE_W = 200 # 200 px z odległości jednego metra

# DHCP ma stałe IP dla konkretnego adresu mac
SERVER_URL = "192.168.50.10"
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

#Konfiguracja sterowaniavvvvvvvvvvvvvvvv ; Kuba
DRONE_UDP_PORT = 5005
DEADZONE = 0.15
LOOP_RATE_HZ = 50

#flaga odpowiedzalna za działanie programu; Kuba
is_running = True
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

#flaga odpowiedzalna za sterowanie automatyczne; Bartek
auto_mode: bool = False
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

#flaga odpowiedzalna za sterowanie automatyczne; Bartek
target_id = 1
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
    global auto_mode
    global target_id

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
        if(btn_b):
            auto_mode = bool(False if auto_mode else True)
        btn_lb = pad.get_button(4)
        if(btn_lb and target_id > 1):
            target_id -= 1
        btn_rb = pad.get_button(5)
        if(btn_rb):
            target_id += 1

        # Wypisywanie wartości na żywo w jednej linijce konsoli; jeśli tryb sterowania ręcznego Bartek
        if(not auto_mode):
            sys.stdout.write(f"\r[PAD] THR:{throttle} YAW:{yaw} PITCH:{pitch} ROLL:{roll} A:{btn_a} B:{btn_b}  lb:{btn_lb} rb{btn_rb}")
            sys.stdout.flush()

            #Pakowanie i wysykłka UDP; Kuba
            packet = struct.pack("!4H1B", roll, pitch, throttle, yaw, btn_a)
            try:
                sock.sendto(packet, (SERVER_URL,DRONE_UDP_PORT))
            except Exception as e:
                pass

        elapsed = time.perf_counter() - t_start
        time.sleep(max(0.0, interval - elapsed))
    
    sock.close()
    pygame.quit()
    print("Zakończony wątek UDP sterowania")

#nowy main ze starą pętlą ; Kuba
def __main__():
    global is_running
    global target_id
    #uruchomienie wątku sterowania; Kuba
    ctrl_thread = threading.Thread(target = gamepad_thread, daemon = True)
    ctrl_thread.start()

    rtsp_url = "rtsp://SERVER_URL:8554/live"
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    array_id = 0
    auto_old = False    
    est_d = 0
    alpha = 0.8

    while cap.isOpened():
        start = time.perf_counter()

        ret, frame = cap.read()

        if not ret:
            time.sleep(0.1)
            break

        results = model.track(
            frame,
            classes=[0, 2],
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False #wyłączenie spamu w kontroli
        )
        annotated = results[0].plot()
        target = results[0].boxes

        if(len(target) > 0):

            ids = target.id.int().tolist()

            if target_id in ids:
                array_id = ids.index(target_id)
            cv2.putText(annotated,f"Tracked object: {target_id}",(10, 30),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 255, 255),2)
            cv2.imshow("Drones POV", annotated)

            if(auto_mode and auto_old == False):
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

            if auto_mode:
                target = target[array_id]
                x1, y1, x2, y2 = target.xyxy[0].tolist()

                x = (x1 + x2) / 2
                y = (y1 + y2) / 2
                height, width = frame.shape[:2]

                ex = 1/(width/2) * x -1
                ey = 1/(height/2) * y -1

                d_h = BBOX_SCALE_H / abs(y1 - y2) # albo ogniskowa(px) * wys(m)/wys(px) ogniskowa = wys(px)/wys(m) dla 1m
                d_w = BBOX_SCALE_W / abs(x1 - x2)
                d = 0.7 * d_h + 0.3 * d_w
                est_d = alpha * est_d + (1 - alpha) * d

                packet = struct.pack("!3H", ex, ey, est_d)
                try:
                    sock.sendto(packet, (SERVER_URL,DRONE_UDP_PORT))
                except Exception as e:
                    pass

            if(auto_mode==False and auto_old):
                sock.close()

            auto_old = auto_mode

        if cv2.waitKey(1) == 27:
            is_running = False #zamykanie wątku sterowania; Kuba
            break

        elapsed = time.perf_counter() - start
        remaining = FRAME_TIME - elapsed

        if remaining > 0:
            time.sleep(remaining)

    cap.release()
    cv2.destroyAllWindows() 
    ctrl_thread.join(timeout=1.0)
    