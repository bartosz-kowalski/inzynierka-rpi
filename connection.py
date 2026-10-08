import sys #Kuba
import os #Kuba

import numpy as np

import time

import cv2

import socket
import struct
import threading #wielowątkowość żeby wysyłąnie ramek działało niezależnie od odczytów z pada ;Kuba
import pygame #zczytwanie pada ;Kuba

from ultralytics import YOLO

from KF import KalmanFilter
from PID import PID

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
auto_mode: bool; auto_mode_old: bool = False, False
roll_a, pitch_a, yaw_a, throttle_a = 0, 0, 0, 0
roll, pitch, yaw, throttle, btn_a = 0, 0, 0, 0, 0
z = None
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

#flaga odpowiedzalna za sterowanie automatyczne; Bartek
target_id = 1
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

#ochrona pamięci
data_lock = threading.Lock()

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
    global auto_mode, auto_mode_old
    global roll, pitch, yaw, throttle, btn_a
    global target_id

    pygame.init()
    pygame.joystick.init()

    if pygame.joystick.get_count() == 0:
        print("Nie wykryto pada, sterowania wyłączone")
        return

    pad = pygame.joystick.Joystick(0)
    prev_b, prev_rb, prev_lb = 0, 0, 0
    pad.init()
    print(f"Podłączono z padem : {pad.get_name()}")

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
        with data_lock:
            yaw = scale_to_pwm(raw_yaw)
            throttle = scale_to_pwm(raw_thr, reverse = True)
            roll = scale_to_pwm(raw_roll)
            pitch = scale_to_pwm(raw_pitch, reverse = True)
            btn_a = pad.get_button(0)

            b_b = pad.get_button(1)
            if(b_b and not prev_b):
                auto_mode_old = auto_mode
                auto_mode = bool(False if auto_mode else True)

        btn_lb = pad.get_button(4)
        if(btn_lb and target_id > 1 and not prev_lb):
            target_id -= 1
        btn_rb = pad.get_button(5)
        if(btn_rb and not prev_rb):
            target_id += 1

        prev_b = b_b
        prev_lb = btn_lb
        prev_rb = btn_rb

        elapsed = time.perf_counter() - t_start
        time.sleep(max(0.0, interval - elapsed))
    
    pygame.quit()
    print("Zakończony wątek odczytu z pada")

def udp_thread():
    global roll_a, pitch_a, yaw_a, throttle_a
    global roll, pitch, yaw, throttle, btn_a
    global auto_mode

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    interval = 1.0 / LOOP_RATE_HZ

    while is_running:
        t_start = time.perf_counter()

        with data_lock:
            auto_local = auto_mode
            if(auto_local):
                r, p, t, y = roll_a, pitch_a, throttle_a, yaw_a
            else:
                r, p, t, y = roll, pitch, throttle, yaw
            b_a = btn_a

        #Pakowanie i wysykłka UDP 
        packet = struct.pack("!4H1B", r, p, t, y, b_a)
        try:
            sock.sendto(packet, (SERVER_URL, DRONE_UDP_PORT))
        except Exception:
            pass

        elapsed = time.perf_counter() - t_start
        time.sleep(max(0.0, interval - elapsed))

    sock.close()
    print("Zakończony wątek wysyłania rozkazów")


def control_thread():
    global auto_mode, auto_mode_old
    global roll_a, pitch_a, yaw_a, throttle_a

    control_frequency = 20
    control_time = 1 / control_frequency

    filter = KalmanFilter(0.05)
    PI_x = PID(1, 0, 0, -5, 5)
    PI_y = PID(1, 0, 0, -5, 5)
    PI_d = PID(1, 0, 0, -5, 5)
    while is_running:
        t_s = time.monotonic()

        with data_lock:
            current_z = z.copy() if z is not None else None
            auto_local = auto_mode
            auto_local_old = auto_mode_old

        if(auto_local and not auto_local_old and current_z is not None):
            filter = KalmanFilter(control_time, x0 = current_z)

        if(auto_local and current_z is not None):
            ex, ey, d, d_ex, d_ey, d_d = filter.predict().flatten()

            ux = PI_x.solve(ex, d_ex)
            uy = PI_y.solve(ey, d_ey)
            ud = PI_d.solve(d, d_d)

            ###
            ###     OBLICZENIA TYMCZASOWE DO POPRAWY UWZGLĘDNIĆ GEOMETRIĘ GIMBALA
            ###
            temp_roll = np.interp(ux, [-5, 5], [-20, 20])
            temp_yaw = np.interp(ux, [-5, 5], [-30, 30])
            temp_pitch = np.interp(uy, [-5, 5], [-20, 20])
            temp_throttle = np.interp(ud, [0, 10], [10, 30])

            temp_roll = scale_to_pwm(temp_roll)
            temp_pitch = scale_to_pwm(temp_pitch)
            temp_yaw = scale_to_pwm(temp_yaw)
            temp_throttle = scale_to_pwm(temp_throttle)

            with data_lock:
                roll_a, pitch_a, yaw_a, throttle_a = temp_roll, temp_pitch, temp_yaw, temp_throttle

            filter.update(current_z)

        elapsed = time.monotonic() - t_s

        if(elapsed < control_time):
            time.sleep(control_time - elapsed)

    print("Zakończony wątek sterowania autoamtycznego")

# Główna funkcja programu
def main():
    global is_running
    global target_id
    global z

    # threads initialization
    pad_thread = threading.Thread(target = gamepad_thread, daemon = True)
    pad_thread.start()

    ctrl_thread = threading.Thread(target = control_thread, daemon = True)
    ctrl_thread.start()

    connection_thread = threading.Thread(target = udp_thread, daemon = True)
    connection_thread.start()

    # video server URL
    rtsp_url = f"rtsp://{SERVER_URL}:8554/live"
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    array_id = 0

    # additional variables, used to calculate error derivatives for the kalman filter
    ex_old = 0
    ey_old = 0
    d_old = 0

    while cap.isOpened():
        start = time.perf_counter()
        t_s = time.monotonic()

        ret, frame = cap.read()

        if not ret:
            time.sleep(0.1)
            break

        results = model.track(
            frame,
            classes=[0, 2],
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False 
        )
        annotated = results[0].plot()
        target = results[0].boxes

        if(len(target) > 0 and target.id is not None):

            ids = target.id.int().tolist()

            if target_id in ids:
                array_id = ids.index(target_id)
            cv2.putText(annotated,f"Tracked object: {target_id}",(10, 30),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 255, 255),2)
            cv2.imshow("Drones POV", annotated)

            with data_lock:
                auto_local = auto_mode

            if(auto_local):
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

                dt = time.monotonic() - t_s
                d_ex = (ex - ex_old) / dt
                d_ey = (ey - ey_old) / dt
                d_d = (d - d_old) / dt

                with data_lock:
                    z = np.array([[ex], [ey], [d], [d_ex], [d_ey], [d_d]], dtype=np.float32)

                ex_old = ex
                ey_old = ey
                d_old = d

        if cv2.waitKey(1) == 27:
            is_running = False # closing all the threads
            break

        elapsed = time.perf_counter() - start
        remaining = FRAME_TIME - elapsed

        if remaining > 0:
            time.sleep(remaining)

    cap.release()
    cv2.destroyAllWindows() 
    ctrl_thread.join(timeout=1.0)
    pad_thread.join(timeout=1.0)
    connection_thread.join(timeout=1.0)
    print("Program kończy pracę")
    
if __name__ == "__main__":
    main()
