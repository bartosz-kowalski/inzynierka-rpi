from ultralytics import YOLO
import subprocess

model = YOLO("yolo26n.pt")

model.export(format="ncnn", int8=True)

subprocess.run("sudo ./rpi_wifi_setup.sh")
