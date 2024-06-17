import cv2
import time
import threading
import pyttsx3
from ultralytics import YOLO
import kivy
from kivy.app import App
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
import android
from android.permissions import request_permissions, Permission
request_permissions([Permission.WRITE_EXTERNAL_STORAGE, Permission.READ_EXTERNAL_STORAGE, Permission.INTERNET, Permission.NET_ADMIN])
engine = pyttsx3.init()

def speak_objects(objects_in_front):
    if objects_in_front:
        engine.say(f"In front of you, there is {objects_in_front}")
    else:
        engine.say("No objects detected in front of you")
    engine.runAndWait()

# Load YOLO model
model = YOLO('yolov8n.pt')

# Class names for YOLO
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "pen"]

# Set up webcam
cap = cv2.VideoCapture(0) # Using the default webcam
cap.set(3, 1280)
cap.set(4, 720)

detection_interval = 10 # 10 seconds interval for object detection
objects_in_front = "" # Variable to hold detected objects

# Flag to control the main loop
running = False
detection_thread = None

# Function to perform object detection and update objects_in_front
def detect_objects():
    global objects_in_front
    prev_objects = ""
    while running:
        # Read frame from webcam
        success, img = cap.read()
        if not success:
            continue

        # Get detection results from YOLO model
        results = model(img)

        objects_in_front = ""
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy()
            
            for i in range(len(boxes)):
                cls = int(class_ids[i])
                conf = round(float(confidences[i]), 2)
                label = f'{classNames[cls]} {conf}'
                objects_in_front += label + ", "

        # Check if new objects are detected
        if objects_in_front != prev_objects:
            speak_objects(objects_in_front)  # Speak the detected objects
            prev_objects = objects_in_front

        # Check the running flag during sleep
        for _ in range(detection_interval):
            if not running:
                break
            time.sleep(1)

# Function to start object detection
def start_detection():
    global running, detection_thread
    if not running:
        engine.say("Object detection started.")
        engine.runAndWait()
        running = True
        detection_thread = threading.Thread(target=detect_objects, daemon=True)
        detection_thread.start()

# Function to stop object detection
def stop_detection():
    global running
    if running:
        engine.say("Stopping object detection.")
        engine.runAndWait()
        running = False

# Kivy App
class ObjectDetectionApp(App):

    def build(self):
        layout = GridLayout(cols=1)

        # Create labels to display instructions and detected objects
        instruction_label = Label(text="Press 'Start' to begin object detection and 'Stop' to end it.")
        layout.add_widget(instruction_label)

        self.objects_label = Label(text="")
        layout.add_widget(self.objects_label)

        # Create buttons to start and stop object detection
        start_button = Button(text="Start", on_press=self.start_detection)
        layout.add_widget(start_button)
        stop_button = Button(text="Stop", on_press=self.stop_detection)
        layout.add_widget(stop_button)

        # Update objects label periodically
        self.update_objects_label()
        return layout

    def speak_objects(self, objects_in_front):
        speak_objects(objects_in_front)

    # Function to update the detected objects label
    def update_objects_label(self):
        self.objects_label.text = objects_in_front
        threading.Timer(1, self.update_objects_label).start()

    def start_detection(self, instance):
        start_detection()

    def stop_detection(self, instance):
        stop_detection()

# Run the Kivy app
if __name__ == '__main__':
    ObjectDetectionApp().run()

# Release the webcam
cap.release()
