import sys
import cv2
import pyautogui
import numpy as np
import math
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QRadioButton, QGroupBox, QSizePolicy
)
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot


try:
    from comtypes import CLSCTX_ALL
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
    from ctypes import cast, POINTER
except ImportError:
    print("pycaw not found. Volume control will be disabled. Install with: pip install pycaw comtypes")
    IAudioEndpointVolume = None 


import platform
import subprocess


pyautogui.FAILSAFE = False 


MIN_HAND_AREA = 5000 
MAX_HAND_AREA = 100000 
GRASP_CONVEXITY_DEFECT_THRESHOLD = 8 
PINCH_FINGER_TIP_DIST_THRESHOLD_RATIO = 0.08 
SCROLL_SENSITIVITY = 5 
VOLUME_SENSITIVITY = 1.0 

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage, object)
    processing_fps_signal = pyqtSignal(float)

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.cap = None
        self.active_feature = "None"

        self.prev_hand_y = None 
        self.initial_volume_hand_x = None 

    def run(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open video stream. Please ensure webcam is connected and not in use.")
            self._run_flag = False
            return

        prev_frame_time = 0
        new_frame_time = 0

        while self._run_flag:
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Failed to grab frame. Stream might have ended or connection lost.")
                break

            frame = cv2.flip(frame, 1)
            h, w, ch = frame.shape
            bytes_per_line = ch * w

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lower_skin = np.array([0, 20, 70], dtype=np.uint8)
            upper_skin = np.array([20, 255, 255], dtype=np.uint8)

            mask = cv2.inRange(hsv, lower_skin, upper_skin)
            kernel = np.ones((5,5), np.uint8)
            mask = cv2.erode(mask, kernel, iterations=1)
            mask = cv2.dilate(mask, kernel, iterations=2)
            mask = cv2.GaussianBlur(mask, (5, 5), 0)

            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            largest_contour = None
            max_area = 0

            for contour in contours:
                area = cv2.contourArea(contour)
                if MIN_HAND_AREA < area < MAX_HAND_AREA:
                    if area > max_area:
                        max_area = area
                        largest_contour = contour

            hand_data_for_gestures = None
            if largest_contour is not None:
                cv2.drawContours(frame, [largest_contour], -1, (0, 0, 255), 2)

                hull = cv2.convexHull(largest_contour, returnPoints=False)
                defects = cv2.convexityDefects(largest_contour, hull)

                hand_data_for_gestures = {
                    "contour": largest_contour,
                    "hull": hull,
                    "defects": defects,
                    "bbox": cv2.boundingRect(largest_contour),
                    "area": max_area
                }

                if self.active_feature != "None":
                    self.process_gestures(hand_data_for_gestures, w, h)

            new_frame_time = cv2.getTickCount()
            fps = 1 / ((new_frame_time - prev_frame_time) / cv2.getTickFrequency())
            prev_frame_time = new_frame_time
            self.processing_fps_signal.emit(fps)

            qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_BGR888)
            self.change_pixmap_signal.emit(qt_image, hand_data_for_gestures)

        self.cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()

    def set_active_feature(self, feature_name):
        self.active_feature = feature_name
        self.prev_hand_y = None 
        self.initial_volume_hand_x = None 

    def process_gestures(self, hand_data, frame_w, frame_h):
        contour = hand_data["contour"]
        defects = hand_data["defects"]
        x, y, w, h = hand_data["bbox"]
        area = hand_data["area"]

        is_grasping = False
        if defects is not None:
            finger_count = 0
            for i in range(defects.shape[0]):
                s,e,f,d = defects[i,0]
                start = tuple(contour[s][0])
                end = tuple(contour[e][0])
                far = tuple(contour[f][0])

                a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
                b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
                c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)

                if a == 0 or b == 0:
                    continue

                angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 180 / math.pi

                if angle <= 90 and d > 20:
                    finger_count += 1
            
            is_grasping = finger_count < GRASP_CONVEXITY_DEFECT_THRESHOLD

        aspect_ratio = w / h if h != 0 else 0
        is_pinching = (area < MIN_HAND_AREA * 1.5) and (aspect_ratio > 0.5 and aspect_ratio < 1.5)

        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = x + w // 2, y + h // 2

        if self.active_feature == "Mouse Control":
            pyautogui.moveTo(cx, cy)

            if is_pinching:
                pyautogui.click()
                print("Left Click (Approximated)")
            elif is_grasping:
                pyautogui.rightClick()
                print("Right Click (Approximated)")

            if self.prev_hand_y is not None:
                delta_y = cy - self.prev_hand_y
                if abs(delta_y) > SCROLL_SENSITIVITY:
                    scroll_amount = -delta_y // 5
                    pyautogui.scroll(scroll_amount)
                    print(f"Scroll: {scroll_amount}")
            self.prev_hand_y = cy

        elif self.active_feature == "Volume Control":
            if is_pinching:
                if self.initial_volume_hand_x is None:
                    self.initial_volume_hand_x = cx 
                
                delta_x = cx - self.initial_volume_hand_x
                if abs(delta_x) > 5:
                    current_volume = self.get_system_volume()
                    if current_volume is not None:
                        new_volume = current_volume + (delta_x * VOLUME_SENSITIVITY / 100.0)
                        new_volume = max(0, min(1, new_volume))
                        self.set_system_volume(new_volume)
                        print(f"Volume: {int(new_volume*100)}%")
                    self.initial_volume_hand_x = cx
            else:
                self.initial_volume_hand_x = None

    def get_system_volume(self):
        if platform.system() == "Windows" and IAudioEndpointVolume:
            try:
                devices = AudioUtilities.GetSpeakers()
                interface = devices.Activate(
                    IAudioEndpointVolume._iid_, CLSCTX_ALL, None
                )
                volume = cast(interface, POINTER(IAudioEndpointVolume))
                return volume.GetMasterVolumeLevelScalar()
            except Exception as e:
                print(f"Error getting volume (Windows): {e}")
                return None
        elif platform.system() == "Linux":
            try:
                result = subprocess.run(['amixer', 'get', 'Master'], capture_output=True, text=True)
                if result.returncode == 0:
                    for line in result.stdout.splitlines():
                        if '[%]' in line and 'Playback' in line:
                            vol_percent = int(line.split('[')[1].split('%')[0])
                            return vol_percent / 100.0
                return None
            except Exception as e:
                print(f"Error getting volume (Linux): {e}")
                return None
        else:
            print("Volume control not supported on this OS or pycaw/amixer not available.")
            return None

    def set_system_volume(self, volume_level):
        if platform.system() == "Windows" and IAudioEndpointVolume:
            try:
                devices = AudioUtilities.GetSpeakers()
                interface = devices.Activate(
                    IAudioEndpointVolume._iid_, CLSCTX_ALL, None
                )
                volume = cast(interface, POINTER(IAudioEndpointVolume))
                volume.SetMasterVolumeLevelScalar(volume_level, None)
            except Exception as e:
                print(f"Error setting volume (Windows): {e}")
        elif platform.system() == "Linux":
            try:
                vol_percent = int(volume_level * 100)
                subprocess.run(['amixer', 'set', 'Master', f'{vol_percent}%'], capture_output=True)
            except Exception as e:
                print(f"Error setting volume (Linux): {e}")
        else:
            print("Volume control not supported on this OS or pycaw/amixer not available.")


class HandGestureControlApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hand Gesture Control")
        self.setGeometry(100, 100, 1000, 700)
        self.init_ui()

        self.video_thread = VideoThread()
        self.video_thread.change_pixmap_signal.connect(self.update_image)
        self.video_thread.processing_fps_signal.connect(self.update_fps)

        self.current_feature = "None"
        self.update_feature_status()

    def init_ui(self):
        main_layout = QHBoxLayout()
        self.setLayout(main_layout)

        control_panel_layout = QVBoxLayout()
        control_panel_layout.setAlignment(Qt.AlignTop)
        main_layout.addLayout(control_panel_layout, 1)

        title_label = QLabel("Hand Gesture Control")
        title_label.setFont(QFont("Inter", 24, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("color: #4CAF50;")
        control_panel_layout.addWidget(title_label)
        control_panel_layout.addSpacing(20)

        feature_group = QGroupBox("Select Feature")
        feature_group.setFont(QFont("Inter", 12))
        feature_group_layout = QVBoxLayout()
        feature_group.setLayout(feature_group_layout)
        
        self.radio_none = QRadioButton("None (Display Only)")
        self.radio_mouse = QRadioButton("Mouse Control (Approximate Pinch: Left Click, Approximate Grasp: Right Click, Flick: Scroll)")
        self.radio_volume = QRadioButton("Volume Control (Approximate Pinch + Slide: Adjust Volume)")

        self.radio_none.setChecked(True)
        
        self.radio_none.toggled.connect(lambda: self.set_feature("None"))
        self.radio_mouse.toggled.connect(lambda: self.set_feature("Mouse Control"))
        self.radio_volume.toggled.connect(lambda: self.set_feature("Volume Control"))

        feature_group_layout.addWidget(self.radio_none)
        feature_group_layout.addWidget(self.radio_mouse)
        feature_group_layout.addWidget(self.radio_volume)
        control_panel_layout.addWidget(feature_group)
        control_panel_layout.addSpacing(20)

        self.status_label = QLabel("Status: Ready")
        self.status_label.setFont(QFont("Inter", 10))
        self.status_label.setStyleSheet("color: #2196F3;")
        control_panel_layout.addWidget(self.status_label)

        self.fps_label = QLabel("FPS: 0.0")
        self.fps_label.setFont(QFont("Inter", 10))
        control_panel_layout.addWidget(self.fps_label)
        control_panel_layout.addStretch(1)

        video_panel_layout = QVBoxLayout()
        main_layout.addLayout(video_panel_layout, 3)

        self.image_label = QLabel()
        self.image_label.setFixedSize(640, 480)
        self.image_label.setStyleSheet("background-color: black; border: 2px solid #555; border-radius: 10px;")
        self.image_label.setAlignment(Qt.AlignCenter)
        video_panel_layout.addWidget(self.image_label)
        
        self.start_stop_button = QPushButton("Start Webcam")
        self.start_stop_button.setFont(QFont("Inter", 14, QFont.Bold))
        self.start_stop_button.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; border-radius: 8px; padding: 10px 20px; }"
            "QPushButton:hover { background-color: #45a049; }"
        )
        self.start_stop_button.clicked.connect(self.toggle_webcam)
        video_panel_layout.addWidget(self.start_stop_button)

        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setScaledContents(True)

    def toggle_webcam(self):
        if self.video_thread.isRunning():
            self.video_thread.stop()
            self.start_stop_button.setText("Start Webcam")
            self.start_stop_button.setStyleSheet(
                "QPushButton { background-color: #4CAF50; color: white; border-radius: 8px; padding: 10px 20px; }"
                "QPushButton:hover { background-color: #45a049; }"
            )
            self.status_label.setText("Status: Webcam Stopped")
            self.image_label.clear()
        else:
            self.video_thread.start()
            self.start_stop_button.setText("Stop Webcam")
            self.start_stop_button.setStyleSheet(
                "QPushButton { background-color: #F44336; color: white; border-radius: 8px; padding: 10px 20px; }"
                "QPushButton:hover { background-color: #D32F2F; }"
            )
            self.status_label.setText("Status: Webcam Running")

    @pyqtSlot(QImage, object)
    def update_image(self, qt_image, hand_data):
        if not self.video_thread._run_flag:
            return

        pixmap = QPixmap.fromImage(qt_image)
        self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    @pyqtSlot(float)
    def update_fps(self, fps):
        self.fps_label.setText(f"FPS: {fps:.1f}")

    def set_feature(self, feature_name):
        self.current_feature = feature_name
        self.video_thread.set_active_feature(feature_name)
        self.update_feature_status()

    def update_feature_status(self):
        self.status_label.setText(f"Status: Webcam Running | Active Feature: {self.current_feature}")
        if not self.video_thread.isRunning():
            self.status_label.setText("Status: Webcam Stopped")


    def closeEvent(self, event):
        self.video_thread.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = HandGestureControlApp()
    window.show()
    sys.exit(app.exec_())
