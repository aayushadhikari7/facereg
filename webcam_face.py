import sys
import cv2
import numpy as np
import time 

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QPushButton,
    QDialog, QHBoxLayout, QCheckBox, QTextEdit, QSizePolicy
)
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QMutex, QWaitCondition



try:
    import torch 
    from ultralytics import YOLO
except ImportError:
    print("Error: ultralytics library not found. Please install it using 'pip install ultralytics'")
    sys.exit(1)



class VideoStreamThread(QThread):
  
    raw_frame_signal = pyqtSignal(np.ndarray)
    error_signal = pyqtSignal(str)

    def __init__(self, rtsp_url):
        super().__init__()
        self.rtsp_url = rtsp_url
        self._run_flag = True
        self.cap = None
        self._last_frame_time = time.time()
        self._frame_count = 0

    def run(self):
        try:
            self.cap = cv2.VideoCapture(self.rtsp_url)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if not self.cap.isOpened():
                self.error_signal.emit(f"Error: Could not open video stream from {self.rtsp_url}. Please check the URL/device index, camera status, and network connectivity if applicable.")
                self._run_flag = False
                return

            while self._run_flag:
                ret, frame = self.cap.read()
                if ret:
                    self.raw_frame_signal.emit(frame)
                    
                    self._frame_count += 1
                    if self._frame_count % 30 == 0: # Update FPS every 30 frames
                        current_time = time.time()
                        elapsed_time = current_time - self._last_frame_time
                        if elapsed_time > 0:
                            fps = self._frame_count / elapsed_time
                            print(f"Video Capture FPS: {fps:.2f}")
                        self._last_frame_time = current_time
                        self._frame_count = 0

                else:
                    self.error_signal.emit("Error: Failed to read frame from stream. Stream may have ended or encountered an issue. Reconnecting may be necessary.")
                    break # Exit loop on read failure
                # Add a small delay to prevent consuming 100% CPU on frame reading
                QThread.msleep(1) # Sleep for 1ms to yield control

        except Exception as e:
            self.error_signal.emit(f"An unexpected error occurred in video stream: {e}")
        finally:
            if self.cap and self.cap.isOpened():
                self.cap.release()
            print("Video stream thread stopped.")

    def stop(self):
        self._run_flag = False
        self.wait() # Wait for the thread to finish execution


class ObjectDetectionThread(QThread):
    processed_pixmap_signal = pyqtSignal(QImage)
    error_signal = pyqtSignal(str)
    new_object_detected_signal = pyqtSignal(str, str) 

    LOG_COOLDOWN_PERIOD = 5 
    MOTION_LOG_COOLDOWN = 2 # Cooldown for motion log entry

    def __init__(self, model_path='yolov8x.pt'): # Using yolov8x.pt 
        super().__init__()
        self.model_path = model_path
        self.model = None
        self._detection_enabled = False
        self._motion_detection_enabled = False 
        self._run_flag = True 
        self.frame_queue = [] 
        self.queue_lock = QMutex() 
        self.queue_condition = QWaitCondition() 
        
        self.last_logged_time_per_class = {}
        self._last_detection_time = time.time()
        self._detection_frame_count = 0

        self._previous_gray_frame = None # For motion detection
        self._last_motion_log_time = 0 # Track last time motion was logged


        print(f"YOLO will attempt to use CUDA if available.")


    def run(self):
        try:
            print(f"Loading YOLO model from {self.model_path}...")
            self.model = YOLO(self.model_path)
            
            if torch.cuda.is_available():
                print("CUDA GPU detected by PyTorch. YOLO will likely use GPU.")
            else:
                print("CUDA GPU not detected by PyTorch. YOLO will use CPU.")

            print("YOLO model loaded.")

            while self._run_flag:
                self.queue_lock.lock()
                if not self.frame_queue:
                    self.queue_condition.wait(self.queue_lock)
                
                if not self._run_flag:
                    self.queue_lock.unlock()
                    break

                frame_bgr = self.frame_queue.pop() 
                self.frame_queue.clear() 
                self.queue_lock.unlock()

                annotated_frame = frame_bgr.copy() # Base frame for drawing

                if self._motion_detection_enabled and frame_bgr is not None:
                    gray_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                    gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)

                    if self._previous_gray_frame is None:
                        self._previous_gray_frame = gray_frame
                        pass 
                    else:
                        frame_delta = cv2.absdiff(self._previous_gray_frame, gray_frame)
                        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
                        thresh = cv2.dilate(thresh, None, iterations=2)
                        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                        motion_detected = False
                        for contour in contours:
                            if cv2.contourArea(contour) < 500:
                                continue
                            motion_detected = True
                            (x, y, w, h) = cv2.boundingRect(contour)
                            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 0, 255), 2) 

                        if motion_detected:
                            cv2.putText(annotated_frame, "Motion Detected!", (10, 30), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2) 
                            
                            current_time = time.time()
                            if (current_time - self._last_motion_log_time) > self.MOTION_LOG_COOLDOWN:
                                log_entry = f"[{time.strftime('%H:%M:%S')}] *** MAJOR MOTION DETECTED! ***" 
                                self.new_object_detected_signal.emit(log_entry, 'motion') # Emit with 'motion' style
                                self._last_motion_log_time = current_time
                    
                    self._previous_gray_frame = gray_frame 


                if self._detection_enabled and frame_bgr is not None:
                    results = self.model(frame_bgr, verbose=False, stream=False) 

                    current_time = time.time()
                    for r in results:
                        annotated_frame = r.plot(img=annotated_frame) 

                        for box in r.boxes:
                            confidence = float(box.conf[0])
                            class_id = int(box.cls[0])
                            
                            if class_id < len(self.model.names):
                                class_name = self.model.names[class_id]
                            else:
                                class_name = "Unknown" 

                            if confidence > 0.4:
                                if (class_name not in self.last_logged_time_per_class or
                                    (current_time - self.last_logged_time_per_class[class_name]) > self.LOG_COOLDOWN_PERIOD):
                                    
                                    log_entry = f"[{time.strftime('%H:%M:%S')}] Detected: {class_name} ({confidence:.2f})"
                                    self.new_object_detected_signal.emit(log_entry, 'default') # Emit with 'default' style
                                    self.last_logged_time_per_class[class_name] = current_time
                    
                    self._detection_frame_count += 1
                    if self._detection_frame_count % 10 == 0:
                        current_detection_time = time.time()
                        elapsed_detection_time = current_detection_time - self._last_detection_time
                        if elapsed_detection_time > 0:
                            fps = self._detection_frame_count / elapsed_detection_time
                            print(f"Object Detection FPS: {fps:.2f}")
                        self._last_detection_time = current_detection_time
                        self._detection_frame_count = 0

                if not self._detection_enabled and not self._motion_detection_enabled:
                    rgb_image = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                else:
                    rgb_image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                scaled_qt_image = qt_image.scaled(640, 480, Qt.KeepAspectRatio)
                self.processed_pixmap_signal.emit(scaled_qt_image)

        except Exception as e:
            self.error_signal.emit(f"An unexpected error occurred in detection thread: {e}")
        finally:
            print("Object/Motion detection thread stopped.")

    def process_frame(self, frame):
        self.queue_lock.lock()
        self.frame_queue.append(frame)
        self.queue_condition.wakeOne()
        self.queue_lock.unlock()

    def set_detection_enabled(self, enabled):
        self._detection_enabled = enabled
        self.queue_lock.lock()
        self.frame_queue.clear()
        self.queue_lock.unlock()
        self.last_logged_time_per_class.clear()

    def set_motion_detection_enabled(self, enabled):
        self._motion_detection_enabled = enabled
        self.queue_lock.lock()
        self.frame_queue.clear()
        self.queue_lock.unlock()
        self._previous_gray_frame = None 
        self._last_motion_log_time = 0 # Reset motion log cooldown on disable

    def stop(self):
        self._run_flag = False
        self.queue_lock.lock()
        self.queue_condition.wakeAll()
        self.queue_lock.unlock()
        self.wait()


class CustomMessageDialog(QDialog):
    def __init__(self, title, message, parent=None, is_error=False):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setFixedSize(400, 200)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Dialog)

        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        title_label = QLabel(title)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setFont(QFont("Inter", 16, QFont.Bold))
        if is_error:
            title_label.setStyleSheet("color: #f44336;")
        else:
            title_label.setStyleSheet("color: #e0e0e0;")
        layout.addWidget(title_label)

        message_label = QLabel(message)
        message_label.setAlignment(Qt.AlignCenter)
        message_label.setWordWrap(True)
        message_label.setFont(QFont("Inter", 10))
        message_label.setStyleSheet("color: #a0a0a0;")
        layout.addWidget(message_label)

        ok_button = QPushButton("OK")
        ok_button.setFixedSize(100, 35)
        ok_button.clicked.connect(self.accept)
        ok_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 6px;
                font-size: 14px;
                font-weight: 500;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3e8e41;
                box-shadow: 0 1px 2px rgba(0, 0, 0, 0.2);
            }
        """)
        
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        button_layout.addWidget(ok_button)
        button_layout.addStretch()
        layout.addLayout(button_layout)

        self.setLayout(layout)
        self.setStyleSheet("""
            QDialog {
                background-color: #3c3c3c;
                border-radius: 12px;
                border: 1px solid #555;
            }
        """)


class CameraFeedApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Electron Themed Camera Feed Viewer")
        self.setGeometry(100, 100, 1000, 750) 
        self.rtsp_url = 0 # Changed to use default webcam

        self.video_thread = None
        self.object_detection_thread = None
        
        self.is_detection_active = False
        self.is_motion_detection_active = False

        self.init_ui()
        self.apply_theme()

    def init_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(30, 30, 30, 30)
        main_layout.setSpacing(20)

        title_label = QLabel("Real-time Camera Security Feed")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setFont(QFont("Inter", 24, QFont.Bold))
        title_label.setStyleSheet("color: #4CAF50;") 
        main_layout.addWidget(title_label)

        self.video_label = QLabel("Waiting for camera feed...")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setFixedSize(640, 480) 
        self.video_label.setStyleSheet("""
            QLabel {
                background-color: #1a1a1a;
                color: #a0a0a0;
                border-radius: 8px;
                font-size: 18px;
                border: 1px solid #2a2a2a;
            }
        """)
        self.video_label.setFont(QFont("Inter", 18))
        main_layout.addWidget(self.video_label, alignment=Qt.AlignCenter)

        control_layout = QHBoxLayout()
        control_layout.setSpacing(15)

        self.start_button = QPushButton("Start Camera Feed")
        self.start_button.clicked.connect(self.start_feed)
        control_layout.addWidget(self.start_button)

        self.stop_button = QPushButton("Stop Camera Feed")
        self.stop_button.clicked.connect(self.stop_feed)
        self.stop_button.setEnabled(False)
        control_layout.addWidget(self.stop_button)

        self.detection_checkbox = QCheckBox("Enable Object Detection")
        self.detection_checkbox.setChecked(False)
        self.detection_checkbox.stateChanged.connect(self.toggle_detection)
        self.detection_checkbox.setStyleSheet("""
            QCheckBox {
                color: #e0e0e0;
                font-size: 16px;
                spacing: 10px;
            }
            QCheckBox::indicator {
                width: 20px;
                height: 20px;
                border-radius: 5px;
                border: 2px solid #555;
                background-color: #3c3c3c;
            }
            QCheckBox::indicator:checked {
                background-color: #4CAF50;
                border: 2px solid #4CAF50;
            }
            QCheckBox::indicator:hover {
                border: 2px solid #777;
            }
        """)
        control_layout.addWidget(self.detection_checkbox)
        
        self.motion_detection_checkbox = QCheckBox("Enable Motion Detection")
        self.motion_detection_checkbox.setChecked(False)
        self.motion_detection_checkbox.stateChanged.connect(self.toggle_motion_detection)
        self.motion_detection_checkbox.setStyleSheet("""
            QCheckBox {
                color: #e0e0e0;
                font-size: 16px;
                spacing: 10px;
            }
            QCheckBox::indicator {
                width: 20px;
                height: 20px;
                border-radius: 5px;
                border: 2px solid #555;
                background-color: #3c3c3c;
            }
            QCheckBox::indicator:checked {
                background-color: #007bff; /* Blue for motion detection */
                border: 2px solid #007bff;
            }
            QCheckBox::indicator:hover {
                border: 2px solid #777;
            }
        """)
        control_layout.addWidget(self.motion_detection_checkbox)


        main_layout.addLayout(control_layout)

        self.history_log = QTextEdit()
        self.history_log.setReadOnly(True)
        self.history_log.setPlaceholderText("Object and motion detection history will appear here...")
        self.history_log.setStyleSheet("""
            QTextEdit {
                background-color: #1a1a1a;
                color: #e0e0e0;
                border-radius: 8px;
                padding: 10px;
                font-size: 14px;
                border: 1px solid #2a2a2a;
            }
        """)
        self.history_log.setFont(QFont("Inter", 10))
        self.history_log.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        main_layout.addWidget(self.history_log)

        self.setLayout(main_layout)

    def apply_theme(self):
        QApplication.setStyle("Fusion") 
        self.setStyleSheet("""
            QWidget {
                background-color: #2e2e2e;
                color: #e0e0e0;
                font-family: 'Inter', sans-serif;
            }
            QPushButton {
                padding: 12px 28px;
                border: none;
                border-radius: 8px;
                font-size: 16px;
                font-weight: 500;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
                transition: all 0.2s ease-in-out;
            }
            QPushButton#start_button {
                background-color: #4CAF50;
                color: white;
            }
            QPushButton#start_button:hover {
                background-color: #45a049;
                box-shadow: 0 6px 12px rgba(0, 0, 0, 0.3);
            }
            QPushButton#start_button:pressed {
                background-color: #3e8e41;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
            }
            QPushButton#stop_button {
                background-color: #f44336;
                color: white;
            }
            QPushButton#stop_button:hover {
                background-color: #da190b;
                box-shadow: 0 6px 12px rgba(0, 0, 0, 0.3);
            }
            QPushButton#stop_button:pressed {
                background-color: #b00000;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
            }
            QPushButton:disabled {
                opacity: 0.6;
                background-color: #555;
                color: #bbb;
                box-shadow: none;
            }
        """)
        self.start_button.setObjectName("start_button")
        self.stop_button.setObjectName("stop_button")


    def start_feed(self):
        if self.video_thread is None or not self.video_thread.isRunning():
            self.video_label.setText("Attempting to connect to camera...")
            self.video_label.setPixmap(QPixmap())
            self.history_log.clear()

            self.video_thread = VideoStreamThread(self.rtsp_url)
            self.video_thread.error_signal.connect(self.display_error)
            
            self.object_detection_thread = ObjectDetectionThread(model_path='yolov8x.pt') 
            self.object_detection_thread.error_signal.connect(self.display_error)
            self.object_detection_thread.processed_pixmap_signal.connect(self.update_image)
            self.object_detection_thread.new_object_detected_signal.connect(self.update_history_log_styled)
            self.object_detection_thread.start()

            self.video_thread.raw_frame_signal.connect(self.object_detection_thread.process_frame)

            self.object_detection_thread.set_detection_enabled(self.detection_checkbox.isChecked())
            self.object_detection_thread.set_motion_detection_enabled(self.motion_detection_checkbox.isChecked())

            self.video_thread.start()

            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.detection_checkbox.setEnabled(True)
            self.motion_detection_checkbox.setEnabled(True)
        else:
            self.show_custom_message("Info", "Camera feed is already running.")


    def stop_feed(self):
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.stop()
            self.video_thread = None
        
        if self.object_detection_thread and self.object_detection_thread.isRunning():
            self.object_detection_thread.stop()
            self.object_detection_thread = None

        self.video_label.setText("Camera feed stopped.")
        self.video_label.setPixmap(QPixmap())
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.detection_checkbox.setEnabled(False)
        self.detection_checkbox.setChecked(False)
        self.motion_detection_checkbox.setEnabled(False)
        self.motion_detection_checkbox.setChecked(False)
        self.history_log.clear()


    def update_image(self, qt_image):
        """Updates the image_label with a new QImage (potentially with detections)"""
        self.video_label.setPixmap(QPixmap.fromImage(qt_image))

    def update_history_log_styled(self, log_entry, style_type='default'):
        """Appends a new entry to the object detection history log with specific styling."""
        html_entry = ""
        if style_type == 'motion':
            html_entry = f'<span style="color: red; font-weight: bold;">{log_entry}</span>'
        else: # default
            html_entry = f'<span style="color: #e0e0e0;">{log_entry}</span>'

        self.history_log.append(html_entry)


    def toggle_detection(self, state):
        self.is_detection_active = (state == Qt.Checked)
        if self.object_detection_thread:
            self.object_detection_thread.set_detection_enabled(self.is_detection_active)
        
        if self.is_detection_active:
            print("Object detection enabled.")
        else:
            print("Object detection disabled.")

    def toggle_motion_detection(self, state):
        self.is_motion_detection_active = (state == Qt.Checked)
        if self.object_detection_thread:
            self.object_detection_thread.set_motion_detection_enabled(self.is_motion_detection_active)

        if self.is_motion_detection_active:
            print("Motion detection enabled.")
            self.history_log.append(f"[{time.strftime('%H:%M:%S')}] Motion Detection Enabled.")
        else:
            print("Motion detection disabled.")
            self.history_log.append(f"[{time.strftime('%H:%M:%S')}] Motion Detection Disabled.")


    def display_error(self, message):
        self.show_custom_message("Application Error", message, is_error=True)
        self.video_label.setText(f"Error: {message}")
        self.stop_feed()


    def show_custom_message(self, title, message, is_error=False):
        dialog = CustomMessageDialog(title, message, self, is_error)
        dialog.exec_()

    def closeEvent(self, event):
        self.stop_feed()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CameraFeedApp()
    window.show()
    sys.exit(app.exec_())
