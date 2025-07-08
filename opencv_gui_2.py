import sys
import time
import numpy as np
import cv2
import onnxruntime as ort
from PyQt5.QtWidgets import (
    QApplication, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QWidget, QHBoxLayout, QSpacerItem, QSizePolicy
)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
from PIL import Image
from hand_tracker2 import HandTracker

image_count = 0
class StyleTransferApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-time Neural Style Transfer")
        self.resize(1000, 700)

        # ====== 樣式設定 ======
        self.setStyleSheet("""
            QWidget {
                background-color: #f0f2f5;
                font-family: "Helvetica Neue", "Microsoft JhengHei", sans-serif;
                font-size: 15px;
                color: #333;
            }
            QLabel#titleLabel {
                font-size: 26px;
                font-weight: bold;
                color: #2c3e50;
            }
            QLabel#statusLabel {
                margin-bottom: 10px;
                font-size: 16px;
            }
            QPushButton {
                background-color: #3498db;
                color: white;
                padding: 8px 20px;
                border: none;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:disabled {
                background-color: #bdc3c7;
            }
        """)

        # ====== UI 元件 ======
        self.title = QLabel("🎨 Real-time Neural Style Transfer")
        self.title.setObjectName("titleLabel")
        self.title.setAlignment(Qt.AlignCenter)

        self.label = QLabel("請上傳一張以上的風格圖片")
        self.label.setObjectName("statusLabel")
        self.label.setAlignment(Qt.AlignCenter)

        self.label2 = QLabel("")
        self.label2.setObjectName("statusLabel")
        self.label2.setAlignment(Qt.AlignCenter)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFixedSize(640, 480)
        self.image_label.setStyleSheet("background-color: white; border: 1px solid #ccc;")

        self.upload_btn = QPushButton("上傳多張風格圖片")
        self.upload_btn.clicked.connect(self.upload_style_images)

        self.start_btn = QPushButton("開始風格轉換")
        self.start_btn.clicked.connect(self.start_video)
        self.start_btn.setEnabled(False)

        self.next_btn = QPushButton("下一張風格")
        self.next_btn.clicked.connect(self.next_style)
        self.next_btn.setEnabled(False)

        self.stop_btn = QPushButton("停止風格轉換")
        self.stop_btn.clicked.connect(self.stop_video)
        self.stop_btn.setEnabled(False)

        # ====== 版面配置 ======
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        button_layout.addWidget(self.upload_btn)
        button_layout.addWidget(self.start_btn)
        button_layout.addWidget(self.next_btn)
        button_layout.addWidget(self.stop_btn)
        button_layout.addStretch()

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.title)
        main_layout.addSpacing(10)
        main_layout.addWidget(self.label)
        main_layout.addWidget(self.label2)
        main_layout.addLayout(button_layout)
        main_layout.addSpacing(20)
        main_layout.addWidget(self.image_label, alignment=Qt.AlignCenter)
        self.setLayout(main_layout)

        # ====== ONNX 模型與變數 ======
        self.session = self.load_onnx_model()
        self.input_names = [i.name for i in self.session.get_inputs()]
        self.label.setText(f" 模型已載入，使用: {self.session.get_providers()[0]}")
        print("\n\n\n使用裝置：", self.session.get_providers()[0])
        print("\n\n\n")
        # ====== 攝影機 + 計時器 ======
        self.capture = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.prev_time = time.time()
        self.frame_count = 0

        # ====== 風格圖片 ======
        self.style_images = []
        self.style_names = []
        self.style_index = 0

        # ====== 手勢偵測器 ======
        self.hand_tracker = HandTracker()
        self.last_gesture_time = time.time()
        self.last_scisssort_time = time.time()
        self.cooldown_seconds = 3
        self.scissort_cooldown_seconds = 4
        self.to_saved = 0
        self.photo_time = 0
        self.sc_buffer = 0

    def load_onnx_model(self):
        try:
            session = ort.InferenceSession("stylization_simplified.onnx", providers=[
                "CUDAExecutionProvider", "CPUExecutionProvider"
            ])
        except:
            session = ort.InferenceSession("stylization_simplified.onnx", providers=["CPUExecutionProvider"])
        return session

    def upload_style_images(self):
        files, _ = QFileDialog.getOpenFileNames(self, "選擇風格圖片", "", "Images (*.png *.jpg *.jpeg)")
        if files:
            self.style_images.clear()
            self.style_names.clear()
            for file_path in files:
                img = Image.open(file_path).convert('RGB')
                img_np = np.array(img).astype(np.float32) / 255.
                img_np = cv2.resize(img_np, (256, 256))
                img_np = np.expand_dims(img_np, axis=0).astype(np.float32)
                self.style_images.append(img_np)
                self.style_names.append(file_path.split('/')[-1])
            self.style_index = 0
            self.label.setText(f" 載入 {len(self.style_images)} 張風格圖片 | 當前: {self.style_names[self.style_index]}")
            self.start_btn.setEnabled(True)
            self.next_btn.setEnabled(True)

    def start_video(self):
        self.capture = cv2.VideoCapture(0)
        if not self.capture.isOpened():
            self.label.setText("❌ 無法啟動攝影機")
            return
        self.timer.start(1)
        self.label.setText(f"風格轉換進行中... 當前: {self.style_names[self.style_index]}")
        self.stop_btn.setEnabled(True)
        self.start_btn.setEnabled(False)
        self.upload_btn.setEnabled(False)

    def stop_video(self):
        self.timer.stop()
        if self.capture:
            self.capture.release()
        self.image_label.clear()
        self.label.setText("已停止風格轉換，請重新上傳或開始")
        self.label2.setText("")
        self.stop_btn.setEnabled(False)
        self.start_btn.setEnabled(True)
        self.upload_btn.setEnabled(True)

    def next_style(self):
        if self.style_images:
            self.style_index = (self.style_index + 1) % len(self.style_images)
            self.label.setText(f"目前風格：{self.style_names[self.style_index]}")

    def update_frame(self):
        ret, frame = self.capture.read()
        if not ret or not self.style_images:
            return

        self.frame_count += 1
        current_time = time.time()
        elapsed = current_time - self.prev_time
        if elapsed >= 1.0:
            fps = self.frame_count / elapsed
            self.label.setText(f"FPS: {fps:.2f} | 使用: {self.session.get_providers()[0]} | 當前風格：{self.style_names[self.style_index]}")
            self.prev_time = current_time
            self.frame_count = 0

        # 偵測手勢（加上冷卻）
        if time.time() - self.last_gesture_time > self.cooldown_seconds:
            if self.hand_tracker.is_open_hand(frame):
                if self.sc_buffer > 2:
                    self.next_style()
                    self.last_gesture_time = time.time()
                    self.sc_buffer = 0
                else:
                    self.sc_buffer += 1
        if time.time() - self.last_scisssort_time > self.scissort_cooldown_seconds:    
            global image_count
            if self.hand_tracker.is_scissor_hand(frame):
                self.to_saved = 1
                self.last_scisssort_time = time.time()

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        content_tensor = cv2.resize(frame_rgb, (256, 256)).astype(np.float32) / 255.
        content_tensor = np.expand_dims(content_tensor, axis=0)

        result = self.session.run(None, {
            self.input_names[0]: content_tensor,
            self.input_names[1]: self.style_images[self.style_index]
        })

        stylized_frame = (result[0][0] * 255).astype(np.uint8)
        h, w, ch = stylized_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(stylized_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image).scaled(640, 480)
        self.image_label.setPixmap(pixmap)
        if self.to_saved == 1:
            self.photo_time = time.time() + 5
            self.to_saved = 0

        if self.photo_time > time.time():
            countdown = int(self.photo_time - time.time()) + 1
            self.label2.setText(f"Counting down ... {countdown}")

        if time.time() >= self.photo_time and self.photo_time > 0:
            filename = f"saved_image/Image{image_count}.jpg"
            bgr_frame = cv2.cvtColor(stylized_frame, cv2.COLOR_RGB2BGR)
            resized_frame = cv2.resize(bgr_frame, (512, 512))  # ← 新增這行
            cv2.imwrite(filename, resized_frame)
            self.label2.setText(f"image save to saved_image/Image{image_count}.jpg")
            print(f"Saved {filename}")
            image_count += 1
            cv2.putText(frame, "Scissor hand detected - Image saved", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            self.photo_time = -1

    def closeEvent(self, event):
        self.stop_video()
        self.hand_tracker.close()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = StyleTransferApp()
    window.show()
    sys.exit(app.exec_())
