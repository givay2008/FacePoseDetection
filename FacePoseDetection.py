"""
使用MediaPipe人脸关键点，通过脸部中心与鼻尖的偏移量计算角度
纯几何计算，无需深度学习，速度快
"""

import cv2
import mediapipe as mp
from mediapipe import tasks
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import urllib.request
import os
import sys
import tkinter as tk
from threading import Thread

# 设置UTF-8编码
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

class MovableWindow:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Direction Window")
        self.window_width = 300
        self.window_height = 200
        self.screen_width = self.root.winfo_screenwidth()
        self.screen_height = self.root.winfo_screenheight()
        self.root.geometry(f"{self.window_width}x{self.window_height}")
        self.root.attributes('-topmost', True)

        self.label = tk.Label(self.root, text="CENTER", font=("Arial", 24, "bold"), bg="white", fg="black")
        self.label.pack(expand=True, fill='both')
        self.current_direction = None

        self.positions = { "LEFT_UP": ("LEFT-UP", 0, 0),
            "RIGHT_UP": ("RIGHT-UP", self.screen_width - self.window_width, 0),
            "UP": ("UP", (self.screen_width - self.window_width) // 2, 0),
            "LEFT": ("LEFT", 0, (self.screen_height - self.window_height) // 2),
            "RIGHT": ("RIGHT", self.screen_width - self.window_width, (self.screen_height - self.window_height) // 2),
            "CENTER": ("CENTER", (self.screen_width - self.window_width) // 2, (self.screen_height - self.window_height) // 2) }

        self.colors = {
            "LEFT_UP": "#00FFFF",     # 青色
            "RIGHT_UP": "#FF00FF",    # 紫色
            "UP": "#FFFF00",          # 黄色
            "LEFT": "#00FF00",        # 绿色
            "RIGHT": "#FF0000",       # 红色
            "CENTER": "#FFFFFF"       # 白色
        }

        self.move_to("CENTER")

    def move_to(self, direction):
        if direction in self.positions:
            name, x, y = self.positions[direction]
            self.root.geometry(f"{self.window_width}x{self.window_height}+{x}+{y}")
            self.label.config( text=name, bg=self.colors.get(direction, "#FFFFFF") )
            self.current_direction = direction

    def update_direction(self, direction):
        if direction and direction != self.current_direction:
            self.move_to(direction)

    def update(self):
        try:
            self.root.update()
        except:
            pass

    def destroy(self):
        try:
            self.root.destroy()
        except:
            pass


class FacePoseDetector:
    """
    人脸姿态检测
    """

    def __init__(self):
        # 下载或加载模型
        model_path = 'face_landmarker.task'
        if not os.path.exists(model_path):
            print("Downloading Face Landmarker model...")
            model_url = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task'
            try:
                urllib.request.urlretrieve(model_url, model_path)
                print("Model downloaded successfully!")
            except Exception as e:
                print(f"Model download failed: {e}")
                raise

        # 创建检测器
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

        # MediaPipe 478点模型
        # 水平检测（左右转头）
        self.left_face_boundary = 234    # 左侧脸部边界点
        self.right_face_boundary = 454   # 右侧脸部边界点
        self.nose_tip = 1                # 鼻尖

        # 垂直检测（抬头）- 新增
        self.forehead_top = 10           # 额头顶部
        self.chin_bottom = 152           # 下巴底部

        # === 偏移阈值设置 ===
        self.h_offset_threshold = 0.03   # 水平偏移阈值
        self.v_ratio_threshold = 0.65    # 垂直比例阈值

        # 防抖动设置
        self.prev_direction = None
        self.direction_counter = 0
        self.stability_frames = 3        # 需要连续3帧相同才确认

        # 历史偏移量（用于平滑）
        self.offset_history = []
        self.history_size = 5

    def calculate_metrics(self, image):
        """
        计算水平和垂直
        """
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        detection_result = self.detector.detect(mp_image)

        if not detection_result.face_landmarks:
            return None, None, None, None, None

        face_landmarks = detection_result.face_landmarks[0]

        # 水平偏移计算
        left_boundary = face_landmarks[self.left_face_boundary]
        right_boundary = face_landmarks[self.right_face_boundary]
        nose = face_landmarks[self.nose_tip]

        left_x = left_boundary.x
        right_x = right_boundary.x
        nose_x = nose.x

        # 计算中心点和偏移量
        face_center_x = (left_x + right_x) / 2
        h_offset = nose_x - face_center_x

        # 垂直比例计算
        forehead = face_landmarks[self.forehead_top]
        chin = face_landmarks[self.chin_bottom]

        forehead_y = forehead.y
        chin_y = chin.y

        # 计算脸部高度和宽度
        face_height = abs(chin_y - forehead_y)
        face_width = abs(right_x - left_x)

        # 计算纵横比
        if face_height > 0:
            v_ratio = face_width / face_height
        else:
            v_ratio = 1.0

        # 偏移量平滑
        self.offset_history.append(h_offset)
        if len(self.offset_history) > self.history_size:
            self.offset_history.pop(0)

        smoothed_h_offset = np.mean(self.offset_history)

        return smoothed_h_offset, face_center_x, nose_x, v_ratio, face_landmarks

    def recognize_direction(self, h_offset, v_ratio):
        """
        水平偏移量和垂直比例判断方向
        """
        if h_offset is None or v_ratio is None:
            self.prev_direction = None
            self.direction_counter = 0
            return None

        current_direction = None

        # 判断是否抬头
        is_up = v_ratio > self.v_ratio_threshold

        # 判断水平方向
        is_left = h_offset < -self.h_offset_threshold
        is_right = h_offset > self.h_offset_threshold

        print(f"[DETECT] h_offset={h_offset:+.4f} (th={self.h_offset_threshold:.3f}), "
              f"v_ratio={v_ratio:.4f} (th={self.v_ratio_threshold:.3f}), "
              f"is_left={is_left}, is_right={is_right}, is_up={is_up}")

        if is_up and is_left:
            current_direction = "LEFT_UP"
            print(f"[DEBUG] LEFT_UP")
        elif is_up and is_right:
            current_direction = "RIGHT_UP"
            print(f"[DEBUG] RIGHT_UP")
        elif is_up and not is_left and not is_right:
            current_direction = "UP"
            print(f"[DEBUG] UP")
        elif is_left and not is_up:
            current_direction = "LEFT"
        elif is_right and not is_up:
            current_direction = "RIGHT"
        else:
            current_direction = "CENTER"

        if current_direction == self.prev_direction:
            self.direction_counter = 0
            return self.prev_direction
        else:
            self.direction_counter += 1
            if self.direction_counter >= self.stability_frames:
                self.prev_direction = current_direction
                self.direction_counter = 0
                return current_direction
            else:
                return self.prev_direction

    def draw_visualization(self, image, h_offset, face_center_x, nose_x, v_ratio, direction, landmarks):
        if h_offset is None:
            img_h, img_w = image.shape[:2]
            cv2.putText(image, "NO FACE DETECTED", (img_w // 2 - 200, img_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            return image

        img_h, img_w = image.shape[:2]

        # 左上
        cv2.rectangle(image, (0, 0), (400, 280), (0, 0, 0), -1)
        cv2.putText(image, "=== Geometric Detection ===", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                   
        # 水平
        cv2.putText(image, f"H-Offset: {h_offset:+.4f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        h_threshold_px = self.h_offset_threshold * img_w
        cv2.putText(image, f"H-Threshold: {self.h_offset_threshold:.3f} ({h_threshold_px:.0f}px)", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        # 垂直
        v_color = (0, 255, 0) if v_ratio > self.v_ratio_threshold else (100, 100, 100)
        cv2.putText(image, f"V-Ratio: {v_ratio:.4f}", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, v_color, 2)
        cv2.putText(image, f"V-Threshold: {self.v_ratio_threshold:.3f}", (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        is_up = v_ratio > self.v_ratio_threshold 
        is_left = h_offset < -self.h_offset_threshold
        is_right = h_offset > self.h_offset_threshold
        up_text = "UP" if is_up else "Level"
        up_color = (0, 255, 0) if is_up else (100, 100, 100)
        cv2.putText(image, f"Status: {up_text}", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, up_color, 2)

        h_dir = "LEFT" if is_left else "RIGHT" if is_right else "CENTER"
        h_color = (0, 255, 0) if is_left else (255, 0, 0) if is_right else (150, 150, 150)
        cv2.putText(image, f"H-Dir: {h_dir}", (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.5, h_color, 2)

        if is_up and is_left:
            raw_dir = "LEFT_UP"
        elif is_up and is_right:
            raw_dir = "RIGHT_UP"
        elif is_up and not is_left and not is_right:
            raw_dir = "UP"
        elif is_left:
            raw_dir = "LEFT"
        elif is_right:
            raw_dir = "RIGHT"
        else:
            raw_dir = "CENTER"
        cv2.putText(image, f"Raw: {raw_dir}", (10, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        if landmarks:
            # 左边界点
            left_point = landmarks[self.left_face_boundary]
            cv2.circle(image, (int(left_point.x * img_w), int(left_point.y * img_h)), 8, (0, 255, 0), -1)

            # 右边界点
            right_point = landmarks[self.right_face_boundary]
            cv2.circle(image, (int(right_point.x * img_w), int(right_point.y * img_h)), 8, (255, 0, 0), -1)

            # 脸部中心点
            center_y = (left_point.y + right_point.y) / 2
            cv2.circle(image, (int(face_center_x * img_w), int(center_y * img_h)), 10, (255, 255, 255), -1)
            cv2.circle(image, (int(face_center_x * img_w), int(center_y * img_h)), 10, (0, 0, 0), 2)

            # 鼻尖
            nose_point = landmarks[self.nose_tip]
            cv2.circle(image, (int(nose_point.x * img_w), int(nose_point.y * img_h)), 10, (0, 255, 255), -1)

            cv2.line(image, (int(face_center_x * img_w), int(center_y * img_h)), (int(nose_x * img_w), int(nose_point.y * img_h)), (255, 0, 255), 2)

            # 额头和下巴
            forehead = landmarks[self.forehead_top]
            chin = landmarks[self.chin_bottom]
            cv2.circle(image, (int(forehead.x * img_w), int(forehead.y * img_h)), 6, (255, 0, 255), -1)
            cv2.circle(image, (int(chin.x * img_w), int(chin.y * img_h)), 6, (255, 0, 255), -1)

        if direction and direction != "CENTER":
            direction_info = {"LEFT": ("Looking Left", (0, 255, 0), "LEFT SCREEN"),
                "RIGHT": ("Looking Right", (255, 0, 0), "RIGHT SCREEN"),
                "LEFT_UP": ("Left Up", (0, 255, 255), "LEFT-UP"),
                "RIGHT_UP": ("Right Up", (255, 0, 255), "RIGHT-UP"),
                "UP": ("Looking Up", (255, 255, 0), "UP"), }

            if direction in direction_info:
                english, color, subtitle = direction_info[direction]

                text_size = cv2.getTextSize(english, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)[0]
                text_x = (img_w - text_size[0]) // 2
                text_y = int(img_h * 0.7)

                padding = 15
                cv2.rectangle(image, (text_x - padding, text_y - text_size[1] - padding),
                            (text_x + text_size[0] + padding, text_y + padding), (0, 0, 0), -1)

                cv2.putText(image, english, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)

                subtitle_size = cv2.getTextSize(subtitle, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                subtitle_x = (img_w - subtitle_size[0]) // 2
                cv2.putText(image, subtitle, (subtitle_x, text_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            else:
                cv2.putText(image, f"Direction: {direction}", (img_w // 2 - 150, img_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

        bar_width = img_w - 60
        bar_height = 40
        bar_x = 30
        bar_y = img_h - 80

        cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (40, 40, 40), -1)
        center_x = bar_x + bar_width // 2
        cv2.line(image, (center_x, bar_y), (center_x, bar_y + bar_height), (255, 255, 255), 2)
        threshold_offset = self.h_offset_threshold
        left_threshold_x = int(bar_x + bar_width // 2 - threshold_offset * bar_width * 2)
        right_threshold_x = int(bar_x + bar_width // 2 + threshold_offset * bar_width * 2)
        cv2.line(image, (left_threshold_x, bar_y), (left_threshold_x, bar_y + bar_height), (0, 255, 0), 2)
        cv2.line(image, (right_threshold_x, bar_y), (right_threshold_x, bar_y + bar_height), (255, 0, 0), 2)
        display_offset = np.clip(h_offset, -0.2, 0.2)
        offset_x = int(bar_x + bar_width // 2 + display_offset * bar_width * 2)

        offset_color = (255, 255, 0)        
        if h_offset < -self.h_offset_threshold:
            offset_color = (0, 255, 0)
        elif h_offset > self.h_offset_threshold:
            offset_color = (255, 0, 0)
        else:
            offset_color = (255, 255, 0)

        cv2.circle(image, (offset_x, bar_y + bar_height // 2), 15, offset_color, -1)
        cv2.circle(image, (offset_x, bar_y + bar_height // 2), 15, (255, 255, 255), 2)

        cv2.putText(image, "LEFT", (bar_x, bar_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(image, "CENTER", (center_x - 35, bar_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(image, "RIGHT", (bar_x + bar_width - 60, bar_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        v_bar_width = 40
        v_bar_height = img_h - 200
        v_bar_x = img_w - 60
        v_bar_y = 80

        cv2.rectangle(image, (v_bar_x, v_bar_y), (v_bar_x + v_bar_width, v_bar_y + v_bar_height), (40, 40, 40), -1)

        threshold_ratio = 0.5
        threshold_y = int(v_bar_y + v_bar_height * (1 - (self.v_ratio_threshold - threshold_ratio) / 0.3))
        cv2.line(image, (v_bar_x, threshold_y), (v_bar_x + v_bar_width, threshold_y), (0, 255, 255), 2)

        display_v_ratio = np.clip(v_ratio, 0.5, 0.8)
        ratio_normalized = (display_v_ratio - threshold_ratio) / 0.3
        ratio_y = int(v_bar_y + v_bar_height * (1 - ratio_normalized))

        fill_height = v_bar_y + v_bar_height - ratio_y
        v_color = (0, 255, 0) if v_ratio > self.v_ratio_threshold else (100, 100, 100)
        cv2.rectangle(image, (v_bar_x + 5, ratio_y), (v_bar_x + v_bar_width - 5, v_bar_y + v_bar_height), v_color, -1)

        cv2.circle(image, (v_bar_x + v_bar_width // 2, ratio_y), 12, v_color, -1)
        cv2.circle(image, (v_bar_x + v_bar_width // 2, ratio_y), 12, (255, 255, 255), 2)

        cv2.putText(image, "UP", (v_bar_x - 5, v_bar_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(image, "DOWN", (v_bar_x - 20, v_bar_y + v_bar_height + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        return image

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("ERROR: Cannot open camera")
            return

        print("=" * 70)
        print("Geometric Face Pose Detection System (5 Directions)")
        print("=" * 70)
        print()
        print("Detection Method: Geometric Algorithm (Pure Math)")
        print("  H-Offset = nose_x - (left_x + right_x) / 2")
        print("  V-Ratio = face_height / face_width")
        print()
        print("5 Directions:")
        print("  1. LEFT_UP  - 左上 (左转 + 抬头)")
        print("  2. RIGHT_UP - 右上 (右转 + 抬头)")
        print("  3. UP       - 正上 (只抬头)")
        print("  4. LEFT     - 左转 (左侧屏幕)")
        print("  5. RIGHT    - 右转 (右侧屏幕)")
        print()
        print("Rules:")
        print("  H-Offset > threshold & V-Ratio > 0.65  → RIGHT_UP")
        print("  H-Offset < -threshold & V-Ratio > 0.65 → LEFT_UP")
        print("  |H-Offset| < threshold & V-Ratio > 0.65 → UP")
        print("  H-Offset > threshold & V-Ratio <= 0.65 → RIGHT")
        print("  H-Offset < -threshold & V-Ratio <= 0.65 → LEFT")
        print()
        print("Features:")
        print("  - 30 FPS real-time detection")
        print("  - Noise filtering with 5-frame smoothing")
        print("  - 3-frame stability check")
        print("  - Auto-moving window based on direction")
        print()
        print("Controls:")
        print("  Q     - Quit")
        print("  +/-   - Adjust horizontal sensitivity")
        print("  W/S   - Adjust vertical sensitivity")
        print("  R     - Reset thresholds")
        print("=" * 70)
        print()

        movable_window = MovableWindow()

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                continue

            image = cv2.flip(image, 1)

            h_offset, face_center_x, nose_x, v_ratio, landmarks = self.calculate_metrics(image)

            # 识别方向
            direction = self.recognize_direction(h_offset, v_ratio)

            if direction:
                movable_window.update_direction(direction)

            image = self.draw_visualization(image, h_offset, face_center_x, nose_x, v_ratio, direction, landmarks)

            cv2.imshow('Face Pose Detection', image)

            movable_window.update()

            # 按键处理
            key = cv2.waitKey(5) & 0xFF
            if key == ord('q') or key == ord('Q'):
                print("\nExiting...")
                break
            elif key == ord('+') or key == ord('='):
                self.h_offset_threshold = max(0.01, self.h_offset_threshold - 0.005)
                print(f"H-Sensitivity UP - Threshold: {self.h_offset_threshold:.3f}")
            elif key == ord('-') or key == ord('_'):
                self.h_offset_threshold = min(0.10, self.h_offset_threshold + 0.005)
                print(f"H-Sensitivity DOWN - Threshold: {self.h_offset_threshold:.3f}")
            elif key == ord('w') or key == ord('W'):
                self.v_ratio_threshold = max(0.3, self.v_ratio_threshold - 0.05)
                print(f"V-Sensitivity UP - Threshold: {self.v_ratio_threshold:.2f}")
            elif key == ord('s') or key == ord('S'):
                self.v_ratio_threshold = min(1.5, self.v_ratio_threshold + 0.05)
                print(f"V-Sensitivity DOWN - Threshold: {self.v_ratio_threshold:.2f}")
            elif key == ord('r') or key == ord('R'):
                self.h_offset_threshold = 0.03
                self.v_ratio_threshold = 0.65
                print(f"Threshold RESET: H={self.h_offset_threshold:.3f}, V={self.v_ratio_threshold:.2f}")

        cap.release()
        cv2.destroyAllWindows()
        self.detector.close()
        movable_window.destroy()
        print("System closed.")

def main():
    print("\nInitializing Face Pose Detection System...")
    detector = FacePoseDetector()
    detector.run()

if __name__ == "__main__":
    main()
