
# import cv2
# from src.detector.pose_detector import PoseDetector
# from src.config.settings import Settings
# from src.core.reminder.reminder import Reminder
# from src.core.ui.ui_handler import UIHandler
#
#
# def main():
#     """主函数"""
#     # 加载配置
#     settings = Settings()
#
#     # 初始化摄像头
#     cap = cv2.VideoCapture(settings.CAMERA_INDEX)
#
#     # 创建姿态检测器实例
#     detector = PoseDetector()
#
#     # 创建提醒器（包含持续时间监测）
#     reminder = Reminder(
#         enable_sound=settings.ENABLE_SOUND,
#         enable_notification=settings.ENABLE_NOTIFICATION,
#         data_dir=settings.DATA_DIR
#     )
#
#     # 创建UI处理器
#     ui_handler = UIHandler(
#         window_title=settings.WINDOW_TITLE,
#         window_width=settings.WINDOW_WIDTH,
#         window_height=settings.WINDOW_HEIGHT
#     )
#
#     # 创建窗口
#     ui_handler.create_window()
#
#     # 全屏模式
#     if settings.ENABLE_FULLSCREEN:
#         ui_handler.toggle_fullscreen()
#
#     try:
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 print("无法读取摄像头画面")
#                 break
#
#             # 处理每一帧
#             image, results = detector.process_frame(frame)
#
#             # 更新坐姿持续时间并检查是否需要提醒
#             duration = reminder.update_posture(results)
#             if results != "Good posture" and results != "No person detected":
#                 if duration > settings.POSTURE_THRESHOLD:
#                     reminder.remind(f"不良坐姿已持续 {int(duration)} 秒，请调整坐姿！", results)
#
#             # 绘制UI
#             image = ui_handler.draw_ui(image, results, duration)
#
#             # 显示结果
#             ui_handler.show_image(image)
#
#             # 处理按键
#             key = ui_handler.wait_key(1)
#             if key & 0xFF == ord('q'):
#                 break
#             elif key & 0xFF == ord('f'):
#                 ui_handler.toggle_fullscreen()
#
#     finally:
#         # 保存会话数据
#         reminder.save_session()
#
#         # 释放资源
#         cap.release()
#         detector.close()
#         ui_handler.destroy_window()
#         cv2.destroyAllWindows()
#
#
# if __name__ == "__main__":
#     main()
"""
主入口文件：整合所有功能模块
职责：协调 Detector, UI, Reminder, Tracker 之间的数据流动
"""

# app.py 完整版（适配新版UI）
import sys
import cv2
import time
import numpy as np
from PySide6.QtWidgets import QApplication, QMessageBox
from PySide6.QtCore import QTimer
from src.core.ui.ui_handler import UIHandler
from src.detector.pose_detector import PoseDetector
import mediapipe as mp
import math as m

class PostureApp:
    def __init__(self):
        # 运行状态
        self.is_running = False
        self.is_paused = False
        self.cap = None
        self.camera_index = 0

        # 统计数据
        self.bad_count = 0
        self.total_alerts = 0
        self.bad_duration = 0.0
        self.last_posture = "良好"

        # UI 与定时器
        self.ui = UIHandler("智能坐姿识别系统 V1.0")
        self.timer = QTimer()
        self.timer.timeout.connect(self.detection_loop)

        self.detector = PoseDetector()

        self.last_time = time.time()

        # 绑定按钮信号
        self._connect_signals()

    def _connect_signals(self):
        self.ui.signal_start_detection.connect(self.start_detection)
        self.ui.signal_pause_detection.connect(self.pause_detection)
        self.ui.signal_stop_detection.connect(self.stop_detection)
        self.ui.signal_query_report.connect(self.show_report)

    def start_detection(self):
        if self.is_running:
            return

        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            QMessageBox.critical(self.ui, "错误", "无法打开摄像头！")
            return

        self.is_running = True
        self.is_paused = False
        self.timer.start(30)  # ~33 FPS
        print("[INFO] 开始检测")

    def pause_detection(self):
        if not self.is_running:
            return
        self.is_paused = not self.is_paused
        print("[INFO]", "已暂停" if self.is_paused else "已恢复")

    def stop_detection(self):
        self.is_running = False
        self.is_paused = False
        self.timer.stop()

        if self.cap:
            self.cap.release()
            self.cap = None

        black = np.zeros((480, 640, 3), np.uint8)
        self.ui.update_display(black, {
            "posture": "未检测",
            "camera_status": "已停止"
        })
        print("[INFO] 停止检测")

    def detection_loop(self):
        if not self.is_running or self.is_paused or not self.cap:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.ui.update_display(None, {
                "camera_status": "异常",
                "frame_status": "未检测到人"
            })
            return

        # ===================== 坐姿检测（替换成你的算法）=====================
        processed_frame, eng_posture, is_person_detected, angles = self.detector.process_frame(frame)

        # 英文坐姿 → 中文显示
        posture_map = {
            "Good posture": "良好",
            "Looking down": "低头",
            "Looking up": "仰头",
            "Uneven shoulders": "高低肩",
            "Left tilt": "左歪头",
            "Right tilt": "右歪头",
            "Leaning on desk": "趴桌",
            "No person detected": "未检测到人"
        }
        display_posture = posture_map.get(eng_posture, "未开始检测")
        posture = eng_posture
        # ==================================================================

        # 不良坐姿计时统计
        if is_person_detected and posture not in ["良好", "未检测", "No person detected"]:
            self.bad_duration += 0.03
            if self.last_posture != eng_posture:
                self.bad_count += 1
                self.total_alerts += 1
        else:
            if not is_person_detected:
                self.bad_duration = 0
        self.last_posture = posture

        current_time = time.time()
        real_fps = int(1 / (current_time - self.last_time)) if (current_time - self.last_time) > 0 else 30
        self.last_time = current_time

        # 传给 UI 的数据
        ui_data = {
            "posture": display_posture,
            "duration": round(self.bad_duration, 1),
            "bad_count": self.bad_count,
            "total_alerts": self.total_alerts,
            "fps": real_fps,
            "camera_status": "已连接",
            "frame_status": "已检测到人" if is_person_detected else "未检测到人",
            "angles": angles,
            "suggestion": self._get_suggestion(eng_posture)
        }

        self.ui.update_display(processed_frame, ui_data)

    def _get_suggestion(self, posture):
        # 完全匹配你UI界面上的坐姿建议文字
        tips = {
            "Looking down": "请抬头，保持颈部挺直",
            "Looking up": "头部保持中正，不要过度仰头",
            "Uneven shoulders": "双肩放平，放松颈部",
            "Left tilt": "头部回正，不要向左倾斜",
            "Right tilt": "头部回正，不要向右倾斜",
            "Leaning on desk": "坐直，不要趴桌",
            "Good posture": "保持正确坐姿！",
            "未检测": "请开始检测",
            "No person detected": "未检测到人，请调整位置"
        }
        return tips.get(posture, "请保持正确坐姿")

    def show_report(self):
        QMessageBox.information(self.ui, "报告查询",
            f"今日坐姿统计\n\n"
            f"不良次数：{self.bad_count}\n"
            f"累计提醒：{self.total_alerts}")

    def run(self):
        self.ui.show()
        black = np.zeros((480, 640, 3), np.uint8)
        self.ui.update_display(black)

def main():
    app = QApplication(sys.argv)
    window = PostureApp()
    window.run()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
