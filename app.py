import sys
import cv2
import time
import numpy as np
from PySide6.QtWidgets import QApplication, QMessageBox
from PySide6.QtCore import QTimer
from src.core.ui.ui_handler import UIHandler
from src.detector.pose_detector import PoseDetector

from src.config.settings import settings  # 新增导入 --cwy
from src.core.reminder.reminder import MultiModalReminder  # 确保导入提醒类  --zyx

# --wsy sy--
from src.core.tracking.tracking import Tracking
# --wsy sy--

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
        self.ui = UIHandler("智能坐姿识别系统")
        self.timer = QTimer()
        self.timer.timeout.connect(self.detection_loop)

        # 连接设置信号 --cwy\zyx
        self.ui.signal_apply_settings.connect(self._apply_settings)

        self.detector = PoseDetector() # 检测器配置（骨骼显示）
        # --wsy sy--
        self.tracking = Tracking(data_dir="data")
        # --wsy sy--

        # 初始化提醒器（使用配置参数） --cwy\zyx
        self.reminder = MultiModalReminder(
            enable_sound=settings.enable_sound,
            volume=settings.alert_volume,
            threshold=settings.posture_threshold
        )
        self.last_time = time.time()
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
        # --wsy sy--
        if self.is_paused:
            # 调用tracking的pause方法
            self.tracking.pause()
            print("[INFO] 已暂停")
        else:
            # 调用tracking的resume方法
            self.tracking.resume()
            print("[INFO] 已恢复")
        # --wsy sy--

    def stop_detection(self):
        # --wsy sy--
        if not self.is_running:
            return
        # --wsy sy--
        self.is_running = False
        self.is_paused = False
        self.timer.stop()

        if self.cap:
            self.cap.release()
            self.cap = None

        # --wsy sy--
        # 保存会话数据
        filename = self.tracking.save_session()
        print(f"[INFO] 会话数据已保存: {filename}")
        # --wsy sy--

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

        # --wsy sy--
        # 使用tracking更新坐姿数据
        self.tracking.update_posture(posture)
        # --wsy sy--

        # 不良坐姿计时统计
        # 逻辑修改：不再在这里判断阈值，而是把状态交给 Reminder 类去判断

        # 判断是否为不良姿势：检测到人 且 不是 "Good posture"
        if is_person_detected and posture != "Good posture":
            self.bad_duration += 0.03

            # 统计次数（仅在姿势切换时）
            if self.last_posture != eng_posture:
                self.bad_count += 1

            # 调用提醒器：传入 True，表示“我现在是不良姿势，请帮我计时/提醒”
            self.reminder.remind(True, eng_posture)

        else:
            # 姿势良好或无人时
            if not is_person_detected:
                self.bad_duration = 0

            # 调用提醒器：传入 False，表示“我现在姿势良好，请重置计时器”
            self.reminder.remind(False, eng_posture)
        self.last_posture = posture
        current_time = time.time()
        real_fps = int(1 / (current_time - self.last_time)) if (current_time - self.last_time) > 0 else 30
        self.last_time = current_time

        # 传给 UI 的数据
        ui_data = {
            "posture": display_posture,
            "duration": round(self.bad_duration, 1),
            "bad_count": self.bad_count,
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
        # --wsy sy--
        from src.core.ui.ui_handler import ReportDialog
        dialog = ReportDialog(self.ui, data_dir="data")
        dialog.exec()
        # --wsy sy--

    def run(self):
        self.ui.show()
        black = np.zeros((480, 640, 3), np.uint8)
        self.ui.update_display(black)

    # --- 新增/修改：应用设置的槽函数 cwy\zyx
    def _apply_settings(self, settings_dict):
        """
        接收来自UI的设置字典，并更新内部对象
        """
        print(f"[App] 正在应用新设置: {settings_dict}")

        # 1. 更新 Reminder 模块
        if hasattr(self, 'reminder'):
            self.reminder.update_settings(  # Reminder 类需要有这个方法
                enable_sound=settings_dict.get('enable_sound'),
                volume=settings_dict.get('alert_volume'),
                threshold=settings_dict.get('posture_threshold')  # 更新提醒阈值
            )

        # 2. 更新 Detector 模块（可视化设置）
        if hasattr(self, 'detector'):
            self.detector.update_display_settings(  # Detector 类需要预留此方法
                show_landmarks=settings_dict.get('show_landmarks'),
                show_lines=settings_dict.get('show_lines')
            )

        # 3. 更新状态栏提示
        self.ui.statusBar().showMessage("设置已应用", 2000)

def main():
    app = QApplication(sys.argv)
    window = PostureApp()
    window.run()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()