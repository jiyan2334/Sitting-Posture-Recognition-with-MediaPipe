from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QGroupBox, QFormLayout,
    QCheckBox, QDoubleSpinBox, QDialog, QMessageBox
)
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt, Signal
import cv2
import numpy as np
import sys
from typing import Dict, Any, Optional

class UIHandler(QMainWindow):
    """
    智能坐姿识别系统 V1.0 - 修复版GUI
    完全匹配文档：蓝白柔和+卡片式+全圆角+响应式布局
    支持：骨骼点高亮、坐姿状态实时显示、全屏切换、设置弹窗
    """
    # 核心交互信号
    signal_start_detection = Signal()
    signal_pause_detection = Signal()
    signal_stop_detection = Signal()
    signal_query_report = Signal()
    signal_open_settings = Signal()

    def __init__(self, window_title="智能坐姿识别系统 V1.0",
                 window_width=1200, window_height=800):
        super().__init__()
        self.window_title = window_title
        self.window_width = window_width
        self.window_height = window_height
        self.is_fullscreen = False
        self.current_pixmap = None

        # 系统状态数据
        self.system_data = {
            "posture": "未检测",
            "duration": 0,
            "bad_count": 0,
            "total_alerts": 0,
            "fps": 0,
            "camera_status": "未连接",
            "frame_status": "未检测到人",
            "angles": {"neck": 0, "shoulder": 0, "waist": 0},
            "suggestion": "请开始检测"
        }

        self.init_ui()
        self.apply_modern_style()

    def init_ui(self):
        """初始化完整GUI布局（严格匹配文档）"""
        self.setWindowTitle(self.window_title)
        self.resize(self.window_width, self.window_height)

        # 主容器
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)

        # ====================== 左侧功能面板（1/4 宽度）======================
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(10, 10, 10, 10)
        left_layout.setSpacing(12)

        # 1. 快捷控制区（开始/暂停/停止）
        self._create_control_section(left_layout)
        # 2. 功能操作区（系统设置/报告查询）
        self._create_action_section(left_layout)
        # 3. 状态卡片（当前坐姿/时长/不良次数/提醒次数）
        self._create_status_section(left_layout)

        left_layout.addStretch()

        # ====================== 右侧主界面（3/4宽度）======================
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(10, 10, 10, 10)
        right_layout.setSpacing(12)

        # 1. 实时视频画面区（最大区域）
        self._create_video_section(right_layout)
        # 2. 坐姿分析信息卡片
        self._create_info_section(right_layout)
        # 3. 底部状态栏（FPS/摄像头/画面状态）
        self._create_bottom_status(right_layout)

        # ====================== 组装主布局 ======================
        main_layout.addWidget(left_panel, 1)
        main_layout.addWidget(right_panel, 3)

    def _create_control_section(self, parent_layout):
        """快捷控制区"""
        group = QGroupBox("快捷控制")
        layout = QVBoxLayout(group)
        layout.setSpacing(8)

        self.btn_start = QPushButton("▶ 开始检测")
        self.btn_start.setObjectName("start_btn")
        self.btn_start.clicked.connect(self.signal_start_detection)

        self.btn_pause = QPushButton("⏸ 暂停检测")
        self.btn_pause.setObjectName("pause_btn")
        self.btn_pause.clicked.connect(self.signal_pause_detection)

        self.btn_stop = QPushButton("🛑 停止检测")
        self.btn_stop.setObjectName("stop_btn")
        self.btn_stop.clicked.connect(self.signal_stop_detection)

        layout.addWidget(self.btn_start)
        layout.addWidget(self.btn_pause)
        layout.addWidget(self.btn_stop)
        parent_layout.addWidget(group)

    def _create_action_section(self, parent_layout):
        """功能操作区"""
        group = QGroupBox("功能操作")
        layout = QVBoxLayout(group)
        layout.setSpacing(8)

        self.btn_settings = QPushButton("⚙ 系统设置")
        self.btn_settings.setObjectName("settings_btn")
        self.btn_settings.clicked.connect(self.open_settings_dialog)

        self.btn_report = QPushButton("📊 报告查询")
        self.btn_report.setObjectName("report_btn")
        self.btn_report.clicked.connect(self.signal_query_report)

        layout.addWidget(self.btn_settings)
        layout.addWidget(self.btn_report)
        parent_layout.addWidget(group)

    def _create_status_section(self, parent_layout):
        """状态信息卡片"""
        group = QGroupBox("当前状态")
        layout = QFormLayout(group)
        layout.setSpacing(8)
        layout.setLabelAlignment(Qt.AlignRight)

        self.lbl_posture = QLabel("未检测")
        self.lbl_duration = QLabel("0 秒")
        self.lbl_bad_count = QLabel("0 次")
        self.lbl_total_alerts = QLabel("0 次")

        layout.addRow("当前坐姿:", self.lbl_posture)
        layout.addRow("不良持续:", self.lbl_duration)
        layout.addRow("今日不良:", self.lbl_bad_count)
        layout.addRow("累计提醒:", self.lbl_total_alerts)
        parent_layout.addWidget(group)

    def _create_video_section(self, parent_layout):
        """实时视频画面区"""
        group = QGroupBox("实时监控画面")
        layout = QVBoxLayout(group)

        self.video_label = DraggableLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(600, 400)
        self.video_label.setStyleSheet("""
            QLabel {
                background-color: #1a1a1a;
                border-radius: 10px;
                color: #ffffff;
                font-size: 14px;
            }
        """)
        self.video_label.setText("等待摄像头启动...\n双击画面切换全屏")
        self.video_label.mouseDoubleClickEvent = self.toggle_fullscreen

        layout.addWidget(self.video_label)
        parent_layout.addWidget(group, 3)

    def _create_info_section(self, parent_layout):
        """坐姿分析详情卡片"""
        group = QGroupBox("坐姿分析")
        layout = QHBoxLayout(group)
        layout.setSpacing(15)

        # 左侧：检测结果 + 关键角度
        left_info = QWidget()
        info_layout = QFormLayout(left_info)
        info_layout.setLabelAlignment(Qt.AlignRight)

        self.lbl_result = QLabel("等待检测")
        self.lbl_shoulder_angle = QLabel("水平")

        info_layout.addRow("坐姿结果:", self.lbl_result)
        info_layout.addRow("肩部角度:", self.lbl_shoulder_angle)

        # 右侧：调整建议
        right_suggest = QGroupBox("调整建议")
        suggest_layout = QVBoxLayout(right_suggest)
        self.lbl_suggestion = QLabel("请开始检测")
        self.lbl_suggestion.setWordWrap(True)
        suggest_layout.addWidget(self.lbl_suggestion)

        layout.addWidget(left_info, 1)
        layout.addWidget(right_suggest, 1)
        parent_layout.addWidget(group, 1)

    def _create_bottom_status(self, parent_layout):
        """底部状态栏"""
        self.bottom_status_label = QLabel()
        self.bottom_status_label.setAlignment(Qt.AlignLeft)
        self.bottom_status_label.setStyleSheet("""
            QLabel {
                color: #666666;
                font-size: 12px;
                padding: 5px;
                background: transparent;
                qproperty-wordWrap: false; /* 禁止自动换行 */
            }
        """)
        parent_layout.addWidget(self.bottom_status_label)

    def apply_modern_style(self):
        """应用现代化蓝白柔和UI样式（全圆角+卡片式）"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f7fa;
            }
            QGroupBox {
                background-color: #ffffff;
                border: 1px solid #e5e7eb;
                border-radius: 12px;
                margin-top: 20px;
                font-size: 15px;
                font-weight: 600;
                color: #333333;
                padding: 16px 12px 12px 12px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 16px;
                top: 0px;
                padding: 0 8px;
                background-color: #ffffff;
            }
            QPushButton {
                font-family: "Microsoft YaHei";
                font-size: 14px;
                border-radius: 8px;
                padding: 10px 12px;
                min-height: 38px;
            }
            QPushButton#start_btn {
                background-color: #22c55e;
                color: white;
                border: none;
            }
            QPushButton#start_btn:hover {
                background-color: #16a34a;
            }
            QPushButton#pause_btn {
                background-color: #f59e0b;
                color: white;
                border: none;
            }
            QPushButton#pause_btn:hover {
                background-color: #d97706;
            }
            QPushButton#stop_btn {
                background-color: #ef4444;
                color: white;
                border: none;
            }
            QPushButton#stop_btn:hover {
                background-color: #dc2626;
            }
            QPushButton#settings_btn {
                background-color: #3b82f6;
                color: white;
                border: none;
            }
            QPushButton#settings_btn:hover {
                background-color: #2563eb;
            }
            QPushButton#report_btn {
                background-color: #8b5cf6;
                color: white;
                border: none;
            }
            QPushButton#report_btn:hover {
                background-color: #7c3aed;
            }
            QLabel {
                font-family: "Microsoft YaHei";
                color: #333333;
                font-size: 14px;
            }
            QFormLayout QLabel {
                font-size: 13px;
                color: #64748b;
            }
            /* 修复设置窗口文字显示 */
            QDialog QGroupBox {
                font-size: 14px;
                padding: 15px;
            }
            QDialog QLabel {
                font-size: 13px;
                min-height: 20px;
            }
            QDialog QCheckBox {
                font-size: 13px;
                min-height: 20px;
            }
            QDialog QDoubleSpinBox {
                font-size: 13px;
                min-height: 28px;
                min-width: 80px;
            }
        """)

    def update_display(self, frame: Optional[np.ndarray] = None,
                       data: Optional[Dict[str, Any]] = None):
        """统一更新画面与数据（核心渲染方法）"""
        if data:
            self.system_data.update(data)
            self._refresh_all_labels()

        if frame is not None:
            try:
                h, w, ch = frame.shape
                bytes_per_line = ch * w
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                qt_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                self.current_pixmap = QPixmap.fromImage(qt_img).scaled(
                    self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
                self.video_label.setPixmap(self.current_pixmap)
            except Exception:
                self.video_label.setText("画面渲染中...")

    def _refresh_all_labels(self):
        """刷新所有状态文本"""
        d = self.system_data
        # 左侧状态
        self.lbl_posture.setText(d.get("posture", "未知"))
        self.lbl_duration.setText(f"{d.get('duration', 0)} 秒")
        self.lbl_bad_count.setText(f"{d.get('bad_count', 0)} 次")
        self.lbl_total_alerts.setText(f"{d.get('total_alerts', 0)} 次")

        # 右侧信息
        self.lbl_result.setText(d.get("posture", "未知"))
        # 肩部角度
        shoulder_val = d.get('angles', {}).get('shoulder', 0)
        if abs(shoulder_val) < 3:  # 阈值设为3度
            shoulder_text = "水平"
        elif shoulder_val > 0:
            shoulder_text = f"左高右低 ({shoulder_val:.1f}°)"
        else:
            shoulder_text = f"右高左低 ({abs(shoulder_val):.1f}°)"
        self.lbl_shoulder_angle.setText(shoulder_text)

        self.lbl_suggestion.setText(d.get("suggestion", "无"))

        # 底部状态栏
        self.bottom_status_label.setText(
            f"FPS: {d.get('fps', 0)}  |  "
            f"摄像头: {d.get('camera_status', '未知')}  |"
            f"  画面: {d.get('frame_status', '未知')}  |"
            f"  画面偏转角: {d.get('angles', {}).get('frame_angle', 0):.1f}°"
        )

    def toggle_fullscreen(self, event=None):
        """双击全屏/退出全屏"""
        if self.is_fullscreen:
            self.showNormal()
            self.is_fullscreen = False
        else:
            self.showFullScreen()
            self.is_fullscreen = True

    def open_settings_dialog(self):
        """打开设置窗口（匹配文档：声音+可视化+参数）"""
        dialog = SettingsDialog(self)
        dialog.exec()

    def closeEvent(self, event):
        reply = QMessageBox.question(
            self, "退出确认", "确定要退出系统吗？",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.signal_stop_detection.emit()
            event.accept()
        else:
            event.ignore()

class DraggableLabel(QLabel):
    """支持双击全屏的视频显示标签"""
    def mouseDoubleClickEvent(self, event):
        if hasattr(self.window(), "toggle_fullscreen"):
            self.window().toggle_fullscreen()
        super().mouseDoubleClickEvent(event)

class SettingsDialog(QDialog):
    """系统设置弹窗（完全修复文字显示问题）"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("系统设置")
        # 增大窗口尺寸，彻底解决文字挤压
        self.setFixedSize(480, 520)
        self.setWindowModality(Qt.ApplicationModal)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        # 增大边距，避免文字贴边
        layout.setContentsMargins(25, 25, 25, 25)
        layout.setSpacing(20)

        # 1. 提醒设置
        alert_group = QGroupBox("🔔 提醒设置")
        alert_layout = QFormLayout(alert_group)
        # 增大行间距，避免文字重叠
        alert_layout.setSpacing(12)
        alert_layout.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        self.chk_sound = QCheckBox("启用声音提醒")
        self.chk_sound.setChecked(True)
        self.chk_sound.setMinimumHeight(25)

        self.spin_volume = QDoubleSpinBox()
        self.spin_volume.setRange(0, 1)
        self.spin_volume.setSingleStep(0.1)
        self.spin_volume.setValue(0.8)
        self.spin_volume.setMinimumHeight(30)
        self.spin_volume.setMinimumWidth(100)

        alert_layout.addRow(self.chk_sound)
        alert_layout.addRow("提示音量:", self.spin_volume)

        # 2. 可视化设置
        viz_group = QGroupBox("🎨 可视化设置")
        viz_layout = QFormLayout(viz_group)
        viz_layout.setSpacing(12)
        viz_layout.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        self.chk_landmarks = QCheckBox("显示骨骼关键点")
        self.chk_landmarks.setChecked(True)
        self.chk_landmarks.setMinimumHeight(25)

        self.chk_lines = QCheckBox("显示骨骼连线")
        self.chk_lines.setChecked(True)
        self.chk_lines.setMinimumHeight(25)
        self.chk_landmarks.toggled.connect(self.chk_lines.setEnabled)

        viz_layout.addRow(self.chk_landmarks)
        viz_layout.addRow(self.chk_lines)

        # 3. 检测参数
        param_group = QGroupBox("⚙️ 检测参数")
        param_layout = QFormLayout(param_group)
        param_layout.setSpacing(12)
        param_layout.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        self.spin_conf = QDoubleSpinBox()
        self.spin_conf.setRange(0.1, 1.0)
        self.spin_conf.setValue(0.5)
        self.spin_conf.setSingleStep(0.05)
        self.spin_conf.setMinimumHeight(30)
        self.spin_conf.setMinimumWidth(100)

        param_layout.addRow("检测置信度:", self.spin_conf)

        # 按钮
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(15)
        btn_save = QPushButton("保存设置")
        btn_cancel = QPushButton("取消")
        btn_save.setMinimumHeight(38)
        btn_cancel.setMinimumHeight(38)
        btn_save.clicked.connect(self.accept)
        btn_cancel.clicked.connect(self.reject)
        btn_layout.addStretch()
        btn_layout.addWidget(btn_save)
        btn_layout.addWidget(btn_cancel)

        layout.addWidget(alert_group)
        layout.addWidget(viz_group)
        layout.addWidget(param_group)
        layout.addStretch()
        layout.addLayout(btn_layout)

# 测试入口
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = UIHandler()

    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(test_frame, "Camera Feed", (180, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    window.update_display(test_frame)
    window.show()
    sys.exit(app.exec())
