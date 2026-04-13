from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QGroupBox, QFormLayout,
    QCheckBox, QDoubleSpinBox, QDialog, QMessageBox, QLineEdit, QComboBox, QFileDialog, QListWidget, QScrollArea
)
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt, Signal
import cv2
import numpy as np
import sys
# --wsy sy--
import os
import json
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
# --wsy sy--
from typing import Dict, Any, Optional
# --sy--
from datetime import date
import calendar
# --sy--
# --wsy sy--
# Matplotlib 导入
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
# --wsy sy--
from src.config.settings import settings    #cwy、zyx
from PySide6.QtCore import Signal # 确保导入的是 PySide6  --cwy,zyx

class UIHandler(QMainWindow):
    """
    智能坐姿识别系统 V1.0 - 修复版GUI
    完全匹配文档：蓝白柔和+卡片式+全圆角+响应式布局
    支持：骨骼点高亮、坐姿状态实时显示、全屏切换、设置弹窗
    """
    # 核心交互信号
    # 把 signal_apply_settings 加在这里 (核心修复点) ---
    # 这样它就变成了对象的属性，app.py 才能通过 self.ui 访问到 --cwy,zyx
    signal_apply_settings = Signal(dict)
    signal_start_detection = Signal()
    signal_pause_detection = Signal()
    signal_stop_detection = Signal()
    signal_query_report = Signal()
    signal_open_settings = Signal()

    def __init__(self, window_title="智能坐姿识别系统 V1.0",
                 window_width=1200, window_height=700):
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
        # 3. 状态卡片（当前坐姿/时长/不良次数）
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
        # --- 新增这一行：点击时触发文字切换函数 ---cwy
        self.btn_pause.clicked.connect(self._switch_pause_text)

        self.btn_stop = QPushButton("🛑 停止检测")
        self.btn_stop.setObjectName("stop_btn")
        self.btn_stop.clicked.connect(self.signal_stop_detection)

        layout.addWidget(self.btn_start)
        layout.addWidget(self.btn_pause)
        layout.addWidget(self.btn_stop)
        parent_layout.addWidget(group)

    # 新增切换文字功能
    def _switch_pause_text(self):
        current_text = self.btn_pause.text()
        if "暂停" in current_text:
            self.btn_pause.setText("⏯ 继续检测")
        else:
            self.btn_pause.setText("⏸ 暂停检测")

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

        layout.addRow("当前坐姿:", self.lbl_posture)
        layout.addRow("不良持续:", self.lbl_duration)
        layout.addRow("今日不良:", self.lbl_bad_count)
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
                border: 1px solid #d0d8e6;
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
                color: #777777;
                font-size: 15px;
                font-weight: bold;
                padding: 8px;
                background: transparent;
                qproperty-wordWrap: false; /* 禁止自动换行 */
            }
        """)
        parent_layout.addWidget(self.bottom_status_label)
    # 字体大小调整cwy
    def apply_modern_style(self):
        """应用现代化蓝白柔和UI样式（全圆角+卡片式）"""
        self.setStyleSheet("""
            QMainWindow {
            background: qlineargradient(
                x1: 0, y1: 0, x2: 1, y2: 1,
                stop: 0 #fff9e6, stop: 0.25 #d9f9e6,
                stop: 0.5 #d9f2ff, stop: 0.75 #f2e6ff,
                stop: 1 #ffe6f2
            );
        }

        /* 统一卡片样式 + 轻微阴影 */
        QGroupBox {
            background-color: rgba(255, 255, 255, 0.95);
            border: 1px solid #d0d8e6;
            border-radius: 12px;
            margin-top: 16px;
            font-size: 15px;
            font-weight: bold;
            color: #333333;
            padding: 16px 12px 12px 12px;
            box-shadow: 0px 2px 8px rgba(0, 0, 0, 0.08);
        }

        /* 统一标题框样式（淡蓝圆角） */
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            left: 12px;
            top: 0px;
            padding: 4px 12px;
            background-color: #e6f7ff;
            border: 1px solid #91d5ff;
            border-radius: 10px;
            font-size: 15px;
            font-weight: bold;
            color: #1890ff;
        }

        /* 按钮通用样式 + 过渡动画 */
        QPushButton {
            font-family: "Microsoft YaHei", "微软雅黑", sans-serif;
            font-size: 16px;
            font-weight: bold;
            border-radius: 10px;
            padding: 12px 16px;
            min-height: 50px;
            border: none;
            color: white;
            transition: all 0.2s ease;
        }

        /* 开始按钮 */
        QPushButton#start_btn {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #b4f988, stop:1 #00c853);
        }
        QPushButton#start_btn:hover {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #99e670, stop:1 #00a843);
            transform: scale(1.02);
            box-shadow: 0px 3px 6px rgba(0, 0, 0, 0.15);
        }

        /* 暂停/继续 */
        QPushButton#pause_btn {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #ffcc4d, stop:1 #ff6d00);
        }
        QPushButton#pause_btn:hover {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #ffb733, stop:1 #ff5700);
            transform: scale(1.02);
            box-shadow: 0px 3px 6px rgba(0, 0, 0, 0.15);
        }

        /* 停止 */
        QPushButton#stop_btn {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #ffab91, stop:1 #f50057);
        }
        QPushButton#stop_btn:hover {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #ff8a80, stop:1 #d5004f);
            transform: scale(1.02);
            box-shadow: 0px 3px 6px rgba(0, 0, 0, 0.15);
        }

        /* 设置 */
        QPushButton#settings_btn {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #82b1ff, stop:1 #2979ff);
        }
        QPushButton#settings_btn:hover {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #699eff, stop:1 #1565c0);
            transform: scale(1.02);
            box-shadow: 0px 3px 6px rgba(0, 0, 0, 0.15);
        }

        /* 报告 */
        QPushButton#report_btn {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #ea80fc, stop:1 #9c27b0);
        }
        QPushButton#report_btn:hover {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #d81b60, stop:1 #7b1fa2);
            transform: scale(1.02);
            box-shadow: 0px 3px 6px rgba(0, 0, 0, 0.15);
        }

        /* 统一文字样式 */
        QLabel {
            font-family: "Microsoft YaHei", "微软雅黑", sans-serif;
            color: #333333;
            font-size: 16px;
            line-height: 1.5;
        }

        /* 调整建议框样式 */
        #suggestion_frame {
            background-color: #f0f9ff;
            border: 1px solid #91d5ff;
            border-radius: 12px;
            padding: 8px 12px;
        }
        #suggestion_title {
            background-color: #e6f7ff;
            border: 1px solid #91d5ff;
            border-radius: 8px;
            padding: 2px 10px;
            font-size: 14px;
            font-weight: bold;
            color: #1890ff;
        }

        /* 监控区域样式 */
        #camera_label {
            background-color: #000000;
            border: 1px solid #d0d8e6;
            border-radius: 8px;
        }

        /* 状态文字排版优化 */
        #status_label, #analysis_label {
            padding: 8px 12px;
            line-height: 1.8;
        }

        /* 弹窗样式不变 */
        QDialog {
            background-color: #ffffff;
        }
        QDialog QGroupBox {
            font-size: 14px;
            padding: 15px;
            color: #000000;
            background-color: #ffffff;
        }
        QDialog QLabel {
            font-size: 13px;
            min-height: 20px;
            color: #000000;
        }
        QDialog QCheckBox {
            font-size: 13px;
            min-height: 20px;
            color: #000000;
        }
        QDialog QDoubleSpinBox, QDialog QSpinBox {
            font-size: 13px;
            min-height: 28px;
            min-width: 80px;
            color: #000000;
            background-color: #ffffff;
            border: 1px solid #cccccc;
            border-radius: 4px;
        }
        QDialog QComboBox {
            color: #000000;
            background-color: #ffffff;
            border: 1px solid #cccccc;
            border-radius: 4px;
        }
        QDialog QPushButton {
            color: #ffffff;
            background-color: #2979ff;
            border: none;
            border-radius: 6px;
            padding: 4px 10px;
            min-height: 28px;
            min-width: 90px;
            font-size: 14px;
        }

        QStatusBar {
            font-family: "Microsoft YaHei", "微软雅黑", sans-serif;
            color: #666666;
            font-size: 15px;
            font-weight: bold;
            padding: 8px 12px;
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

        # --- 2. 右侧分析区（核心：动态颜色）---cwy
        current_posture = d.get("posture", "未知")
        # 【核心逻辑】判断坐姿并设置颜色
        if current_posture == "良好":
            color_style = "color: green; font-size: 22px; font-weight: bold;"  # 良好：绿色，超大字号
        else:
            color_style = "color: red; font-size: 22px; font-weight: bold;"  # 错误：红色，超大字号
        # 应用样式到显示结果的标签
        self.lbl_result.setText(current_posture)
        self.lbl_result.setStyleSheet(color_style)

        # 右侧信息
        self.lbl_result.setText(d.get("posture", "未知"))
        # 肩部角度 ---cwy
        shoulder_val = d.get('angles', {}).get('shoulder', 0)
        angle = d.get('angles', {}).get('frame_angle', 0)
        if abs(angle) < 30:
            if abs(shoulder_val) >= 85:  # 阈值设为5度
                shoulder_text = "水平"
            elif shoulder_val > 0:
                shoulder_text = f"左高右低 ({-shoulder_val + 90:.1f}°)"
            else:
                shoulder_text = f"右高左低 ({shoulder_val + 90:.1f}°)"
        if abs(angle) >= 30:
            if abs(shoulder_val) >= 80:  # 阈值设为10度
                shoulder_text = "水平"
            elif shoulder_val > 0:
                shoulder_text = f"左高右低 ({-shoulder_val + 90:.1f}°)"
            else:
                shoulder_text = f"右高左低 ({shoulder_val + 90:.1f}°)"

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
    # 全改closeEvent -cwy
    def closeEvent(self, event):
        # 美化退出确认框：中文按钮 + 柔和样式
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("退出确认")
        msg_box.setText("确定要退出智能坐姿识别系统吗？")
        msg_box.setIcon(QMessageBox.Question)

        # 添加中文按钮
        btn_confirm = msg_box.addButton("确定", QMessageBox.AcceptRole)
        btn_cancel = msg_box.addButton("取消", QMessageBox.RejectRole)

        # 统一风格（和主界面蓝白柔和风匹配）
        msg_box.setStyleSheet("""
            QMessageBox {
                background-color: #ffffff;
            }
            /* 专门控制图标占位符：强制留出 60px 宽度 */
            QMessageBox .QLabel {
                min-height: 60px;
                vertical-align: middle;
            }
            /* 精准控制图标本身的样式 */
            QMessageBox QIcon {
                width: 48px;
                height: 48px;
            }
            QMessageBox QLabel#qt_msgbox_label {
                color: #333333;
                font-size: 16px;
                padding-left: 15px; /* 给文字和图标留点间距 */
            }
            QPushButton {
                font-size: 15px;
                padding: 6px 16px;
                border-radius: 6px;
                min-width: 80px;
            }
            QPushButton:first-child {
                background-color: #ef4444;
                color: white;
                border: none;
            }
            QPushButton:first-child:hover {
                background-color: #dc2626;
            }
            QPushButton:last-child {
                background-color: #e5e7eb;
                color: #333333;
                border: none;
            }
            QPushButton:last-child:hover {
                background-color: #d1d5db;
            }
        """)

        msg_box.exec()

        if msg_box.clickedButton() == btn_confirm:
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

# 新增cwy\zyx
class SettingsDialog(QDialog):
    """系统设置弹窗（完全修复文字显示问题）"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("系统设置")
        # 增大窗口尺寸，彻底解决文字挤压
        self.setFixedSize(480, 580)
        self.setWindowModality(Qt.ApplicationModal)

        self._init_ui()
        # 加载当前配置到控件 --cwy、zyx
        self._load_settings()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        # 增大边距，避免文字贴边
        layout.setContentsMargins(20, 20, 20, 20)   # cwy\zyx
        layout.setSpacing(20)

        # 1. 提醒设置
        alert_group = QGroupBox("🔔 提醒设置")
        alert_layout = QFormLayout(alert_group)
        # 新增cwy\zyx 两条
        alert_layout.setLabelAlignment(Qt.AlignLeft)
        alert_layout.setFormAlignment(Qt.AlignLeft)
        # 增大行间距，避免文字重叠
        alert_layout.setSpacing(15)
        # alert_layout.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter) -cwy\zyx注释

        self.chk_sound = QCheckBox("启用声音提醒")
        # self.chk_sound.setChecked(True)  --注释cwy、zyx
        alert_layout.addRow(self.chk_sound)  # --cwy\zyx

        #  cwy\zyx
        #  提示音量 (下拉框)
        self.combo_volume = QComboBox()
        # 添加 0% 到 100% 的选项
        self.combo_volume.addItems(["0.1", "0.3", "0.5", "0.7", "0.8", "1.0"])
        self.combo_volume.setToolTip("选择提醒音量大小")
        self.combo_volume.setMinimumHeight(30)
        alert_layout.addRow("提示音量:", self.combo_volume)

        #  报警阈值 (下拉框)
        self.combo_threshold = QComboBox()
        # 添加 3秒 到 10秒 的选项
        self.combo_threshold.addItems(["3", "5", "8", "10", "15"])
        self.combo_threshold.setToolTip("持续不良姿势多少秒后报警")
        self.combo_threshold.setMinimumHeight(30)
        alert_layout.addRow("提示阈值(秒):", self.combo_threshold)

        # 2. 可视化设置
        viz_group = QGroupBox("🎨 可视化设置")
        viz_layout = QFormLayout(viz_group)
        # viz_layout.setSpacing(12) --注释cwy、zyx
        # viz_layout.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter) --注释cwy\zyx

        self.chk_landmarks = QCheckBox("显示骨骼关键点")
        self.chk_landmarks.setChecked(True)
        # self.chk_landmarks.setMinimumHeight(25)   --注释cwy、zyx

        self.chk_lines = QCheckBox("显示骨骼连线")
        self.chk_lines.setChecked(True)
        # self.chk_lines.setMinimumHeight(25)   --注释cwy、zyx
        self.chk_landmarks.toggled.connect(self.chk_lines.setEnabled)

        # 2. 新增逻辑：如果关键点取消，强制取消连线（解决：关键点取消 -> 连线也取消勾选）--cwy、zyx
        def sync_lines_state(checked):
            if not checked:
                self.chk_lines.setChecked(False)

        self.chk_landmarks.toggled.connect(sync_lines_state)

        viz_layout.addRow(self.chk_landmarks)
        viz_layout.addRow(self.chk_lines)

        # 3. 检测参数   此处删除 -cwy\zyx

        # 按钮
        btn_layout = QHBoxLayout()
        # btn_layout.setSpacing(15) --注释cwy、zyx
        btn_save = QPushButton("保存设置")
        btn_cancel = QPushButton("取消")
        #修改 cwy\zyx
        btn_save.setFixedSize(90,28)
        btn_cancel.setFixedSize(90,28)
        btn_save.clicked.connect(self.accept)
        btn_cancel.clicked.connect(self.reject)
        btn_layout.addStretch()
        btn_layout.addWidget(btn_save)
        btn_layout.addWidget(btn_cancel)

        layout.addWidget(alert_group)
        layout.addWidget(viz_group) #此处下面删除了一条--cwy、zyx
        layout.addStretch()
        layout.addLayout(btn_layout)

    # 新增cwy\zyx
    def _load_settings(self):
        """从全局设置加载数据到界面"""
        try:
            from src.config.settings import settings

            # 1. 加载复选框
            self.chk_sound.setChecked(settings.enable_sound)
            self.chk_landmarks.setChecked(settings.show_landmarks)
            self.chk_lines.setChecked(settings.show_lines)

            # 音量是浮点数，转成字符串匹配
            idx_vol = self.combo_volume.findText(str(settings.alert_volume))
            if idx_vol >= 0:
                self.combo_volume.setCurrentIndex(idx_vol)

            # 阈值是整数，转成字符串匹配
            idx_th = self.combo_threshold.findText(str(settings.posture_threshold))
            if idx_th >= 0:
                self.combo_threshold.setCurrentIndex(idx_th)

        except Exception as e:
            print(f"加载设置失败: {e}")

    # 新增 cwy\zyx
    def accept(self):
        """保存设置逻辑"""
        try:
            from src.config.settings import settings

            # 下拉框里存的是字符串，需要转回 float/int
            volume = float(self.combo_volume.currentText())
            threshold = int(self.combo_threshold.currentText())

            # 2. 获取复选框状态
            enable_sound = self.chk_sound.isChecked()
            show_landmarks = self.chk_landmarks.isChecked()
            show_lines = self.chk_lines.isChecked()


            # 3. 更新全局设置
            settings.alert_volume = volume
            settings.posture_threshold = threshold
            settings.enable_sound = enable_sound
            settings.show_landmarks = show_landmarks
            settings.show_lines = show_lines
            settings.save_to_file()  # 保存到磁盘

            # 4. 发射信号通知主窗口
            self.parent().signal_apply_settings.emit({
                'alert_volume': volume,
                'posture_threshold': threshold,
                'enable_sound': enable_sound,
                'show_landmarks': show_landmarks,
                'show_connections': show_lines
            })

            super().accept()

        except Exception as e:
            print(f"保存设置失败: {e}")


# --wsy sy--
class ReportDialog(QDialog):
    """历史记录可视化弹窗"""

    def __init__(self, parent=None, data_dir="data"):
        super().__init__(parent)
        self.setWindowTitle("历史记录可视化")
        self.setFixedSize(1000, 700)
        self.setWindowModality(Qt.ApplicationModal)
        self.data_dir = data_dir
        self.selected_filename = None  # 存储当前选中的会话文件名
        self.filtered_files = []  # 存储过滤后的文件列表
        self._init_ui()

    def _init_ui(self):
        # 创建主布局
        main_layout = QVBoxLayout(self)

        # 创建滚动区域
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("QScrollArea { border: none; }")

        # 创建滚动区域的内容widget
        scroll_content = QWidget()
        layout = QVBoxLayout(scroll_content)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        # 顶部：会话选择
        session_group = QGroupBox("📅 会话选择")
        session_group.setMinimumHeight(200)  # 增加QGroupBox高度
        session_layout = QVBoxLayout(session_group)

        self.session_list = QListWidget()
        self.session_list.setMinimumHeight(150)  # 增加高度

        # 设置字体大小
        font = self.session_list.font()
        font.setPointSize(11)  # 增加字体大小
        self.session_list.setFont(font)

        # 设置行高
        self.session_list.setSpacing(5)  # 增加行间距

        # 去掉边框
        self.session_list.setStyleSheet("QListWidget { border: none; }")

        self.session_list.itemClicked.connect(self._on_session_selected)
        session_layout.addWidget(self.session_list)

        # 中间：可视化选项
        option_group = QGroupBox("⚙️ 可视化选项")
        option_layout = QHBoxLayout(option_group)
        option_layout.setSpacing(15)

        # 时间范围选择
        time_range_box = QWidget()
        time_range_box.setMinimumWidth(200)
        time_range_box.setMaximumWidth(200)
        time_range_layout = QVBoxLayout(time_range_box)
        time_range_layout.addWidget(QLabel("时间范围:"))
        self.time_range_combo = QComboBox()
        self.time_range_combo.addItems(["日", "月", "年"])
        self.time_range_combo.currentIndexChanged.connect(self._update_date_combo)
        time_range_layout.addWidget(self.time_range_combo)
        option_layout.addWidget(time_range_box)

        # 日期选择
        date_box = QWidget()
        date_box.setMinimumWidth(300)  # 设置固定宽度，确保布局稳定
        date_box.setMaximumWidth(300)
        date_layout = QVBoxLayout(date_box)

        # 保存"选择日期："标签的引用
        self.date_label = QLabel("选择日期:")
        date_layout.addWidget(self.date_label)

        # 年月日选择组件
        date_picker_layout = QHBoxLayout()
        date_picker_layout.setSpacing(10)

        # 年份选择
        year_layout = QHBoxLayout()
        self.year_combo = QComboBox()
        self.year_combo.setMinimumWidth(80)
        year_layout.addWidget(self.year_combo)
        self.year_label = QLabel("年")
        year_layout.addWidget(self.year_label)
        date_picker_layout.addLayout(year_layout)

        # 月份选择
        month_layout = QHBoxLayout()
        self.month_combo = QComboBox()
        self.month_combo.setMinimumWidth(60)
        month_layout.addWidget(self.month_combo)
        self.month_label = QLabel("月")
        month_layout.addWidget(self.month_label)
        date_picker_layout.addLayout(month_layout)

        # 日期选择
        day_layout = QHBoxLayout()
        self.day_combo = QComboBox()
        self.day_combo.setMinimumWidth(60)
        day_layout.addWidget(self.day_combo)
        self.day_label = QLabel("日")
        day_layout.addWidget(self.day_label)
        date_picker_layout.addLayout(day_layout)

        date_layout.addLayout(date_picker_layout)
        option_layout.addWidget(date_box)

        # 应用按钮
        apply_btn = QPushButton("应用设置")
        apply_btn.setStyleSheet("""
            QPushButton {
                background-color: #3b82f6;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px 12px;
                min-height: 38px;
            }
            QPushButton:hover {
                background-color: #2563eb;
            }
        """)
        apply_btn.clicked.connect(self._on_apply_settings)
        option_layout.addWidget(apply_btn)

        # 可视化区域
        viz_group = QGroupBox("📊 数据可视化")
        viz_group.setMinimumHeight(400)  # 增加数据可视化模块的最小高度
        viz_layout = QVBoxLayout(viz_group)

        # 图表类型选择
        self.chart_type_widget = QWidget()
        self.chart_type_layout = QHBoxLayout(self.chart_type_widget)
        self.chart_type_label = QLabel("图表类型:")
        self.chart_type_combo = QComboBox()
        self.chart_type_combo.addItems(["状态时序图", "饼图"])
        self.chart_type_combo.currentIndexChanged.connect(self._on_chart_type_changed)
        self.chart_type_layout.addWidget(self.chart_type_label)
        self.chart_type_layout.addWidget(self.chart_type_combo)
        self.chart_type_layout.addStretch()

        # 添加"返回统计"按钮
        self.btn_back_to_stats = QPushButton("返回统计")
        self.btn_back_to_stats.setStyleSheet("""
            QPushButton {
                background-color: #6b7280;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 6px 12px;
                min-height: 28px;
            }
            QPushButton:hover {
                background-color: #4b5563;
            }
        """)
        self.btn_back_to_stats.clicked.connect(self._on_back_to_stats)
        self.chart_type_layout.addWidget(self.btn_back_to_stats)

        viz_layout.addWidget(self.chart_type_widget)

        # 初始隐藏图表类型选择
        self.chart_type_widget.hide()

        # 图表区域
        self.charts_layout = QHBoxLayout()
        self.charts_layout.setSpacing(15)

        # 饼图
        self.pie_canvas = FigureCanvas(Figure(figsize=(6, 4)))
        # --wsy--
        # 安装事件过滤器，允许鼠标滚轮事件传递给滚动区域
        self.pie_canvas.installEventFilter(self)
        # --wsy--

        # 时间序列图
        self.time_canvas = FigureCanvas(Figure(figsize=(8, 4)))
        # --wsy--
        # 安装事件过滤器，允许鼠标滚轮事件传递给滚动区域
        self.time_canvas.installEventFilter(self)
        # --wsy--

        viz_layout.addLayout(self.charts_layout)

        # 默认显示时序图
        self._on_chart_type_changed(0)

        # 底部：统计信息
        stats_group = QGroupBox("📈 统计信息")
        stats_layout = QFormLayout(stats_group)

        self.lbl_total_duration = QLabel("0 分钟")
        self.lbl_bad_posture = QLabel("0%")
        self.lbl_most_common = QLabel("无")
        self.lbl_session_count = QLabel("0")

        stats_layout.addRow("总持续时间:", self.lbl_total_duration)
        stats_layout.addRow("不良坐姿占比:", self.lbl_bad_posture)
        stats_layout.addRow("最常见坐姿:", self.lbl_most_common)
        stats_layout.addRow("会话数量:", self.lbl_session_count)

        # 按钮
        btn_layout = QHBoxLayout()
        btn_export = QPushButton("导出数据")
        btn_export.clicked.connect(self._on_export_data)
        btn_close = QPushButton("关闭")
        btn_close.setMinimumHeight(38)
        btn_close.clicked.connect(self.reject)
        btn_layout.addWidget(btn_export)
        btn_layout.addStretch()
        btn_layout.addWidget(btn_close)

        layout.addWidget(option_group)
        layout.addWidget(session_group)
        layout.addWidget(viz_group, 2)
        layout.addWidget(stats_group)
        layout.addLayout(btn_layout)

        # 设置滚动区域的内容
        scroll_area.setWidget(scroll_content)

        # 将滚动区域添加到主布局
        main_layout.addWidget(scroll_area)

        # 初始化日期下拉列表
        self._update_date_combo()

        # 加载会话列表
        self._load_session_list()

    def _load_session_list(self):
        """加载会话列表"""
        self.session_list.clear()
        self.filtered_files = []  # 清空过滤后的文件列表
        if not os.path.exists(self.data_dir):
            return

        files = [f for f in os.listdir(self.data_dir) if f.startswith("session_") and f.endswith(".json")]
        files.sort(reverse=True)  # 最新的在前

        # 获取当前选择的时间范围和日期
        time_range = self.time_range_combo.currentText()

        # 根据时间范围获取选择的日期
        selected_date = None
        if time_range == "日":
            try:
                year = int(self.year_combo.currentText())
                month = int(self.month_combo.currentText())
                day = int(self.day_combo.currentText())
                selected_date = datetime(year, month, day)
            except:
                pass
        elif time_range == "月":
            try:
                year = int(self.year_combo.currentText())
                month = int(self.month_combo.currentText())
                selected_date = datetime(year, month, 1)
            except:
                pass
        elif time_range == "年":
            try:
                year = int(self.year_combo.currentText())
                selected_date = datetime(year, 1, 1)
            except:
                pass

        for file in files:
            try:
                filepath = os.path.join(self.data_dir, file)
                # 从文件名中提取时间信息
                # 文件名格式: session_年月日_时分秒.json
                # 例如: session_20260411_144117.json
                try:
                    # 提取文件名中的时间部分
                    time_str = file.split("_")[1] + file.split("_")[2].split(".")[0]
                    if len(time_str) == 14:
                        # 前8位是年月日，后6位是时分秒
                        year = int(time_str[0:4])
                        month = int(time_str[4:6])
                        day = int(time_str[6:8])
                        hour = int(time_str[8:10])
                        minute = int(time_str[10:12])
                        second = int(time_str[12:14])
                        dt = datetime(year, month, day, hour, minute, second)
                    else:
                        # 如果文件名格式不正确，尝试从文件中读取start_time
                        with open(filepath, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        start_time = data.get("start_time", "")
                        if start_time:
                            dt = datetime.fromisoformat(start_time)
                        else:
                            # 如果都没有时间信息，跳过
                            continue
                except Exception:
                    # 如果文件名格式不正确，尝试从文件中读取start_time
                    with open(filepath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    start_time = data.get("start_time", "")
                    if start_time:
                        dt = datetime.fromisoformat(start_time)
                    else:
                        # 如果都没有时间信息，跳过
                        continue

                # 读取持续时间
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                duration = data.get('total_duration', 0)

                # 根据时间范围过滤会话
                if selected_date:
                    if time_range == "日":
                        # 只显示当天的会话
                        if dt.year == selected_date.year and dt.month == selected_date.month and dt.day == selected_date.day:
                            # 计算时长
                            duration_text = self._format_duration(duration)
                            # 只显示时间，不显示日期
                            display_text = f"{dt.strftime('%H:%M:%S')} - 时长: {duration_text}"
                            self.session_list.addItem(display_text)
                            self.filtered_files.append(file)  # 添加到过滤后的文件列表
                    elif time_range == "月":
                        # 只显示当月的会话
                        if dt.year == selected_date.year and dt.month == selected_date.month:
                            # 计算时长
                            duration_text = self._format_duration(duration)
                            # 不显示年份，只显示月日和时间
                            display_text = f"{dt.strftime('%m-%d %H:%M:%S')} - 时长: {duration_text}"
                            self.session_list.addItem(display_text)
                            self.filtered_files.append(file)  # 添加到过滤后的文件列表
                    elif time_range == "年":
                        # 只显示当年的会话
                        if dt.year == selected_date.year:
                            # 计算时长
                            duration_text = self._format_duration(duration)
                            # 不显示年份，只显示月日和时间
                            display_text = f"{dt.strftime('%m-%d %H:%M:%S')} - 时长: {duration_text}"
                            self.session_list.addItem(display_text)
                            self.filtered_files.append(file)  # 添加到过滤后的文件列表
                else:
                    # 如果没有选择日期，显示所有会话
                    # 计算时长
                    duration_text = self._format_duration(duration)
                    display_text = f"{dt.strftime('%Y-%m-%d %H:%M:%S')} - 时长: {duration_text}"
                    self.session_list.addItem(display_text)
                    self.filtered_files.append(file)  # 添加到过滤后的文件列表
            except Exception:
                continue

    # --sy--
    def _on_session_selected(self, item):
        """选择会话后的处理"""
        index = self.session_list.row(item)
        if index < len(self.filtered_files):
            self.selected_filename = self.filtered_files[index]  # 从过滤后的文件列表中获取文件名
            print(f"Selected session file: {self.selected_filename}")  # 打印打开的文件名
            self._load_session_data(self.selected_filename)
            # 显示图表类型选择
            self.chart_type_widget.show()
            # 默认选择饼图
            self.chart_type_combo.setCurrentIndex(0)
            self._on_chart_type_changed(0)

    # --sy--
    # --wsy--
    def _load_session_data(self, filename):
        """加载会话数据并可视化"""
        try:
            print(f"Loading session data from: {filename}")  # 打印加载的文件名
            filepath = os.path.join(self.data_dir, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                session_data = json.load(f)

            # 解析会话数据
            posture_data = session_data.get('posture_data', [])

            # 根据当前选择的图表类型生成相应的图表
            chart_type = self.chart_type_combo.currentText()
            if chart_type == "状态时序图":
                self._generate_time_series_chart(session_data)
            elif chart_type == "饼图":
                self._generate_pie_chart(self.time_range_combo.currentText(),
                                         self.session_list.currentItem().text() if self.session_list.currentItem() else None)
        except Exception as e:
            print(f"加载会话数据出错: {e}")

    # --wsy--

    def _format_duration(self, total_seconds):
        """格式化时长为友好的显示格式"""
        total_minutes = int(total_seconds / 60)
        hours = total_minutes // 60
        minutes = total_minutes % 60
        seconds = int(total_seconds % 60)
        if hours > 0:
            return f"{hours}小时{minutes}分钟"
        elif minutes > 0:
            return f"{minutes}分钟{seconds}秒"
        else:
            return f"{seconds}秒"

    # --sy--
    def _on_apply_settings(self):
        """应用设置"""
        # 重新加载会话列表，根据选择的日期过滤
        self._load_session_list()

        # 1. 获取界面选择的范围类型
        range_type = self.time_range_combo.currentText()

        # 2. 获取开始和结束日期
        year = int(self.year_combo.currentText())

        # 默认值设置，防止下拉框为空
        month = 1
        day = 1

        try:
            # 尝试获取月份
            month_text = self.month_combo.currentText()
            if month_text.isdigit():
                month = int(month_text)
        except:
            pass

        try:
            # 只有在“日”模式下才获取具体日期，否则默认为1
            if range_type == "日":
                day_text = self.day_combo.currentText()
                if day_text.isdigit():
                    day = int(day_text)
        except:
            pass

        # 组合成 Python 的 date 对象
        start_date = date(year, month, day)

        # --- 关键修正：根据 range_type 动态计算 end_date ---
        # 因为 JSON 里存储的是每天的数据，我们需要确保范围覆盖所有相关日期
        if range_type == "日":
            end_date = start_date  # 同一天
        elif range_type == "月":
            # 假设该月最多31天，为了简单匹配，我们将结束日期设为该月最后一天
            # 这里简单粗暴地设为31，实际逻辑中 PieChart 只会读取存在的日期
            last_day = calendar.monthrange(year, month)[1]
            end_date = date(year, month, last_day)
        else:  # "年"
            # 年范围，结束日期设为年底
            end_date = date(year, 12, 31)

        # 3. 调用新方法
        self._draw_summary_pie_chart(range_type, start_date, end_date)

    # --sy--

    def _update_date_combo(self):
        """根据时间范围更新日期下拉列表"""
        time_range = self.time_range_combo.currentText()

        # 获取当前日期
        current_date = datetime.now()

        # 清空所有下拉列表
        self.year_combo.clear()
        self.month_combo.clear()
        self.day_combo.clear()

        # 生成年份选项（最近5年）
        for i in range(5):
            year = current_date - relativedelta(years=i)
            self.year_combo.addItem(str(year.year))

        # 生成月份选项（1-12月）
        for i in range(1, 13):
            self.month_combo.addItem(str(i).zfill(2))

        # 始终显示"选择日期："标签
        self.date_label.show()

        # 根据时间范围显示/隐藏相应的下拉列表
        if time_range == "日":
            # 显示年、月、日
            self.year_combo.show()
            self.month_combo.show()
            self.day_combo.show()
            # 显示标签
            self.year_label.show()
            self.month_label.show()
            self.day_label.show()

            # 生成日期选项（根据当前选择的年月确定天数）
            self._update_day_combo()
            # 连接月份和年份变化信号
            self.month_combo.currentIndexChanged.connect(self._update_day_combo)
            self.year_combo.currentIndexChanged.connect(self._update_day_combo)
        elif time_range == "月":
            # 显示年、月，隐藏日
            self.year_combo.show()
            self.month_combo.show()
            self.day_combo.hide()
            # 显示标签
            self.year_label.show()
            self.month_label.show()
            self.day_label.hide()
        elif time_range == "年":
            # 只显示年，隐藏月、日
            self.year_combo.show()
            self.month_combo.hide()
            self.day_combo.hide()
            # 显示标签
            self.year_label.show()
            self.month_label.hide()
            self.day_label.hide()
        elif time_range == "周":
            # 周模式下显示年、月、日
            self.year_combo.show()
            self.month_combo.show()
            self.day_combo.show()
            # 显示标签
            self.year_label.show()
            self.month_label.show()
            self.day_label.show()

            # 生成日期选项
            self._update_day_combo()
            # 连接月份和年份变化信号
            self.month_combo.currentIndexChanged.connect(self._update_day_combo)
            self.year_combo.currentIndexChanged.connect(self._update_day_combo)

    def _update_day_combo(self):
        """根据选择的年月更新日期选项"""
        self.day_combo.clear()

        try:
            # 获取选择的年和月
            year = int(self.year_combo.currentText())
            month = int(self.month_combo.currentText())

            # 计算该月的天数
            if month == 2:
                # 检查是否是闰年
                if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
                    days = 29
                else:
                    days = 28
            elif month in [4, 6, 9, 11]:
                days = 30
            else:
                days = 31

            # 生成日期选项
            for i in range(1, days + 1):
                self.day_combo.addItem(str(i).zfill(2))
        except:
            # 如果年或月未选择，不生成日期选项
            pass

    # --wsy--
    def _on_chart_type_changed(self, index):
        """图表类型选择变化时的处理"""
        # 清空图表区域（包括伸缩空间）
        while self.charts_layout.count() > 0:
            item = self.charts_layout.takeAt(0)
            # 无论是否有widget，都移除该项
            if item:
                widget = item.widget()
                if widget:
                    widget.hide()
                    self.charts_layout.removeWidget(widget)

        # 重新添加伸缩空间
        self.charts_layout.addStretch()

        # 获取数据并更新统计面板（这部分逻辑保留）
        time_range = self.time_range_combo.currentText()
        selected_item = self.session_list.currentItem()
        selected_session_text = selected_item.text() if selected_item else None

        # 根据选择的图表类型显示相应的图表
        if index == 0:  # 状态时序图
            # 清除旧图表
            self.time_canvas.figure.clear()

            # 加载会话数据并生成时序图
            if selected_item and self.selected_filename:
                print(f"Generating time series chart for: {self.selected_filename}")  # 打印生成时序图的文件名
                self._load_session_data(self.selected_filename)

            # 添加时序图到布局
            self.charts_layout.addWidget(self.time_canvas)
            self.time_canvas.show()

        elif index == 1:  # 饼图
            # 清除旧图表
            self.pie_canvas.figure.clear()

            # 获取当前选择的时间范围
            time_range = self.time_range_combo.currentText()

            # 获取选中的会话（如果没有选中，则为None）
            selected_item = self.session_list.currentItem()
            selected_session_text = selected_item.text() if selected_item else None

            # 调用绘图函数，根据是否有选中的会话决定是绘制"范围汇总"还是"单个会话"
            self._generate_pie_chart(time_range, selected_session_text)

            # 添加饼图到布局
            self.charts_layout.addWidget(self.pie_canvas)
            self.pie_canvas.show()

        # 添加右侧伸缩空间
        self.charts_layout.addStretch()

    # --wsy--

    def _on_back_to_stats(self):
        """返回统计页面"""
        # 清空图表区域
        for i in reversed(range(self.charts_layout.count())):
            widget = self.charts_layout.itemAt(i).widget()
            if widget:
                widget.hide()
                self.charts_layout.removeWidget(widget)

        # 隐藏图表类型选择和返回按钮
        self.chart_type_widget.hide()

        # 取消会话选择
        self.session_list.clearSelection()

        # 清空选中的文件名
        self.selected_filename = None

        # 重新加载会话列表，根据选择的日期过滤
        self._load_session_list()

        # 1. 获取界面选择的范围类型
        range_type = self.time_range_combo.currentText()

        # 2. 获取开始和结束日期
        year = int(self.year_combo.currentText())

        # 默认值设置，防止下拉框为空
        month = 1
        day = 1

        try:
            # 尝试获取月份
            month_text = self.month_combo.currentText()
            if month_text.isdigit():
                month = int(month_text)
        except:
            pass

        try:
            # 只有在“日”模式下才获取具体日期，否则默认为1
            if range_type == "日":
                day_text = self.day_combo.currentText()
                if day_text.isdigit():
                    day = int(day_text)
        except:
            pass

        # 组合成 Python 的 date 对象
        start_date = date(year, month, day)

        # --- 关键修正：根据 range_type 动态计算 end_date ---
        # 因为 JSON 里存储的是每天的数据，我们需要确保范围覆盖所有相关日期
        if range_type == "日":
            end_date = start_date  # 同一天
        elif range_type == "月":
            # 假设该月最多31天，为了简单匹配，我们将结束日期设为该月最后一天
            # 这里简单粗暴地设为31，实际逻辑中 PieChart 只会读取存在的日期
            last_day = calendar.monthrange(year, month)[1]
            end_date = date(year, month, last_day)
        else:  # "年"
            # 年范围，结束日期设为年底
            end_date = date(year, 12, 31)

        # 3. 调用新方法
        self._draw_summary_pie_chart(range_type, start_date, end_date)

    def _on_export_data(self):
        """导出数据"""
        # 这里将来会实现导出数据的逻辑
        # 目前只是一个框架，预留接口
        pass

    # --sy--
    # 新增饼图画法
    def _generate_pie_chart(self, time_range, selected_session_text=None):
        """
        生成饼图（优化版：直接基于统计文件查找，避免字符串解析错误）
        """
        # --- 1. 初始化 ---
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

        # 获取界面上选择的年份
        selected_year = self.year_combo.currentText()
        stats_file = os.path.join(self.data_dir, f"year_{selected_year}_stats.json")
        ax = self.pie_canvas.figure.add_subplot(111)
        ax.clear()

        if not os.path.exists(stats_file):
            ax.text(0.5, 0.5, '统计数据文件不存在', horizontalalignment='center', verticalalignment='center',
                    fontsize=12, color='red')
            ax.set_axis_off()
            self.pie_canvas.draw()
            return

        try:
            with open(stats_file, 'r', encoding='utf-8') as f:
                stats_data = json.load(f)

            total_durations = {}
            total_duration = 0

            # --- 2. 核心逻辑分流 ---
            if selected_session_text:
                # --- 模式 A：绘制单次会话饼图（优化逻辑）---
                found = False
                months_data = stats_data.get("months", {})

                # 获取当前选中的月份和日期（用于后续查找）
                # 注意：这里我们不再从 selected_session_text 解析，而是尝试从统计文件中匹配

                # 由于 selected_session_text 是列表中显示的文本，我们尝试从中提取最核心的信息
                # 例如 "04-12 14:22:19 - 时长: 43秒" 或 "04-12 - 时长: 2小时"
                # 提取时间部分作为唯一标识
                try:
                    # 尝试提取 "MM-DD HH:MM:SS" 或 "MM-DD" 部分
                    time_part = selected_session_text.split(" - ")[0]

                    # 遍历统计文件中的所有数据来寻找匹配的会话
                    for month_key, month_val in months_data.items():
                        days_data = month_val.get("days", {})
                        for day_key, day_val in days_data.items():
                            sessions = day_val.get("sessions", [])
                            for session in sessions:
                                # 获取统计文件中存储的真实日志文件名
                                log_filename = session.get("log_filename", "")

                                # 判断逻辑：如果界面上的文本包含日志文件名中的时间戳，或者两者能对应上
                                # 由于直接匹配文本困难，我们改用：只要界面上选中了会话，我们就取统计文件中该日期下的第一个会话或根据时长匹配
                                # 但更稳妥的方式是：统计文件中的 session 里通常有 start_time 或 duration，我们可以利用 duration 来辅助判断

                                # 简化逻辑：既然界面上的会话是根据 stats_data 加载出来的，selected_filename 应该已经在 _on_session_selected 中确定了
                                # 我们直接使用 self.selected_filename 进行匹配
                                if hasattr(self, 'selected_filename') and self.selected_filename:
                                    if log_filename == self.selected_filename:
                                        # 找到匹配项，提取姿势数据
                                        session_postures = session.get("postures", {})
                                        for posture, duration in session_postures.items():
                                            if posture != "paused":
                                                total_durations[posture] = total_durations.get(posture, 0) + duration
                                                total_duration += duration
                                        found = True
                                        break
                            if found:
                                break
                        if found:
                            break
                except Exception as e:
                    print(f"查找会话数据时出错: {e}")

                if not found:
                    ax.text(0.5, 0.5, '未找到该会话数据', horizontalalignment='center', verticalalignment='center',
                            fontsize=10, color='red')
                    ax.set_axis_off()
                    self.pie_canvas.draw()
                    return

            # --- 3. 绘图逻辑 ---
            if not total_durations or total_duration == 0:
                ax.text(0.5, 0.5, '无有效姿势数据', horizontalalignment='center', verticalalignment='center',
                        fontsize=12)
                ax.set_axis_off()
            else:
                # 数据清洗：合并微小数据
                labels = []
                sizes = []
                threshold = total_duration * 0.01  # 1% 阈值
                others = 0
                for label, size in total_durations.items():
                    if size < threshold:
                        others += size
                    else:
                        labels.append(label)
                        sizes.append(size)
                if others > 0:
                    labels.append("其他")
                    sizes.append(others)

                # 优化颜色
                colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0', '#ffb3e6']
                plot_colors = [colors[i % len(colors)] for i in range(len(labels))]

                # 绘制饼图
                wedges, texts, autotexts = ax.pie(
                    sizes,
                    labels=None,
                    colors=plot_colors,
                    autopct='%1.1f%%',
                    startangle=90,
                    pctdistance=0.6,
                    textprops={'fontsize': 10, 'color': 'white', 'weight': 'bold'}
                )

                # 添加图例
                ax.legend(
                    wedges,
                    labels,
                    title="姿势类别",
                    loc="center left",
                    bbox_to_anchor=(0.9, 0, 0.5, 1),
                    fontsize=9,
                    frameon=True,
                    title_fontsize=10
                )

                # 设置标题
                title_text = f"{selected_year}年"
                if time_range == "月":
                    title_text += f"{self.month_combo.currentText()}月 "
                elif time_range == "日":
                    title_text += f"{self.month_combo.currentText()}月{self.day_combo.currentText()}日 "

                if selected_session_text:
                    title_text += f"- 单次会话"
                else:
                    title_text += f"- {time_range}汇总"

                ax.set_title(title_text, fontsize=12, pad=20, fontweight='bold')
                ax.axis('equal')

                # 调整布局防止图例被切
                self.pie_canvas.figure.subplots_adjust(top=0.8, bottom=0.1, left=0.1, right=0.75)

            self.pie_canvas.draw()

        except Exception as e:
            print(f"生成饼图出错: {e}")
            ax.text(0.5, 0.5, f'绘图错误: {str(e)}', horizontalalignment='center', verticalalignment='center',
                    fontsize=10, color='red')
            ax.set_axis_off()
            self.pie_canvas.draw()

    # --sy--
    # --sy--
    def _update_stats_info(self, durations_dict, total_dur):
        """
        根据传入的数据更新右侧的统计标签
        """
        if total_dur == 0:
            self.lbl_total_duration.setText("0 分钟")
            self.lbl_bad_posture.setText("0%")
            self.lbl_most_common.setText("无")
            return

        # 1. 总持续时间
        # 确保总时长为整数，避免小数位
        total_dur = int(total_dur)
        hours = total_dur // 3600
        minutes = (total_dur % 3600) // 60
        seconds = total_dur % 60

        if hours > 0:
            self.lbl_total_duration.setText(f"{hours}小时{minutes}分钟")
        elif minutes > 0:
            if seconds > 0:
                self.lbl_total_duration.setText(f"{minutes}分钟{seconds}秒")
            else:
                self.lbl_total_duration.setText(f"{minutes}分钟")
        else:
            self.lbl_total_duration.setText(f"{seconds}秒")

        # 2. 不良坐姿占比 & 3. 最常见坐姿
        # 假设 "Good posture" 是良好坐姿，其余均为不良
        good_duration = durations_dict.get("Good posture", 0)
        bad_duration = total_dur - good_duration
        bad_ratio = (bad_duration / total_dur) * 100

        self.lbl_bad_posture.setText(f"{bad_ratio:.1f}%")

        # 寻找最常见坐姿（排除暂停）
        max_posture = "未知"
        max_duration = 0
        for posture, duration in durations_dict.items():
            if posture != "paused" and duration > max_duration:
                max_duration = duration
                max_posture = posture
        self.lbl_most_common.setText(max_posture)

    # --sy--

    # --sy--
    def _draw_summary_pie_chart(self, range_type, start_date, end_date):
        """
        修正版：读取对应年份的统计JSON并绘制汇总饼图
        """
        # --- 0. 解决中文显示问题 ---
        import matplotlib
        matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']
        matplotlib.rcParams['axes.unicode_minus'] = False

        # --- 1. 使用正确的数据目录 ---
        data_dir = self.data_dir  # 直接使用实例变量
        year = start_date.year
        json_filename = f"year_{year}_stats.json"
        json_file_path = os.path.join(data_dir, json_filename)

        # 调试输出
        print(f"正在查找汇总数据文件: {json_file_path}")

        # --- 2. 检查文件是否存在 ---
        if not os.path.exists(json_file_path):
            print(f"错误：未找到文件 {json_file_path}")
            self.pie_canvas.figure.clear()
            ax = self.pie_canvas.figure.add_subplot(111)
            ax.text(0.5, 0.5, f'未找到数据文件:\n{json_filename}',
                    horizontalalignment='center', verticalalignment='center',
                    fontsize=12, color='red')
            ax.set_axis_off()
            self.pie_canvas.draw()
            return

        try:
            # --- 3. 读取并汇总数据（修正数据结构）---
            with open(json_file_path, 'r', encoding='utf-8') as f:
                stats_data = json.load(f)

            total_durations = {}
            total_duration = 0

            # 获取月份数据
            months_data = stats_data.get("months", {})

            # 确定需要处理的月份
            if range_type == "年":
                target_months = list(months_data.keys())
            elif range_type == "月":
                target_months = [f"{start_date.month:02d}"]
            else:  # "日"
                target_months = [f"{start_date.month:02d}"]

            # 遍历数据
            for month_key in target_months:
                if month_key not in months_data:
                    continue

                month_data = months_data[month_key]
                days_data = month_data.get("days", {})

                # 确定需要处理的天数
                if range_type == "日":
                    target_days = [f"{start_date.day:02d}"]
                else:
                    target_days = list(days_data.keys())

                # 累加姿势时长
                # 累加姿势时长，同时收集 valid_total_duration
                valid_total_from_sessions = 0  # 从统计文件中获取的有效总时长
                session_count = 0

                for day_key in target_days:
                    if day_key not in days_data:
                        continue

                    day_data = days_data[day_key]
                    sessions = day_data.get("sessions", [])

                    for session in sessions:
                        session_count += 1
                        # 【核心修改】直接使用统计文件中的 valid_total_duration
                        session_valid_total = session.get("valid_total_duration", 0)
                        valid_total_from_sessions += session_valid_total

                        postures = session.get("postures", {})
                        for posture, duration in postures.items():
                            if posture != "paused":  # 排除暂停状态
                                # 确保时长为整数，避免小数位
                                int_duration = int(duration)
                                total_durations[posture] = total_durations.get(posture, 0) + int_duration
                                total_duration += int_duration

                # 【关键】如果统计文件中有 valid_total_duration，优先使用
                # 否则使用累加值作为后备
                if valid_total_from_sessions > 0:
                    total_duration = valid_total_from_sessions
                    print(f"使用统计文件中的有效总时长: {total_duration}秒")
                else:
                    print(f"使用累加的有效总时长: {total_duration}秒")

            # --- 4. 绘图逻辑 ---
            self.pie_canvas.figure.clear()
            ax = self.pie_canvas.figure.add_subplot(111)

            if not total_durations or total_duration == 0:
                ax.text(0.5, 0.5, '该时间段内无有效姿势数据',
                        horizontalalignment='center', verticalalignment='center',
                        fontsize=12)
                ax.set_axis_off()
            else:
                # 数据清洗：合并小于1%的数据为"其他"
                labels = []
                sizes = []
                threshold = total_duration * 0.01
                others = 0

                for label, size in total_durations.items():
                    if size < threshold:
                        others += size
                    else:
                        labels.append(label)
                        sizes.append(size)

                if others > 0:
                    labels.append("其他")
                    sizes.append(others)

                # 颜色配置
                colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0', '#ffb3e6']
                plot_colors = [colors[i % len(colors)] for i in range(len(labels))]

                # 绘制饼图
                wedges, texts, autotexts = ax.pie(
                    sizes,
                    labels=None,
                    colors=plot_colors,
                    autopct='%1.1f%%',
                    startangle=90,
                    pctdistance=0.6,
                    textprops={'fontsize': 10, 'color': 'white', 'weight': 'bold'}
                )

                # 图例
                ax.legend(
                    wedges,
                    labels,
                    title="姿势类别",
                    loc="center left",
                    bbox_to_anchor=(0.9, 0, 0.5, 1),
                    fontsize=9,
                    title_fontsize=10
                )

                # 动态标题
                title_text = f"{year}年"
                if range_type == "月":
                    title_text += f"{start_date.month}月"
                elif range_type == "日":
                    title_text += f"{start_date.month}月{start_date.day}日"
                title_text += f" {range_type}汇总"

                ax.set_title(title_text, fontsize=12, pad=20, fontweight='bold')
                ax.axis('equal')
                self.pie_canvas.figure.subplots_adjust(top=0.85, bottom=0.1, left=0.1, right=0.75)

            # 确保饼图显示在图表布局中
            # 清空图表布局
            while self.charts_layout.count() > 0:
                item = self.charts_layout.takeAt(0)
                if item:
                    widget = item.widget()
                    if widget:
                        widget.hide()
                        self.charts_layout.removeWidget(widget)

            # 重新添加伸缩空间
            self.charts_layout.addStretch()
            # 添加饼图到布局
            self.charts_layout.addWidget(self.pie_canvas)
            self.pie_canvas.show()
            # 添加右侧伸缩空间
            self.charts_layout.addStretch()

            self.pie_canvas.draw()
            print(f"汇总饼图绘制完成，总时长: {total_duration}秒")

            # 更新统计信息
            self._update_stats_info(total_durations, total_duration)
            self.lbl_session_count.setText(str(len(self.filtered_files)))

            # 隐藏图表类型选择，因为汇总饼图不需要切换图表类型
            self.chart_type_widget.hide()

        except Exception as e:
            print(f"生成汇总饼图出错: {e}")
            import traceback
            traceback.print_exc()

            self.pie_canvas.figure.clear()
            ax = self.pie_canvas.figure.add_subplot(111)
            ax.text(0.5, 0.5, f'绘图错误:\n{str(e)}',
                    horizontalalignment='center', verticalalignment='center',
                    fontsize=10, color='red')
            ax.set_axis_off()
            self.pie_canvas.draw()

    # --sy--

    # --wsy--
    def _generate_time_series_chart(self, session_data):
        """
        生成状态时序图
        横轴为时间，纵轴为状态
        """
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 设置中文显示
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

        # 状态顺序（从下到上）
        posture_order = [
            "paused",
            "No person detected",
            "Leaning on desk",
            "Looking down",
            "Uneven shoulders",
            "Left tilt",
            "Right tilt",
            "Looking up",
            "Good posture"
        ]

        # 为每个状态分配一个数值
        posture_values = {posture: i for i, posture in enumerate(posture_order)}

        # 解析数据，计算时间点和对应的状态值
        segments = []
        current_time = 0

        # 获取总持续时间
        total_duration = session_data.get('total_duration', 0)
        print(f"总持续时长{total_duration}秒")
        posture_data = session_data.get('posture_data', [])

        for item in posture_data:
            posture = item.get('posture')
            duration = item.get('duration', 0)

            if posture in posture_values:
                # 存储每个状态段的信息：(start_time, end_time, posture_value, posture_name)
                segments.append((current_time, current_time + duration, posture_values[posture], posture))
                current_time += duration

        # 清除旧图表
        self.time_canvas.figure.clear()
        ax = self.time_canvas.figure.add_subplot(111)

        if not segments:
            ax.text(0.5, 0.5, '无数据', horizontalalignment='center', verticalalignment='center', fontsize=12)
            ax.set_axis_off()
            self.time_canvas.draw()
            return

        # 为不同姿势设置不同颜色（使用用户指定的RGB值）
        colors = {
            "paused": (12 / 255, 7 / 255, 134 / 255),
            "No person detected": (76 / 255, 2 / 255, 161 / 255),
            "Leaning on desk": (126 / 255, 3 / 255, 167 / 255),
            "Looking down": (203 / 255, 71 / 255, 119 / 255),
            "Uneven shoulders": (229 / 255, 108 / 255, 91 / 255),
            "Left tilt": (248 / 255, 149 / 255, 64 / 255),
            "Right tilt": (251 / 255, 205 / 255, 17 / 255),
            "Looking up": (189 / 255, 214 / 255, 56 / 255),
            "Good posture": (129 / 255, 201 / 255, 152 / 255)
        }

        # 绘制横向条形图（圆角矩形）
        from matplotlib.patches import FancyBboxPatch

        for start, end, value, posture in segments:
            width = end - start
            height = 0.8
            y = value - height / 2

            # 创建圆角矩形（无边框）
            patch = FancyBboxPatch(
                (start, y), width, height,
                boxstyle="round,pad=0.05,rounding_size=0.1",
                facecolor=colors.get(posture, 'gray'),
                edgecolor=None,
                linewidth=0,
                alpha=0.8
            )
            ax.add_patch(patch)

        # 为时间上相连的姿势添加纵向连接线，颜色为二者均值
        for i in range(len(segments) - 1):
            # 获取当前段和下一段
            current_start, current_end, current_value, current_posture = segments[i]
            next_start, next_end, next_value, next_posture = segments[i + 1]

            # 检查是否时间上相连
            if abs(current_end - next_start) < 0.001:  # 允许微小误差
                # 计算两个姿势的颜色均值
                current_color = colors.get(current_posture, (0.5, 0.5, 0.5))
                next_color = colors.get(next_posture, (0.5, 0.5, 0.5))
                mean_color = tuple((c + n) / 2 for c, n in zip(current_color, next_color))

                # 绘制纵向连接线
                ax.plot([current_end, current_end], [current_value - 0.4, next_value + 0.4],
                        color=mean_color, linewidth=1, alpha=0.8)

        # 设置纵轴标签
        ax.set_yticks(range(len(posture_order)))
        ax.set_yticklabels(posture_order)
        ax.invert_yaxis()  # 反转纵轴，使 "Good posture" 在顶部

        # 设置横轴标签和范围
        ax.set_xlabel('时间 (秒)')
        ax.set_xlim(0, max(total_duration, current_time))  # 使用总持续时间作为横轴最大值

        # 设置图表标题
        ax.set_title('状态时序图')

        # 添加网格线
        ax.grid(True, axis='x', linestyle='--', alpha=0.7)

        # 调整边框和坐标轴颜色
        light_color = '#999999'  # 浅灰色

        # 设置边框颜色
        for spine in ax.spines.values():
            spine.set_color(light_color)

        # 设置坐标轴刻度颜色
        ax.tick_params(axis='x', colors=light_color)
        ax.tick_params(axis='y', colors=light_color)

        # 调整布局
        self.time_canvas.figure.tight_layout()

        # 绘制图表
        self.time_canvas.draw()

    # --wsy--

    # --wsy--
    def eventFilter(self, obj, event):
        """
        事件过滤器，处理鼠标滚轮事件，使其传递给滚动区域
        """
        from PySide6.QtCore import QEvent
        if event.type() == QEvent.Wheel:
            # 找到父滚动区域并发送滚动事件
            widget = obj
            while widget.parent():
                widget = widget.parent()
                if isinstance(widget, QScrollArea):
                    # 计算滚动步长
                    scroll_bar = widget.verticalScrollBar()
                    if scroll_bar:
                        # 根据滚轮方向调整滚动位置
                        delta = event.angleDelta().y()
                        scroll_bar.setValue(scroll_bar.value() - delta)
                    return True
        return False
    # --wsy--


# --wsy sy--

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