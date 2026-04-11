import os

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QGroupBox, QFormLayout,
    QCheckBox, QDoubleSpinBox, QDialog, QMessageBox, QLineEdit, QComboBox, QFileDialog
)
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt, Signal
import cv2
import numpy as np
import sys
from typing import Dict, Any, Optional

# from numpy.ma.core import angle

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
    # 字体大小调整cwy
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
                font-size: 18px;
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
                font-size: 16px;
                border-radius: 8px;
                padding: 10px 12px;
                min-height: 45px;
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
                font-size: 16px;
            }
            QFormLayout QLabel {
                font-size: 16px;
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
        btn_save.setFixedHeight(35)
        btn_cancel.setFixedHeight(35)
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
