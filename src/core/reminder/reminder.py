import time
import threading
import pygame
import os
from src.config.settings import settings  # 读取系统设置-cwy\zyx


class MultiModalReminder:
    def __init__(self, enable_sound=True, volume=0.8, threshold=5):#cwy\zyx
        # 核心配置：优先使用传入的参数，否则使用默认值 --cwy\zyx
        self.sound_enabled = enable_sound
        self.volume = volume
        self.first_remind_delay = threshold  # 使用传入的阈值
        # 获取项目根目录的绝对路径
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        # 构建音频文件的绝对路径
        self.audio_file = os.path.join(self.base_dir, "src", "core", "reminder", "audio", "alert.mp3")
        self.repeat_remind_interval = 10  # 重复间隔10秒

        # --- 关键状态变量 ---cwy、zyx
        self.is_currently_bad = False  # 是否正在不良坐姿中
        self.has_reminded_once = False  # 本轮是否已提醒过
        self.bad_posture_start_time = 0  # 不良坐姿开始的时间戳

#以下到remind之间已修改 --cwy\zyx
        # --- 初始化 ---
        self.sound = None
        self._init_audio()  # 调用独立的音频初始化方法

    def _init_audio(self):
        """独立的音频初始化方法，用于在配置改变时重新加载"""
        try:
            # 如果已经初始化，先关闭
            if pygame.mixer.get_init():
                pygame.mixer.quit()

            pygame.mixer.init()
            self.sound = pygame.mixer.Sound(self.audio_file)
            self.set_volume(self.volume)  # 应用当前音量
            print(f"[Reminder] 音频模块初始化成功，音量: {self.volume}")
        except Exception as e:
            print(f"[Reminder] 音频初始化失败：{e}")
            self.sound_enabled = False

    def set_volume(self, volume):
        """安全设置音量"""
        self.volume = volume
        if self.sound:
            self.sound.set_volume(volume)

        # --- 新增：动态更新设置的方法 ---
    def update_settings(self, enable_sound=None, volume=None, threshold=None):
        """
        供外部调用以更新配置
        Args:
            enable_sound: 是否开启声音
            volume: 音量大小 (0.0 - 1.0)
        """
        if enable_sound is not None:
            self.sound_enabled = enable_sound

        if volume is not None:
            self.set_volume(volume)  # 立即应用新音量

        # --- 新增逻辑：接收并更新阈值 ---
        if threshold is not None:
            self.first_remind_delay = threshold
            print(f"[Reminder] 阈值已更新为: {threshold} 秒")
            # 关键：如果阈值变了，重置状态，避免逻辑混乱
            # 例如：旧阈值是10秒，现在改成5秒，如果不重置，可能还需要等5秒才响。
            # 为了立即生效，清空计时
            self._reset_state()

        print(f"[Reminder] 设置已更新: Sound={self.sound_enabled}, Volume={self.volume}, Threshold={self.first_remind_delay}")


    def _play_sound(self):
        if not self.sound_enabled or not self.sound:
            return
        try:
            if pygame.mixer.get_busy():
                pygame.mixer.stop()
            # 使用线程播放，避免阻塞主循环
            threading.Thread(target=self.sound.play, daemon=True).start()
        except Exception as e:
            print(f"[Reminder] 播放音频时出错: {e}")

    def remind(self, is_bad_posture, posture_key="Unknown"):
        """
        修复后的核心逻辑：
        1. 只要 is_bad_posture 为 True，计时器就在走。
        2. 只有 is_bad_posture 为 False 时，才重置所有状态。
        """
        current_time = time.time()

        # --- 情况一：检测到不良姿势 ---
        if is_bad_posture:
            # 如果是刚刚从好姿势变成坏姿势，记录开始时间
            if not self.is_currently_bad:
                self.is_currently_bad = True
                self.bad_posture_start_time = current_time
                self.has_reminded_once = False  # 重置提醒标记，准备新一轮提醒
                print(f"[Reminder] 进入不良姿势: {posture_key}")

            # 计算持续时间
            duration = current_time - self.bad_posture_start_time

            # 1. 首次提醒逻辑 (达到阈值)
            if not self.has_reminded_once and duration >= self.first_remind_delay:
                print(f"[提醒] 首次提醒：不良姿势已持续 {int(duration)} 秒")
                self._play_sound()
                self.has_reminded_once = True


            # 2. 重复提醒逻辑 (达到间隔)
            # 重复提醒判断
            elif self.has_reminded_once and (
                        duration - self.first_remind_delay) % self.repeat_remind_interval < 0.1:  # 简单取模判断或记录last_time
                # 这里为了简化逻辑，也可以用 last_remind_time 判断，防止每帧都响
                pass

        # --- 情况二：姿势良好 ---
        else:
            # 只要姿势好了，就重置所有状态，下次坏了重新计时
            if self.is_currently_bad:
                print("[Reminder] 姿势已恢复，重置计时")
            self.is_currently_bad = False
            self.has_reminded_once = False
            self.bad_posture_start_time = 0
# 这之前到音频初始化都修改了

    def _reset_state(self):
        """重置所有状态变量"""
        self.is_in_bad_posture = False
        self.bad_posture_start = 0
        self.last_remind_time = 0
        self.has_triggered_first = False

# 全局单例
reminder = MultiModalReminder()


if __name__ == "__main__":
    import cv2
    from src.detector.pose_detector import PoseDetector


    def main():
        # 初始化摄像头
        cap = cv2.VideoCapture(0)

        # 创建姿态检测器实例
        detector = PoseDetector()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("无法读取摄像头画面")
                    break

                # 处理每一帧
                image, results = detector.process_frame(frame)

                # 新加的
                from src.core.reminder.reminder import reminder
                reminder.notify(results)

                # 显示结果
                cv2.imshow("Sitting Posture Detection", image)

                # 按 'q' 键退出
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            # 释放资源
            cap.release()
            detector.close()
            cv2.destroyAllWindows()

    main()
