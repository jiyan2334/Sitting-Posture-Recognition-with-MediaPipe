import time
import threading
import pygame
import os


class MultiModalReminder:
    def __init__(self):
        # 核心配置
        self.sound_enabled = True
        # 获取项目根目录的绝对路径
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        # 构建音频文件的绝对路径
        self.audio_file = os.path.join(self.base_dir, "src", "core", "reminder", "audio", "alert.mp3")
        self.first_remind_delay = 5      # 首次延迟5秒
        self.repeat_remind_interval = 10  # 重复间隔10秒

        # 状态记录
        self.reminded_postures = set()
        self.posture_start_time = {}
        self.last_remind_time = {}

        # 初始化音频
        if self.sound_enabled:
            try:
                pygame.mixer.init()
                self.sound = pygame.mixer.Sound(self.audio_file)
            except Exception as e:
                print(f"[提醒] 音频初始化失败：{e}")
                self.sound_enabled = False

    def _play_sound(self):
        if not self.sound_enabled:
            return
        try:
            if pygame.mixer.get_busy():
                pygame.mixer.stop()
            threading.Thread(target=self.sound.play, daemon=True).start()
        except:
            pass

    def notify(self, posture_result):
        current_time = time.time()

        if not posture_result:
            self._reset_all_state()
            return

        current_posture = str(posture_result).strip()
        is_good_posture = current_posture.lower() == "good posture"

        # 良好姿势 → 重置
        if is_good_posture:
            self._reset_all_state()
            return

        # ===================== 不良姿势处理 =====================
        # 第一次出现：记录开始时间
        if current_posture not in self.posture_start_time:
            self.posture_start_time[current_posture] = current_time

        # 1. 还没首次提醒 → 等待5秒
        if current_posture not in self.reminded_postures:
            duration = current_time - self.posture_start_time[current_posture]
            if duration < self.first_remind_delay:
                return  # 不满5秒，不做任何操作

            # 满5秒 → 首次提醒
            self._play_sound()
            self.reminded_postures.add(current_posture)
            self.last_remind_time[current_posture] = current_time  # 重置计时起点
            return

        # 2. 已经提醒过 → 从上次提醒时间重新计时，执行重复提醒
        else:
            time_since_last = current_time - self.last_remind_time[current_posture]
            if time_since_last >= self.repeat_remind_interval:
                self._play_sound()
                self.last_remind_time[current_posture] = current_time  # 再次重置

    def _reset_all_state(self):
        self.reminded_postures.clear()
        self.posture_start_time.clear()
        self.last_remind_time.clear()


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
