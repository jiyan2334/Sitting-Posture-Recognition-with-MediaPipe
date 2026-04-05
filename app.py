import cv2
from src.detector.pose_detector import PoseDetector
from src.config.settings import Settings
from src.core.reminder.reminder import Reminder
from src.core.ui.ui_handler import UIHandler


def main():
    """主函数"""
    # 加载配置
    settings = Settings()
    
    # 初始化摄像头
    cap = cv2.VideoCapture(settings.CAMERA_INDEX)
    
    # 创建姿态检测器实例
    detector = PoseDetector()
    
    # 创建提醒器（包含持续时间监测）
    reminder = Reminder(
        enable_sound=settings.ENABLE_SOUND,
        enable_notification=settings.ENABLE_NOTIFICATION,
        data_dir=settings.DATA_DIR
    )
    
    # 创建UI处理器
    ui_handler = UIHandler(
        window_title=settings.WINDOW_TITLE,
        window_width=settings.WINDOW_WIDTH,
        window_height=settings.WINDOW_HEIGHT
    )
    
    # 创建窗口
    ui_handler.create_window()
    
    # 全屏模式
    if settings.ENABLE_FULLSCREEN:
        ui_handler.toggle_fullscreen()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("无法读取摄像头画面")
                break
            
            # 处理每一帧
            image, results = detector.process_frame(frame)
            
            # 更新坐姿持续时间并检查是否需要提醒
            duration = reminder.update_posture(results)
            if results != "Good posture" and results != "No person detected":
                if duration > settings.POSTURE_THRESHOLD:
                    reminder.remind(f"不良坐姿已持续 {int(duration)} 秒，请调整坐姿！", results)
            
            # 绘制UI
            image = ui_handler.draw_ui(image, results, duration)
            
            # 显示结果
            ui_handler.show_image(image)
            
            # 处理按键
            key = ui_handler.wait_key(1)
            if key & 0xFF == ord('q'):
                break
            elif key & 0xFF == ord('f'):
                ui_handler.toggle_fullscreen()
                
    finally:
        # 保存会话数据
        reminder.save_session()
        
        # 释放资源
        cap.release()
        detector.close()
        ui_handler.destroy_window()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
