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


if __name__ == "__main__":
    main()
