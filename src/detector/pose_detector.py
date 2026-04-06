import cv2
import mediapipe as mp
from src.utils.pose_utils import all_detection

class PoseDetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils
    
    def process_frame(self, frame):
        # 水平翻转帧，实现镜像效果
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        keypoints = self.pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        lm = keypoints.pose_landmarks
        lmPose = self.mp_pose.PoseLandmark
        results = "No person detected"
        
        if lm is not None:
            # 歪头监控
            left_ear_x = int(lm.landmark[lmPose.LEFT_EAR].x * w)    # left ear（7点）x 坐标
            left_ear_y = int(lm.landmark[lmPose.LEFT_EAR].y * h)    # 左耳（7点）y 坐标
            right_ear_x = int(lm.landmark[lmPose.RIGHT_EAR].x * w)  # 右耳（8点）x 坐标
            right_ear_y = int(lm.landmark[lmPose.RIGHT_EAR].y * h)  # 右耳（8点）y 坐标

            # 低头监控
            left_mouth_x = int(lm.landmark[lmPose.MOUTH_LEFT].x * w)    # 左嘴角（9点）x 坐标
            left_mouth_y = int(lm.landmark[lmPose.MOUTH_LEFT].y * h)    # 左嘴角（9点）y 坐标
            left_shoulder_x = int(lm.landmark[lmPose.LEFT_SHOULDER].x * w)    # 左肩膀（11点）x 坐标
            left_shoulder_y = int(lm.landmark[lmPose.LEFT_SHOULDER].y * h)    # 左肩膀（11点）y 坐标

            # 侧脸监控
            left_eye_inner_x = int(lm.landmark[lmPose.LEFT_EYE_INNER].x * w)    # 左眼内（1点）x 坐标
            left_eye_inner_y = int(lm.landmark[lmPose.LEFT_EYE_INNER].y * h)    # 左眼内（1点）y 坐标
            right_eye_inner_x = int(lm.landmark[lmPose.RIGHT_EYE_INNER].x * w)  # 右眼内（4点）x 坐标
            right_eye_inner_y = int(lm.landmark[lmPose.RIGHT_EYE_INNER].y * h)  # 右眼内（4点）y 坐标

            # 高低肩监控
            right_shoulder_x = int(lm.landmark[lmPose.RIGHT_SHOULDER].x * w)  # 右肩膀（12点）x 坐标
            right_shoulder_y = int(lm.landmark[lmPose.RIGHT_SHOULDER].y * h)  # 右肩膀（12点）y 坐标

            # 撑桌监控
            right_mouth_x = int(lm.landmark[lmPose.MOUTH_RIGHT].x * w)  # 左嘴角（10点）x 坐标
            right_mouth_y = int(lm.landmark[lmPose.MOUTH_RIGHT].y * h)  # 左嘴角（10点）y 坐标

            # 仰头监控
            nose_x = int(lm.landmark[lmPose.NOSE].x * w)    # 鼻子（0点）x 坐标
            nose_y = int(lm.landmark[lmPose.NOSE].y * h)    # 鼻子（0点）y 坐标

            # 趴桌监控
            left_shoulder_x_norm = lm.landmark[lmPose.LEFT_SHOULDER].x  # 左肩膀（11点）x 坐标-归一化
            left_shoulder_y_norm = lm.landmark[lmPose.LEFT_SHOULDER].y  # 左肩膀（11点）y 坐标-归一化
            right_shoulder_x_norm = lm.landmark[lmPose.RIGHT_SHOULDER].x  # 右肩膀（12点）x 坐标-归一化
            right_shoulder_y_norm = lm.landmark[lmPose.RIGHT_SHOULDER].y  # 右肩膀（12点）y 坐标-归一化

            results = all_detection(nose_x, nose_y,
                          left_eye_inner_x, left_eye_inner_y,
                          right_eye_inner_x, right_eye_inner_y,
                          left_ear_x, left_ear_y,
                          right_ear_x, right_ear_y,
                          left_mouth_x, left_mouth_y,
                          right_mouth_x, right_mouth_y,
                          left_shoulder_x, left_shoulder_y,
                          right_shoulder_x, right_shoulder_y,
                          left_shoulder_x_norm, left_shoulder_y_norm,
                          right_shoulder_x_norm, right_shoulder_y_norm)
        
        # 绘制关键点
        self.mp_drawing.draw_landmarks(image, keypoints.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
        
        # 显示坐姿状态
        if results == "Good posture":
            # 使用英文显示，避免中文编码问题
            cv2.putText(image, f"Posture: {results}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(image, f"Posture: {results}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            # 添加警告信息
            if results != "No person detected":
                cv2.putText(image, "Please adjust your posture!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        
        return image, results
    
    def close(self):
        self.pose.close()
