import cv2
import mediapipe as mp
from src.utils.pose_utils import all_detection

class PoseDetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils
    
    def preprocess_frame(self, frame):
        # 水平翻转帧，实现镜像效果
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return image, h, w
    
    def extract_keypoints(self, image):
        keypoints = self.pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return keypoints, image
    
    def extract_landmark_coordinates(self, keypoints, h, w):
        lm = keypoints.pose_landmarks
        lmPose = self.mp_pose.PoseLandmark
        
        if lm is None:
            return None
        
        # 提取所有需要的坐标
        return {
            'nose': (int(lm.landmark[lmPose.NOSE].x * w), int(lm.landmark[lmPose.NOSE].y * h)),
            'left_eye_inner': (int(lm.landmark[lmPose.LEFT_EYE_INNER].x * w), int(lm.landmark[lmPose.LEFT_EYE_INNER].y * h)),
            'right_eye_inner': (int(lm.landmark[lmPose.RIGHT_EYE_INNER].x * w), int(lm.landmark[lmPose.RIGHT_EYE_INNER].y * h)),
            'left_ear': (int(lm.landmark[lmPose.LEFT_EAR].x * w), int(lm.landmark[lmPose.LEFT_EAR].y * h)),
            'right_ear': (int(lm.landmark[lmPose.RIGHT_EAR].x * w), int(lm.landmark[lmPose.RIGHT_EAR].y * h)),
            'left_mouth': (int(lm.landmark[lmPose.MOUTH_LEFT].x * w), int(lm.landmark[lmPose.MOUTH_LEFT].y * h)),
            'right_mouth': (int(lm.landmark[lmPose.MOUTH_RIGHT].x * w), int(lm.landmark[lmPose.MOUTH_RIGHT].y * h)),
            'left_shoulder': (int(lm.landmark[lmPose.LEFT_SHOULDER].x * w), int(lm.landmark[lmPose.LEFT_SHOULDER].y * h)),
            'right_shoulder': (int(lm.landmark[lmPose.RIGHT_SHOULDER].x * w), int(lm.landmark[lmPose.RIGHT_SHOULDER].y * h)),
            'left_shoulder_norm': (lm.landmark[lmPose.LEFT_SHOULDER].x, lm.landmark[lmPose.LEFT_SHOULDER].y),
            'right_shoulder_norm': (lm.landmark[lmPose.RIGHT_SHOULDER].x, lm.landmark[lmPose.RIGHT_SHOULDER].y)
        }
    
    def visualize_results(self, image, keypoints, results):
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
        
        return image
    
    def process_frame(self, frame):
        # 图像预处理
        image, h, w = self.preprocess_frame(frame)
        
        # 提取关键点
        keypoints, image = self.extract_keypoints(image)
        
        # 提取坐标
        coordinates = self.extract_landmark_coordinates(keypoints, h, w)
        
        # 评估姿态
        if coordinates is None:
            results = "No person detected"
        else:
            results = all_detection(coordinates)
        
        # 可视化结果
        image = self.visualize_results(image, keypoints, results)
        
        return image, results
    
    def close(self):
        self.pose.close()
