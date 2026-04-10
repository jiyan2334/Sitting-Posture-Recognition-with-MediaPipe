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
            'left_eye_inner': (int(lm.landmark[lmPose.LEFT_EYE_INNER].x * w),
                               int(lm.landmark[lmPose.LEFT_EYE_INNER].y * h)),
            'right_eye_inner': (int(lm.landmark[lmPose.RIGHT_EYE_INNER].x * w),
                                int(lm.landmark[lmPose.RIGHT_EYE_INNER].y * h)),
            'left_ear': (int(lm.landmark[lmPose.LEFT_EAR].x * w), int(lm.landmark[lmPose.LEFT_EAR].y * h)),
            'right_ear': (int(lm.landmark[lmPose.RIGHT_EAR].x * w), int(lm.landmark[lmPose.RIGHT_EAR].y * h)),
            'left_mouth': (int(lm.landmark[lmPose.MOUTH_LEFT].x * w), int(lm.landmark[lmPose.MOUTH_LEFT].y * h)),
            'right_mouth': (int(lm.landmark[lmPose.MOUTH_RIGHT].x * w), int(lm.landmark[lmPose.MOUTH_RIGHT].y * h)),
            'left_shoulder': (int(lm.landmark[lmPose.LEFT_SHOULDER].x * w),
                              int(lm.landmark[lmPose.LEFT_SHOULDER].y * h)),
            'right_shoulder': (int(lm.landmark[lmPose.RIGHT_SHOULDER].x * w),
                               int(lm.landmark[lmPose.RIGHT_SHOULDER].y * h)),
            'left_shoulder_norm': (lm.landmark[lmPose.LEFT_SHOULDER].x, lm.landmark[lmPose.LEFT_SHOULDER].y),
            'right_shoulder_norm': (lm.landmark[lmPose.RIGHT_SHOULDER].x, lm.landmark[lmPose.RIGHT_SHOULDER].y)
        }

    def visualize_results(self, image, keypoints, results):
        # 绘制关键点
        self.mp_drawing.draw_landmarks(image, keypoints.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)




        return image

    def process_frame(self, frame):
        # 图像预处理
        image, h, w = self.preprocess_frame(frame)

        # 提取关键点
        keypoints, image = self.extract_keypoints(image)

        # 提取坐标
        coordinates = self.extract_landmark_coordinates(keypoints, h, w)

        # 评估姿态 增加角度和是否检测到人的信号量--cwy、zyx
        if coordinates is None:
            results = "No person detected"
            person_detected = False
            angles = {"shoulder": 0.0,"frame_angle": 0.0}
        else:
            results, angles = all_detection(coordinates)
            person_detected = True

        # 可视化结果
        image = self.visualize_results(image, keypoints, results)

        return image, results, person_detected, angles

    def close(self):
        self.pose.close()
