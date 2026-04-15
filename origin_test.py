import cv2
import os
import sys
import mediapipe as mp
from origin_utils import all_detection

# 确保项目根目录在Python路径中
sys.path.append(os.path.abspath('.'))

def process_images_in_folder(folder_path):
    """处理指定文件夹中的所有图片"""
    # 初始化MediaPipe姿态检测
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils
    
    try:
        # 检查文件夹是否存在
        if not os.path.exists(folder_path):
            print(f"错误: 文件夹 {folder_path} 不存在")
            return
        
        # 获取文件夹中的所有文件
        files = os.listdir(folder_path)
        
        # 筛选出图片文件
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
        image_files = [f for f in files if any(f.lower().endswith(ext) for ext in image_extensions)]
        
        if not image_files:
            print(f"文件夹 {folder_path} 中没有图片文件")
            return
        
        print(f"发现 {len(image_files)} 张图片，开始处理...\n")
        
        # 初始化统计字典
        posture_count = {}
        
        # 处理每张图片
        for image_file in image_files:
            image_path = os.path.join(folder_path, image_file)
            print(f"处理图片: {image_file}")
            print("-" * 50)
            
            # 读取图片
            frame = cv2.imread(image_path)
            if frame is None:
                print(f"无法读取图片: {image_file}")
                print("-" * 50)
                continue
            
            # 处理图片
            h, w = frame.shape[:2]
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            keypoints = pose.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            lm = keypoints.pose_landmarks
            lmPose = mp_pose.PoseLandmark
            
            # 检查是否检测到人
            if lm is None:
                results = "No person detected"
                person_detected = False
            else:
                # 提取所有需要的坐标
                left_ear_x = int(lm.landmark[lmPose.LEFT_EAR].x * w)
                left_ear_y = int(lm.landmark[lmPose.LEFT_EAR].y * h)
                right_ear_x = int(lm.landmark[lmPose.RIGHT_EAR].x * w)
                right_ear_y = int(lm.landmark[lmPose.RIGHT_EAR].y * h)
                left_mouth_x = int(lm.landmark[lmPose.MOUTH_LEFT].x * w)
                left_mouth_y = int(lm.landmark[lmPose.MOUTH_LEFT].y * h)
                left_shoulder_x = int(lm.landmark[lmPose.LEFT_SHOULDER].x * w)
                left_shoulder_y = int(lm.landmark[lmPose.LEFT_SHOULDER].y * h)
                left_eye_inner_x = int(lm.landmark[lmPose.LEFT_EYE_INNER].x * w)
                left_eye_inner_y = int(lm.landmark[lmPose.LEFT_EYE_INNER].y * h)
                right_eye_inner_x = int(lm.landmark[lmPose.RIGHT_EYE_INNER].x * w)
                right_eye_inner_y = int(lm.landmark[lmPose.RIGHT_EYE_INNER].y * h)
                right_shoulder_x = int(lm.landmark[lmPose.RIGHT_SHOULDER].x * w)
                right_shoulder_y = int(lm.landmark[lmPose.RIGHT_SHOULDER].y * h)
                right_mouth_x = int(lm.landmark[lmPose.MOUTH_RIGHT].x * w)
                right_mouth_y = int(lm.landmark[lmPose.MOUTH_RIGHT].y * h)
                nose_x = int(lm.landmark[lmPose.NOSE].x * w)
                nose_y = int(lm.landmark[lmPose.NOSE].y * h)
                left_shoulder_x_norm = lm.landmark[lmPose.LEFT_SHOULDER].x
                left_shoulder_y_norm = lm.landmark[lmPose.LEFT_SHOULDER].y
                right_shoulder_x_norm = lm.landmark[lmPose.RIGHT_SHOULDER].x
                right_shoulder_y_norm = lm.landmark[lmPose.RIGHT_SHOULDER].y
                
                # 调用all_detection函数
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
                person_detected = True
            
            # 输出结果
            print(f"检测结果: {results}")
            print(f"是否检测到人: {person_detected}")
            print("-" * 50)
            print()
            
            # 更新统计
            if results not in posture_count:
                posture_count[results] = 0
            posture_count[results] += 1
        
        # 输出统计结果
        print("=" * 60)
        print("检测结果统计:")
        print("=" * 60)
        for posture, count in posture_count.items():
            print(f"{posture}: {count} 张")
        print(f"总计: {sum(posture_count.values())} 张图片")
        print("=" * 60)
            
    finally:
        # 关闭姿态检测器
        pose.close()

if __name__ == "__main__":
    # 处理picture文件夹中的图片
    process_images_in_folder('picture/6_Tilt')
