import math as m

from matplotlib.mlab import angle_spectrum


# 度量函数
def findAngle(x1, y1, x2, y2):
    # 防止除零错误
    if y1 == 0:
        return 90  # 默认返回90度

    # 计算两点之间的距离
    distance = m.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    if distance == 0:
        return 90  # 两点重合时返回90度

    # 计算角度
    theta = m.acos((y2 - y1) * (-y1) / (distance * y1))
    degree = int(180 / m.pi) * theta
    return degree


# 通过鼻子与双肩连线计算画面的水平偏转角度
def calculate_image_angle(coordinates):
    # 提取鼻子和双肩的坐标
    nose_x, nose_y = coordinates['nose']
    left_shoulder_x, left_shoulder_y = coordinates['left_shoulder']
    right_shoulder_x, right_shoulder_y = coordinates['right_shoulder']

    # 计算双肩连线的中点
    shoulder_mid_x = (left_shoulder_x + right_shoulder_x) / 2
    shoulder_mid_y = (left_shoulder_y + right_shoulder_y) / 2

    # 计算鼻子与双肩中点的水平距离
    horizontal_distance = nose_x - shoulder_mid_x

    # 计算双肩之间的距离
    shoulder_distance = m.sqrt((right_shoulder_x - left_shoulder_x) ** 2 + (right_shoulder_y - left_shoulder_y) ** 2)

    # 避免除以零
    if shoulder_distance == 0:
        return 0

    # 计算偏转角度（弧度）
    # 使用反正切函数计算角度
    angle_rad = m.atan2(horizontal_distance, shoulder_distance / 2)

    # 转换为角度
    angle_deg = int(180 / m.pi * angle_rad)

    return angle_deg


# 计算真实角度
def calculate_3d_angle(point1, point2, cos_angle):
    """
    计算3D歪头角度
    参数：
        point1: 第一个点的坐标 (x, y)
        point2: 第二个点的坐标 (x, y)
        cos_angle: 偏转角的余弦值
    返回：
        歪头角度（度）
    """
    x1, y1 = point1
    x2, y2 = point2
    y_diff = y1 - y2
    x_diff = x1 - x2

    if y_diff == 0:
        return 90
    else:
        denominator = y_diff * cos_angle
        if denominator == 0:
            return 90
        else:
            return int(m.degrees(m.atan(x_diff / denominator)))


def all_detection(coordinates):  # 接收坐标字典作为参数
    tmp = 'Good posture' # -cwy
    # 计算水平偏转角度
    angle = calculate_image_angle(coordinates)
    # 计算cos(calculate_image_angle)
    cos_angle = m.cos(m.radians(angle))

    # 计算歪头角度
    waitou_inclination = calculate_3d_angle(coordinates['left_ear'], coordinates['right_ear'], cos_angle)
    # 计算低头角度：使用左嘴角和左肩、右嘴角和右肩，取平均值
    left_ditou = findAngle(*coordinates['left_mouth'], *coordinates['left_shoulder'])
    right_ditou = findAngle(*coordinates['right_mouth'], *coordinates['right_shoulder'])
    ditou_inclination = (left_ditou + right_ditou) / 2
    # 计算肩膀角度
    gaodijian_inclination = calculate_3d_angle(coordinates['left_shoulder'], coordinates['right_shoulder'], cos_angle)
    # 计算仰头角度
    left_yangtou = findAngle(*coordinates['nose'], *coordinates['left_ear'])
    right_yangtou = findAngle(*coordinates['nose'], *coordinates['right_ear'])
    yangtou_inclination = (left_yangtou + right_yangtou) / 2
    # 提取归一化坐标用于后续判断
    left_shoulder_y_norm = coordinates['left_shoulder_norm'][1]
    right_shoulder_y_norm = coordinates['right_shoulder_norm'][1]
    left_mouth_y = coordinates['left_mouth'][1]
    right_mouth_y = coordinates['right_mouth'][1]
    left_shoulder_y = coordinates['left_shoulder'][1]
    right_shoulder_y = coordinates['right_shoulder'][1]
    if (left_shoulder_y_norm + right_shoulder_y_norm) > 1.7:
        tmp = 'Leaning on desk'
    elif ditou_inclination < 115:
        tmp = 'Looking down'
    elif yangtou_inclination > 95:
        tmp = 'Looking up'
    elif abs(gaodijian_inclination) < 80:
        tmp = 'Uneven shoulders'
    elif abs(gaodijian_inclination) < 85 and abs(angle) < 30:
        tmp = 'Uneven shoulders'
    elif waitou_inclination < 0 and waitou_inclination > -75:
        tmp = 'Left tilt'
    elif waitou_inclination > 0 and waitou_inclination < 75:
        tmp = 'Right tilt'
    elif (left_mouth_y or right_mouth_y) > (left_shoulder_y or right_shoulder_y):
        tmp = 'Leaning on desk'
    else:
        tmp = 'Good posture'
    # 返回角度 --cwy、zyx
    angles = {
        "shoulder": round(gaodijian_inclination, 1),  # 肩膀倾斜角度
        "frame_angle": round(angle, 1)  # 画面偏转角度
    }

    return tmp,angles


if __name__ == "__main__":
    import cv2
    import sys
    import os

    # 添加项目根目录到Python路径
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
    from src.detector.pose_detector import PoseDetector
    import time

    # 初始化摄像头
    cap = cv2.VideoCapture(0)

    # 创建姿态检测器实例
    detector = PoseDetector()

    # 用于计时
    last_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("无法读取摄像头画面")
                break

            # 处理每一帧
            image, results = detector.process_frame(frame)

            # 提取坐标
            image_processed, h, w = detector.preprocess_frame(frame)
            keypoints, _ = detector.extract_keypoints(image_processed)
            coordinates = detector.extract_landmark_coordinates(keypoints, h, w)

            # 每隔1秒输出角度
            current_time = time.time()
            if current_time - last_time >= 1:
                if coordinates is not None:
                    angle = calculate_image_angle(coordinates)
                    print(f"当前偏转角度: {angle}度")
                else:
                    print("未检测到人体")
                last_time = current_time

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