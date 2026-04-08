import time
import json
import os
from datetime import datetime

class Tracking:
    """时间统计和日志生成器"""
    
    def __init__(self, data_dir="data"):
        """初始化时间统计和日志生成器"""
        self.data_dir = data_dir
        self.posture_start_time = {}  # 记录每种坐姿的开始时间
        self.posture_duration = {}  # 记录每种坐姿的持续时间
        self.session_data = []  # 记录会话数据
        
        # 抖动时间跟踪
        self.shaking_start_time = None  # 抖动的开始时间
        self.shaking_duration = 0  # 抖动的总持续时间
        
        # 确保数据目录存在
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
    
    def update_posture(self, posture):
        """更新坐姿状态，计算持续时间"""
        current_time = time.time()
        timestamp = datetime.now().isoformat()
        
        # 如果是新的坐姿，记录开始时间
        if posture not in self.posture_start_time:
            # 记录上一个姿势的结束时间
            for p in self.posture_start_time:
                duration = self.posture_duration.get(p, 0)
                if duration > 1:
                    # 先记录之前累积的抖动时间（如果有的话）
                    if self.shaking_duration > 0:
                        data = {
                            "posture": "shaking",
                            "duration": self.shaking_duration,
                            "timestamp": datetime.now().isoformat()
                        }
                        self.session_data.append(data)
                        # 重置抖动时间
                        self.shaking_start_time = None
                        self.shaking_duration = 0
                    # 记录当前姿势
                    self._record_posture_data(p, duration)
                else:
                    # 持续时间小于1秒，视为抖动
                    if self.shaking_start_time is None:
                        self.shaking_start_time = current_time - duration
                    self.shaking_duration += duration
            
            self.posture_start_time[posture] = current_time
            self.posture_duration[posture] = 0
        else:
            # 更新持续时间
            self.posture_duration[posture] = current_time - self.posture_start_time[posture]
        
        # 重置其他坐姿的开始时间
        for p in list(self.posture_start_time.keys()):
            if p != posture:
                del self.posture_start_time[p]
                self.posture_duration[p] = 0
        
        return self.posture_duration.get(posture, 0)
    
    def _record_posture_data(self, posture, duration):
        """记录姿势数据"""
        # 只有持续时间大于1秒的姿势才被记录
        if duration > 1:
            data = {
                "posture": posture,
                "duration": duration,
                "timestamp": datetime.now().isoformat()
            }
            self.session_data.append(data)
    
    def get_duration(self, posture):
        """获取指定坐姿的持续时间"""
        return self.posture_duration.get(posture, 0)
    
    def reset_duration(self, posture):
        """重置指定坐姿的持续时间"""
        if posture in self.posture_start_time:
            self._record_posture_data(posture, self.posture_duration.get(posture, 0))
            del self.posture_start_time[posture]
        if posture in self.posture_duration:
            del self.posture_duration[posture]
    
    def save_session(self):
        """保存当前会话数据"""
        # 记录最后一个姿势的数据
        for posture in self.posture_start_time:
            duration = self.posture_duration.get(posture, 0)
            if duration > 1:
                # 先记录之前累积的抖动时间（如果有的话）
                if self.shaking_duration > 0:
                    data = {
                        "posture": "shaking",
                        "duration": self.shaking_duration,
                        "timestamp": datetime.now().isoformat()
                    }
                    self.session_data.append(data)
                    # 重置抖动时间
                    self.shaking_start_time = None
                    self.shaking_duration = 0
                # 记录当前姿势
                self._record_posture_data(posture, duration)
            else:
                # 持续时间小于1秒，视为抖动
                if self.shaking_start_time is None:
                    self.shaking_start_time = time.time() - duration
                self.shaking_duration += duration
        
        # 记录抖动时间
        if self.shaking_duration > 0:
            data = {
                "posture": "shaking",
                "duration": self.shaking_duration,
                "timestamp": datetime.now().isoformat()
            }
            self.session_data.append(data)
        
        # 计算总持续时间（所有姿势的持续时间之和）
        total_duration = sum(item["duration"] for item in self.session_data)
        
        session_data = {
            "timestamp": datetime.now().isoformat(),
            "total_duration": total_duration,
            "posture_data": self.session_data
        }
        
        filename = os.path.join(self.data_dir, f"session_{int(time.time())}.json")
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(session_data, f, ensure_ascii=False, indent=2)
        
        # 清空会话数据
        self.session_data = []
        self.posture_start_time = {}
        self.posture_duration = {}
        self.shaking_start_time = None
        self.shaking_duration = 0
        
        return filename
    
    def get_session_data(self):
        """获取当前会话数据"""
        return self.session_data
    
    def load_session(self, filename):
        """加载会话数据"""
        filepath = os.path.join(self.data_dir, filename)
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        return None




# 测试代码
if __name__ == "__main__":
    import cv2
    from src.detector.pose_detector import PoseDetector
    
    def main():
        # 初始化摄像头
        cap = cv2.VideoCapture(0)
        
        # 创建姿态检测器实例
        detector = PoseDetector()
        
        # 创建 Tracking 实例
        tracking = Tracking()
        
        # 跟踪上一个姿势
        last_posture = None
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("无法读取摄像头画面")
                    break
                
                # 处理每一帧
                image, results = detector.process_frame(frame)
                
                # 使用 tracking 记录坐姿持续时间
                duration = tracking.update_posture(results)
                
                # 在画面上显示持续时间
                if results != "No person detected":
                    cv2.putText(image, f"Duration: {int(duration)}s", (50, 150), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
                # 当姿势发生改变时，输出刚刚持续的姿势
                if last_posture and last_posture != results and results != "No person detected":
                    # 显示会话数据的当前状态
                    session_data = tracking.get_session_data()
                    if session_data:
                        print(f"姿势改变: 从 {last_posture} 到 {results}")
                        print(f"已记录的姿势数据: {session_data[-1]}")
                
                # 更新上一个姿势
                if results != "No person detected":
                    last_posture = results
                
                # 显示结果
                cv2.imshow("Sitting Posture Detection", image)
                
                # 按 'q' 键退出
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            # 保存会话数据
            tracking.save_session()
            print("会话数据已保存")
            
            # 释放资源
            cap.release()
            detector.close()
            cv2.destroyAllWindows()
    
    main()
