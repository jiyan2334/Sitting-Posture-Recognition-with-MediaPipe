import time
import json
import os
from datetime import datetime


class Tracking:
    """时间统计和日志生成器"""

    def __init__(self, data_dir="data"):
        """初始化时间统计和日志生成器"""
        # 获取项目根目录的绝对路径
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        # 构建data目录的绝对路径
        self.data_dir = os.path.join(self.base_dir, "data")
        self.session_start_time = None  # 会话开始时间
        self.session_end_time = None  # 会话结束时间
        self.current_posture = None  # 当前姿势
        self.current_posture_start_time = None  # 当前姿势开始时间
        self.current_posture_timestamp = None  # 当前姿势开始时间戳
        self.session_data = []  # 记录会话数据

        # --wsy sy--
        # 暂停状态跟踪
        self.is_paused = False  # 是否处于暂停状态
        self.pause_start_time = None  # 暂停的开始时间
        self.pause_start_timestamp = None  # 暂停的开始时间戳
        self.pause_duration = 0  # 暂停的总持续时间
        # --wsy sy--

        # 确保数据目录存在
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

    def update_posture(self, posture):
        """更新坐姿状态，计算持续时间"""
        current_time = time.time()
        timestamp = datetime.now().isoformat()

        # 如果是第一次调用（开始检测），记录会话开始时间
        if self.session_start_time is None:
            self.session_start_time = datetime.now()
            self.current_posture = posture
            self.current_posture_start_time = current_time
            self.current_posture_timestamp = timestamp
            return 0

        # 如果姿势没有变化，直接返回当前姿势的持续时间
        if posture == self.current_posture:
            return current_time - self.current_posture_start_time

        # 姿势发生了变化，记录当前姿势
        duration = current_time - self.current_posture_start_time
        if duration > 0:
            self._record_posture_data(self.current_posture, duration, self.current_posture_timestamp)

        # 更新当前姿势和时间戳
        self.current_posture = posture
        self.current_posture_start_time = current_time
        self.current_posture_timestamp = timestamp

        return 0

    def _record_posture_data(self, posture, duration, timestamp):
        """记录姿势数据"""
        # 记录所有持续时间大于0的姿势
        if duration > 0:
            data = {
                "posture": posture,
                "duration": duration,
                "timestamp": timestamp
            }
            self.session_data.append(data)

    def get_duration(self, posture):
        """获取指定坐姿的持续时间"""
        if posture == self.current_posture:
            return time.time() - self.current_posture_start_time
        return 0

    def reset_duration(self, posture):
        """重置指定坐姿的持续时间"""
        if posture == self.current_posture:
            current_time = time.time()
            duration = current_time - self.current_posture_start_time
            if duration > 0:
                self._record_posture_data(posture, duration, self.current_posture_timestamp)
            # 重置当前姿势
            timestamp = datetime.now().isoformat()
            self.current_posture_start_time = current_time
            self.current_posture_timestamp = timestamp

    # --wsy sy--
    def pause(self):
        """暂停检测，记录当前姿势并开始暂停计时"""
        if not self.is_paused:
            current_time = time.time()
            pause_timestamp = datetime.now().isoformat()
            # 记录当前姿势（只记录持续时间大于等于1秒的）
            if self.current_posture:
                duration = current_time - self.current_posture_start_time
                if duration >= 1:
                    self._record_posture_data(self.current_posture, duration, self.current_posture_timestamp)

            # 开始暂停计时，使用当前时间作为暂停的开始时间
            self.is_paused = True
            self.pause_start_time = current_time
            self.pause_duration = 0
            self.pause_start_timestamp = pause_timestamp

    def resume(self):
        """恢复检测，记录暂停状态并开始新的姿势计时"""
        if self.is_paused:
            current_time = time.time()
            # 计算暂停持续时间
            self.pause_duration = current_time - self.pause_start_time
            # 记录暂停状态（无论持续时间多长）
            if self.pause_duration > 0:
                data = {
                    "posture": "paused",
                    "duration": self.pause_duration,
                    "timestamp": getattr(self, 'pause_start_timestamp', datetime.now().isoformat())
                }
                self.session_data.append(data)
            # 重置暂停状态
            self.is_paused = False
            self.pause_start_time = None
            self.pause_duration = 0
            self.pause_start_timestamp = None
            # 重置姿势时间，确保恢复后使用新的开始时间
            timestamp = datetime.now().isoformat()
            self.current_posture_start_time = current_time
            self.current_posture_timestamp = timestamp

    # --wsy sy--

    def save_session(self):
        """保存当前会话数据"""
        # 记录会话结束时间
        self.session_end_time = datetime.now()
        current_time = time.time()

        # 记录最后一个姿势的数据（只记录持续时间大于等于1秒的）
        if self.current_posture:
            duration = current_time - self.current_posture_start_time
            if duration >= 1:
                self._record_posture_data(self.current_posture, duration, self.current_posture_timestamp)

        # --wsy sy--
        # 记录暂停状态（如果当前处于暂停状态）
        if self.is_paused and self.pause_start_time:
            self.pause_duration = current_time - self.pause_start_time
            if self.pause_duration > 0:
                data = {
                    "posture": "paused",
                    "duration": self.pause_duration,
                    "timestamp": getattr(self, 'pause_start_timestamp', datetime.now().isoformat())
                }
                self.session_data.append(data)
        # --wsy sy--

        # 计算总持续时间（使用开始时间和结束时间的差值）
        if self.session_start_time and self.session_end_time:
            total_duration = (self.session_end_time - self.session_start_time).total_seconds()
        else:
            # 如果没有开始或结束时间，使用各姿势持续时间之和
            total_duration = sum(item["duration"] for item in self.session_data)

        session_data = {
            "start_time": self.session_start_time.isoformat() if self.session_start_time else datetime.now().isoformat(),
            "end_time": self.session_end_time.isoformat(),
            "total_duration": total_duration,
            "posture_data": self.session_data
        }

        # 如果没有开始时间（可能是没有检测数据），使用当前时间
        if self.session_start_time is None:
            self.session_start_time = datetime.now()
        timestamp = self.session_start_time.strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.data_dir, f"session_{timestamp}.json")
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(session_data, f, ensure_ascii=False, indent=2)

        # 清空会话数据
        self.session_data = []
        self.session_start_time = None  # 重置会话开始时间
        self.session_end_time = None  # 重置会话结束时间
        self.current_posture = None
        self.start_time = None
        self.start_timestamp = None
        self.middle_time = None
        self.middle_timestamp = None

        # 更新年度统计数据
        stats_manager = YearlyStatsManager(self.data_dir)
        stats_manager.update_yearly_stats()

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


class YearlyStatsManager:
    """
    年度统计管理器
    功能：根据年份生成统计文件，读取日志数据，按月/日排序存储。
    """

    # 定义所有需要统计的姿势类型（与 detector 输出保持一致）
    POSTURE_TYPES = ["Good posture", "Looking down", "Looking up", "Uneven shoulders", "Left tilt", "Right tilt",
                     "Leaning on desk", "No person detected"]

    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        # 确保数据目录存在
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

    def get_yearly_filename(self, year=None):
        """
        根据年份生成统计文件名
        格式：year_2024_stats.json
        """
        target_year = year or datetime.now().year
        return os.path.join(self.data_dir, f"year_{target_year}_stats.json")

    def get_log_files_for_year(self, target_year):
        """
        扫描 data 目录，获取指定年份的所有日志文件（按文件名排序，即按时间排序）
        假设日志文件名格式为 session_YYYYMMDD_HHMMSS.json
        """
        log_files = []
        if not os.path.exists(self.data_dir):
            return log_files

        for filename in os.listdir(self.data_dir):
            if filename.startswith("session_") and filename.endswith(".json"):
                # 提取文件名中的日期部分 (YYYYMMDD)
                try:
                    # 文件名格式：session_20240101_120000.json
                    # 提取 20240101
                    date_str = filename.split('_')[1]
                    file_year = int(date_str[:4])
                    if file_year == target_year:
                        # 存储 (文件名, 完整路径) 以便后续处理
                        log_files.append((filename, os.path.join(self.data_dir, filename)))
                except (IndexError, ValueError) as e:
                    # 跳过不符合格式的文件
                    continue

        # 按文件名排序，即按时间从早到晚排序
        log_files.sort(key=lambda x: x[0])
        return log_files

    def parse_log_file(self, filepath):
        """
        读取单个日志文件，提取总时长和各姿势时长
        如果姿势未出现，默认为 0
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 初始化结果，所有姿势默认为 0
            posture_durations = {pose: 0.0 for pose in self.POSTURE_TYPES}

            # 累加日志中的姿势数据
            logs = data.get("posture_data", [])
            for log in logs:
                p_type = log.get("posture")  # 注意：Tracking 中保存的是 "posture"
                p_duration = log.get("duration", 0)
                if p_type in posture_durations:
                    posture_durations[p_type] += p_duration

            # 获取总时长（如果日志中没有，就计算所有姿势之和）
            total_duration = data.get("total_duration", sum(posture_durations.values()))

            return {
                "total_duration": total_duration,
                "posture_durations": posture_durations
            }
        except Exception as e:
            print(f"读取日志文件 {filepath} 出错: {e}")
            # 出错时返回空数据，避免中断
            return {
                "total_duration": 0.0,
                "posture_durations": {pose: 0.0 for pose in self.POSTURE_TYPES}
            }

    def update_yearly_stats(self):
        """
        核心方法：更新年度统计文件
        1. 确定当前年份
        2. 扫描该年份的所有日志文件
        3. 构建按月/日排序的结构
        4. 保存为 JSON
        """
        current_year = datetime.now().year
        stats_file = self.get_yearly_filename(current_year)

        # 初始化年度统计数据结构
        yearly_data = {
            "year": current_year,
            "generated_at": datetime.now().isoformat(),
            "months": {}
        }

        # 获取该年份所有日志文件 (已排序)
        log_files = self.get_log_files_for_year(current_year)

        # 遍历所有日志文件
        for filename, filepath in log_files:
            # 解析文件名获取日期 (session_20240101_120000.json)
            date_part = filename.split('_')[1]  # "20240101"
            month = int(date_part[4:6])  # 01
            day = int(date_part[6:8])  # 01

            # 读取日志数据
            log_data = self.parse_log_file(filepath)

            # 构建月份键 (如 "01", "12")
            month_key = f"{month:02d}"
            day_key = f"{day:02d}"

            # 初始化月份结构
            if month_key not in yearly_data["months"]:
                yearly_data["months"][month_key] = {
                    "month": month,
                    "days": {}
                }

            # 初始化日期结构
            if day_key not in yearly_data["months"][month_key]["days"]:
                yearly_data["months"][month_key]["days"][day_key] = {
                    "date": f"{current_year}-{month_key}-{day_key}",
                    "sessions": []  # 存储当天的所有会话记录
                }

            # 将该文件的统计数据加入当天
            yearly_data["months"][month_key]["days"][day_key]["sessions"].append({
                "log_filename": filename,  # ① 写清楚文件名
                "total_duration": log_data["total_duration"],
                "postures": log_data["posture_durations"]  # ② 包含所有姿势，未出现的已为0
            })

        # 写入文件
        try:
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(yearly_data, f, ensure_ascii=False, indent=2)
            print(f"年度统计文件已更新: {stats_file}")
            return True
        except Exception as e:
            print(f"更新年度统计文件失败: {e}")
            return False

    # --- 辅助方法：供外部调用 ---
    def get_stats(self, year=None):
        """
        读取指定年份的统计文件
        """
        target_file = self.get_yearly_filename(year)
        if os.path.exists(target_file):
            with open(target_file, 'r', encoding='utf-8') as f:
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
