import json
import os
from dataclasses import dataclass, asdict
from typing import Optional

# 配置文件存储路径
CONFIG_FILE = os.path.join(os.path.dirname(__file__), "app_config.json")


@dataclass
class AppSettings:
    """应用配置类（对应系统设置弹窗的所有参数）"""
    # 提醒设置
    enable_sound: bool = True
    alert_volume: float = 0.8

    # 可视化设置
    show_landmarks: bool = True
    show_lines: bool = True

    # 摄像头配置
    camera_index: int = 0

    # UI配置
    window_title: str = "智能坐姿识别系统 V1.0"
    window_width: int = 1200
    window_height: int = 800
    enable_fullscreen: bool = False

    # 提醒阈值
    posture_threshold: int = 5  # 不良坐姿持续N秒后提醒

    @classmethod
    def load_from_file(cls) -> "AppSettings":
        """从JSON文件加载配置"""
        try:
            if os.path.exists(CONFIG_FILE):
                with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # 兼容旧配置（避免新增字段报错）
                return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})
        except Exception as e:
            print(f"加载配置失败: {e}")
        # 加载失败则返回默认配置
        return cls()

    def save_to_file(self) -> None:
        """保存配置到JSON文件"""
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(CONFIG_FILE), exist_ok=True)
            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(asdict(self), f, ensure_ascii=False, indent=4)
        except Exception as e:
            print(f"保存配置失败: {e}")


# 全局配置实例
settings = AppSettings.load_from_file()
