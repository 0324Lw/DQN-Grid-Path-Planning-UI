import os
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class DataPlotter:
    """
    通用的强化学习数据记录与可视化类
    支持动态解析字典数据、保存为时间戳命名的 CSV 文件，并绘制符合 SCI 风格的平滑曲线。
    """

    def __init__(self, save_dir='./results'):
        self.save_dir = save_dir
        # 如果保存目录不存在，则自动创建
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # 初始化 SCI 绘图风格
        self._set_sci_style()

    def _set_sci_style(self):
        """配置 Matplotlib 的全局参数，使其符合学术论文 (SCI) 风格"""
        # 优先使用 Times New Roman 字体，支持中文字体回退
        plt.rcParams['font.family'] = ['Times New Roman', 'SimHei', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

        # 坐标轴线宽和刻度方向
        plt.rcParams['axes.linewidth'] = 1.5
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        plt.rcParams['xtick.top'] = True  # 顶部显示刻度
        plt.rcParams['ytick.right'] = True  # 右侧显示刻度

        # 图形输出质量
        plt.rcParams['savefig.dpi'] = 300  # 300 DPI 满足大部分期刊要求
        plt.rcParams['figure.figsize'] = (8, 6)  # 标准学术图表长宽比

    def _smooth_data(self, data, window_size):
        """
        利用滑动窗口计算移动平均值 (Moving Average)，生成平滑曲线。
        使用 Pandas 的 rolling 能够很好地处理前几个数据点不足窗口大小的问题。
        """
        return pd.Series(data).rolling(window=window_size, min_periods=1).mean().values

    def save_and_plot(self, data_dict, window_size=100):
        """
        核心方法：处理传入的数据字典，保存 CSV 并逐一绘图。
        :param data_dict: 字典格式，例如 {'Reward': [1,2,3...], 'Loss': [0.1, 0.05...]}
        :param window_size: 平滑窗口的大小
        """
        if not data_dict:
            print("警告：传入的数据字典为空，无法进行保存和绘图。")
            return

        # 1. 生成时间戳
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # 2. 保存为 CSV 文件
        # 考虑到字典中各个列表长度理论上应该一致，如果不一致 pd.DataFrame 会报错
        # 强化学习中通常每个 Episode 记录一次，所以长度应该是一致的
        df = pd.DataFrame(data_dict)
        csv_filename = f"training_data_{timestamp}.csv"
        csv_path = os.path.join(self.save_dir, csv_filename)
        df.to_csv(csv_path, index=False)
        print(f"\n所有训练数据已成功汇总保存至: {csv_path}")

        # 3. 遍历字典，为每一个键值对动态生成并保存图表
        for key, values in data_dict.items():
            plt.figure()
            episodes = np.arange(len(values))

            # 绘制原始波动曲线 (浅色、半透明)
            plt.plot(episodes, values, color='#B0C4DE', alpha=0.7, linewidth=1.0, label=f'Raw {key}')

            # 绘制平滑曲线 (深色、加粗)
            smoothed_values = self._smooth_data(values, window_size)
            plt.plot(episodes, smoothed_values, color='#000080', linewidth=2.0, label=f'Smoothed {key}')

            # 设置标签和标题
            plt.xlabel('Episodes', fontsize=14, fontweight='bold')
            plt.ylabel(key, fontsize=14, fontweight='bold')
            plt.title(f'Training Curve: {key}', fontsize=16, pad=15)

            # 设置图例 (去除图例边框以符合学术规范)
            plt.legend(loc='best', frameon=False, fontsize=12)

            # 添加浅色虚线网格辅助阅读
            plt.grid(True, linestyle='--', alpha=0.4)

            # 紧凑布局并保存
            fig_filename = f"{key}_{timestamp}.png"
            fig_path = os.path.join(self.save_dir, fig_filename)
            plt.savefig(fig_path, bbox_inches='tight')
            plt.close()
            print(f"[{key}]曲线图已保存至: {fig_path}")
