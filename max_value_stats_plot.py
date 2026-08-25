# import json
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.font_manager import FontProperties
#
# # 确保中文能正常显示
# font = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf")  # Windows系统默认黑体路径
#
#
#
#
# def plot_activation_distribution(json_data, layer_name, output_file=None):
#     """
#     绘制神经网络某一层激活值的分布图
#
#     参数:
#     json_data: 加载的JSON数据
#     layer_name: 要绘制的层名称
#     output_file: 输出图片文件路径，若为None则显示图片
#     """
#     try:
#         # 提取指定层的max_value_stats数据
#         layer_data = json_data[layer_name]["max_value_stats"]["mean_sample_max"]
#
#         # 转换为numpy数组以便处理
#         data = np.array(layer_data)
#
#         # 创建图形
#         plt.figure(figsize=(10, 6))
#
#         # 绘制直方图
#         plt.hist(data, bins=30, alpha=0.7, color='skyblue', edgecolor='navy', linewidth=1)
#
#         # 计算并添加统计信息
#         mean = np.mean(data)
#         median = np.median(data)
#         std = np.std(data)
#         min_val = np.min(data)
#         max_val = np.max(data)
#
#         # 添加统计信息文本
#         stats_text = f"均值: {mean:.4f}\n中位数: {median:.4f}\n标准差: {std:.4f}\n最小值: {min_val:.4f}\n最大值: {max_val:.4f}"
#         plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes,
#                  bbox=dict(facecolor='white', alpha=0.8), fontproperties=font)
#
#         # 设置图表标题和标签
#         plt.title(f"{layer_name}层max_value_stats分布", fontproperties=font, fontsize=14)
#         plt.xlabel("激活值", fontproperties=font, fontsize=12)
#         plt.ylabel("频率", fontproperties=font, fontsize=12)
#         plt.grid(axis='y', linestyle='--', alpha=0.7)
#
#         # 调整布局
#         plt.tight_layout()
#
#         # 保存或显示图片
#         if output_file:
#             plt.savefig(output_file, dpi=300, bbox_inches='tight')
#             print(f"分布图已保存至: {output_file}")
#         else:
#             plt.show()
#
#     except KeyError:
#         print(f"错误: 层 '{layer_name}' 不存在或数据格式不正确")
#     except Exception as e:
#         print(f"处理过程中发生错误: {e}")
#
#
# def main():
#     # JSON文件路径，请根据实际情况修改
#     json_file_path = r"C:\Users\LPY\Desktop\ccy实验\q-diffusion-master\cifar10_fp_expected_max\samples\2025-06-09-17-49-17\activation_stats.json"
#
#
#     try:
#         # 读取JSON文件
#         with open(json_file_path, 'r', encoding='utf-8') as f:
#             json_data = json.load(f)
#
#         # 指定要绘制的层名称
#         layer_name = "down.0.block.0.conv2"
#
#         # 绘制分布图
#         plot_activation_distribution(json_data, layer_name, "activation_distribution.png")
#
#     except FileNotFoundError:
#         print(f"错误: 找不到文件 {json_file_path}")
#     except json.JSONDecodeError:
#         print(f"错误: 无法解析JSON文件 {json_file_path}")
#     except Exception as e:
#         print(f"程序执行过程中发生错误: {e}")
#
#
# if __name__ == "__main__":
#     main()

import matplotlib.pyplot as plt
import numpy as np
import json
from scipy.stats import norm, gaussian_kde
import os
import matplotlib
import logging

# 设置日志
logging.basicConfig(level=logging.ERROR)  # 将SciPy的警告级别设为ERROR

# 设置中文字体支持
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号


def plot_activation_distributions(stats_file, output_dir):
    """生成激活分布的可视化图表"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 加载统计文件
    with open(stats_file, 'r') as f:
        stats = json.load(f)

    # 遍历每一层数据
    for layer_name, layer_data in stats.items():
        # 获取该层的统计数据
        act_stats = layer_data["activation_stats"]
        max_stats = layer_data["max_value_stats"]

        mu = act_stats["mean"]
        sigma = act_stats["std"]
        expected_max = act_stats["expected_max"]
        sample_max_values = max_stats["mean_sample_max"]

        # 创建画布
        plt.figure(figsize=(15, 10))
        plt.suptitle(f"层: {layer_name}", fontsize=16)

        # 第一个子图：理论分布
        plt.subplot(2, 1, 1)
        plot_theoretical_distribution(mu, sigma, expected_max)

        # 第二个子图：样本最大值的分布
        plt.subplot(2, 1, 2)
        plot_sample_max_distribution(sample_max_values, expected_max)

        # 保存图像
        safe_name = layer_name.replace(".", "_").replace(" ", "_")
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # 调整布局
        plt.savefig(os.path.join(output_dir, f"{safe_name}.png"))
        plt.close()


def plot_theoretical_distribution(mu, sigma, expected_max):
    """绘制理论正态分布曲线和期望最大值位置"""
    # 创建x轴数据点 (μ ± 4σ范围)
    xmin = mu - 4 * sigma
    xmax = max(mu + 10 * sigma, expected_max + 4 * sigma)
    x = np.linspace(xmin, xmax, 1000)

    # 计算正态分布PDF
    pdf = norm.pdf(x, mu, sigma)

    # 绘制PDF曲线
    plt.plot(x, pdf, 'b-', linewidth=2, label=f'N({mu:.2f}, {sigma:.2f})')  # 蓝色的理论正态分布曲线
    plt.title("激活值理论分布")
    plt.xlabel("激活值")
    plt.ylabel("概率密度")
    plt.grid(True, alpha=0.3)

    # 标记均值
    plt.axvline(mu, color='green', linestyle='--',
                label=f'均值 = {mu:.2f}')

    # 标记期望最大值
    plt.axvline(expected_max, color='red', linestyle='-',
                linewidth=2, label=f'期望最大 = {expected_max:.2f}')

    plt.legend()
    plt.xlim(xmin, xmax)


def plot_sample_max_distribution(sample_max_values, expected_max):
    """绘制样本最大值的分布（修复了核密度估计问题）"""
    # 转换数据并检查有效性
    sample_max_values = np.array(sample_max_values)
    unique_values = np.unique(sample_max_values)

    # 如果所有值几乎相同，使用更简单的图示方法
    if len(unique_values) == 1 or np.ptp(sample_max_values) < 1e-6:
        # 绘制点图而不是直方图
        counts = np.arange(len(sample_max_values))
        plt.plot(sample_max_values, counts, 'o', alpha=0.6, label='样本点值')

        plt.title("样本最大值分布（常数或近似常数）")
        plt.xlabel("样本最大值")
        plt.ylabel("样本序号")
        plt.grid(True, alpha=0.3)
    else:
        # 绘制直方图
        n, bins, patches = plt.hist(
            sample_max_values, bins=30,
            density=True, alpha=0.7,
            color='skyblue', edgecolor='black',
            label='样本最大值分布'
        )

        plt.title("样本最大值分布")
        plt.xlabel("样本最大值")
        plt.ylabel("频率密度")
        plt.grid(True, alpha=0.3)

        # 只添加分布的曲线作为参考
        mu = np.mean(sample_max_values)
        sigma = np.std(sample_max_values)
        if sigma > 1e-6:
            x = np.linspace(min(sample_max_values), max(sample_max_values), 500)
            plt.plot(x, norm.pdf(x, mu, sigma), 'purple', linewidth=2, label='正态分布参考')

    # 标记期望最大值位置
    plt.axvline(expected_max, color='red', linestyle='-',
                linewidth=2, label='期望最大值')

    # 标记样本平均值位置
    sample_mean = np.mean(sample_max_values)
    plt.axvline(sample_mean, color='green', linestyle='--',
                label=f'样本平均 = {sample_mean:.2f}')

    plt.legend()
    plt.tight_layout()


if __name__ == "__main__":
    # 配置路径
    stats_file = r"C:\Users\LPY\Desktop\ccy实验\q-diffusion-master\cifar10_fp_expected_max\samples\2025-06-09-17-49-17\activation_stats.json" # 改为你的JSON文件路径
    output_dir = "activation_plots"  # 输出图像目录

    # 生成并保存图表
    plot_activation_distributions(stats_file, output_dir)

