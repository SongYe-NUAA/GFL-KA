import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

# 设置中文字体
font_path = "C:\\Windows\\Fonts\\simsun.ttc"
chinese_font_prop = font_manager.FontProperties(fname=font_path)
english_font = font_manager.FontProperties(family='Times New Roman', weight='bold')
# 设置英文字体为 Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.weight'] = 'bold'  # 设置全局字体粗细

# 数据集的坐标
datasets = {
    'windturbine': (2137, 1908),
    'Blade30': (1302, 406),
    'NTDTU': (13470, 9351),
    'DTU-reannotation': (7010, 796),
    'WTBSDD': (3808, 8103),
    'WindBlade-30K': (5168, 30437)
}

# 创建图表
plt.figure(figsize=(10, 7))
colors = plt.cm.tab10(range(len(datasets)))  # 使用 colormap 获取颜色
# 绘制散点图，但不添加标签
for (dataset, (x, y)), color in zip(datasets.items(), colors):
    size = y / 5  # 气泡大小
    # 绘制气泡
    scatter = plt.scatter(x, y, s=size, alpha=0.5, color=color)
    # 绘制中心点（小黑点）
    plt.scatter(x, y, s=10, color='black', zorder=3)  # zorder确保点在气泡上面
    # 将标签放在气泡中间，使用Times New Roman
    plt.text(x, y, dataset, fontsize=13, ha='center', va='center',
             fontproperties=english_font)




# 设置对数刻度
plt.xscale('log')
plt.yscale('linear')  # 改为线性刻度以确保等分

# 设置固定的刻度单位
x_ticks = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
y_ticks = range(0, 36000, 1000)  # 从0到35k，每1k一个刻度
plt.xticks(x_ticks, x_ticks, weight='bold')  # 加粗刻度标签
plt.yticks(y_ticks, [f'{int(y/1000)}k' for y in y_ticks], weight='bold')  # 加粗刻度标签

# 设置y轴范围
plt.ylim(0, 35000)  # 明确设置y轴范围为0-35k

# 轴标签加粗
plt.xlabel('Number of images', weight='bold', fontsize=18)
plt.ylabel('Number of instances', weight='bold', fontsize=18)

# 创建自定义图例
legend_elements = [plt.scatter([], [], s=100, alpha=0.5, label=name)
                  for name in datasets.keys()]
plt.legend(handles=legend_elements,
          loc='upper left',  # 图例位置
          bbox_to_anchor=(0.02, 0.98),  # 微调图例位置
          scatterpoints=1,
          frameon=True,
          prop={'weight': 'bold'})  # 图例文字加粗

# 调整布局
plt.tight_layout()
plt.show()