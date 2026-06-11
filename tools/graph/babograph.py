import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager, colors as mcolors
from matplotlib.patches import Circle
from matplotlib.patheffects import withStroke

# 设置英文字体为 Times New Roman
english_font = font_manager.FontProperties(family='Times New Roman', weight='bold')
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.weight'] = 'bold'

# 数据集的坐标
datasets = {
    'DTU': (701, 0),
    'YAWTSD': (13470, 9351),
    'Blade30': (1302, 406),
    'WCVP': (2137, 1908),
    'WTBSDD': (3808, 8103),
    'WindBlade-30K': (5168, 30437)
}

# 图表参数
y_max = 35000
y_min = 0
plt.figure(figsize=(11, 7.5))
# 使用更丰富的colormap
base_colors = plt.cm.tab20(np.linspace(0, 1, len(datasets)-1))

# 分配颜色，WindBlade-30K为红色
dataset_colors = {}
i = 0
for name in datasets.keys():
    if name == 'WindBlade-30K':
        dataset_colors[name] = '#d62728'  # 鲜明红色
    else:
        dataset_colors[name] = base_colors[i]
        i += 1

ax = plt.gca()

# 绘制气泡和标签
for dataset, (x, y) in datasets.items():
    color = dataset_colors[dataset]
    size = y / 5 if y > 0 else 50  # 避免0导致气泡太小
    # 阴影气泡
    ax.scatter(x, y-120, s=size*1.08, alpha=0.18, color='gray', linewidth=0, zorder=1)
    # 主气泡
    ax.scatter(x, y, s=size, alpha=0.8, color=color, edgecolor='black', linewidth=1.5, zorder=2)
    # 中心点
    ax.scatter(x, y, s=18, color='black', zorder=3)
    # 标签y坐标（上移更多）
    label_y = y + max(600, size**0.5*2)
    if label_y > y_max - 300:
        label_y = y_max - 300
    # 颜色加深
    if dataset == 'WindBlade-30K':
        deep_color = '#8b0000'
    else:
        rgb = np.array(mcolors.to_rgb(color))
        deep_color = tuple(np.clip(rgb * 0.5, 0, 1))
    # 标签描边
    txt_effect = [withStroke(linewidth=3, foreground='white')]
    # YAWTSD标签左移
    label_x = x-500 if dataset == 'YAWTSD' else x
    ax.text(label_x, label_y, dataset, fontsize=28, ha='center', va='bottom',
            fontproperties=english_font, weight='bold', color=deep_color, zorder=4,
            path_effects=txt_effect)

# 设置symlog刻度，低区间线性，1000以上对数
plt.xscale('symlog', linthresh=1000)
plt.yscale('linear')

# 设置固定的刻度单位
x_ticks = [500, 1000, 2000, 5000, 10000, 20000]
plt.xticks(x_ticks, x_ticks, weight='bold', fontsize=20)
y_ticks = range(0, 36000, 5000)
plt.yticks(y_ticks, [f'{int(y/1000)}k' for y in y_ticks], weight='bold', fontsize=20)

# 设置y轴范围
plt.ylim(y_min, y_max)
plt.xlim(500, 20000)

# 轴标签加粗
plt.xlabel('Number of images', weight='bold', fontsize=28, labelpad=12)
plt.ylabel('Number of instances', weight='bold', fontsize=28, labelpad=12)

# 添加更细腻的网格线
plt.grid(True, linestyle='--', alpha=0.25, zorder=0, linewidth=1)

# 创建自定义图例
legend_elements = [
    plt.scatter([], [], s=520, alpha=0.8, color=dataset_colors[name], edgecolor='black', linewidth=1.5, label=name)
    for name in datasets.keys()
]
plt.legend(handles=legend_elements,
          loc='upper left',
          bbox_to_anchor=(0.01, 0.99),
          scatterpoints=1,
          frameon=True,
          prop={'weight': 'bold', 'size': 22},
          ncol=1,
          borderpad=0.7,
          labelspacing=0.7)

# 去除顶部和右侧边框
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 设置背景色为白色
ax.set_facecolor('white')
plt.tight_layout()
plt.show()