import matplotlib.pyplot as plt
import numpy as np

# Data
categories = [
    "corrosion-pit",
    "stain",
    "hole",
    "corrosion",
    "degumming",
    "Crack",
    "Sign",
    "Dirt",
    "repair",
    "demould",
    "lightning-arrester",
    "leaf-opex",
    "teeth",
    "painting-peel-off",
    "oil",
    "lightning-arrester-miss",
    "Swell",
]
numbers = [
    18226,
    4420,
    2064,
    1499,
    1037,
    797,
    749,
    320,
    270,
    269,
    218,
    135,
    104,
    96,
    89,
    87,
    57
]

# 设置全局样式
plt.style.use('seaborn-v0_8-paper')  # 使用更专业的样式
plt.rcParams['font.family'] = 'Arial'  # 使用 Arial 字体
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['mathtext.fontset'] = 'stix'  # 数学字体设置为 STIX
plt.rcParams['font.serif'] = ['Times New Roman']  # 设置 serif 字体为 Times New Roman
plt.rcParams['axes.linewidth'] = 1.2   # 加粗轴线

# 自定义科研配色方案
colors = [
    '#4C72B0',  # 深蓝
    '#DD8452',  # 橙色
    '#55A868',  # 绿色
    '#C44E52',  # 红色
    '#8172B3',  # 紫色
    '#937860',  # 棕色
    '#DA8BC3',  # 粉色
    '#8C8C8C',  # 灰色
    '#CCB974',  # 黄色
    '#64B5CD',  # 浅蓝
    '#4C72B0',  # 深蓝（循环）
    '#DD8452',  # 橙色
    '#55A868',  # 绿色
    '#C44E52',  # 红色
    '#8172B3',  # 紫色
    '#937860',  # 棕色
    '#DA8BC3',  # 粉色
]

# Bar Chart
plt.figure(figsize=(16, 8), dpi=300)
ax = plt.gca()

# 使用新的配色方案
bars = ax.bar(range(len(categories)), numbers, color=colors)

# 添加网格线
ax.grid(True, axis='y', linestyle='--', alpha=0.7)
ax.set_axisbelow(True)  # 将网格线置于数据后面

# 设置坐标轴标签和标题，使用更正式的字体样式
ax.set_xlabel('Categories', fontsize=14, fontweight='bold')
ax.set_ylabel('Number of Instances', fontsize=14, fontweight='bold')
ax.set_title('Category Distribution in WindBlade-30K Dataset', fontsize=16, fontweight='bold', pad=20)

# 调整x轴标签，将ha从'right'改为'center'
plt.xticks(range(len(categories)), categories, rotation=45, ha='center',fontsize=12)

# 调整布局以确保标签不会被切掉
plt.tight_layout(pad=1.2)  # 增加一些padding以防止标签被切掉

# 在柱子上方添加数值标注，使用 Times New Roman
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 50,
            f'{numbers[i]}',
            ha='center', va='bottom', 
            family='Times New Roman',
            fontsize=12)

# Donut Chart with legend
fig, ax2 = plt.subplots(figsize=(15, 8), dpi=300)

def make_autopct(values):
    def my_autopct(pct):
        total = sum(values)
        val = int(round(pct*total/100.0))
        # 只显示占比大于等于1.5%的数值
        if pct < 1.5:
            return ''
        return f'{pct:.1f}%'
    return my_autopct

# 计算总数
total = sum(numbers)

# 创建更简洁的标签文本，类别名称和百分比紧凑显示
legend_labels = [f"{cat}({num/total*100:.1f}%)" for cat, num in zip(categories, numbers)]

# 绘制饼图
wedges, texts, autotexts = ax2.pie(numbers,
                                  labels=[''] * len(categories),
                                  autopct=make_autopct(numbers),
                                  startangle=90,
                                  colors=colors,
                                  pctdistance=0.75,
                                  wedgeprops={'edgecolor': 'white',     # 正确的分割线参数
                                            'linewidth': 1.5},          # 设置分割线宽度
                                  textprops={'fontsize': 14})  # 增加饼图文字大小

# 设置饼图中心空白
centre_circle = plt.Circle((0, 0), 0.60, fc='white', edgecolor='gray', linewidth=1)
fig.gca().add_artist(centre_circle)

# 优化图例样式
ax2.legend(wedges, legend_labels,
          title="Categories",
          loc="center left",
          bbox_to_anchor=(1, 0, 0.5, 1),
          fontsize=10,
          title_fontsize=12,
          frameon=True,
          edgecolor='gray',
          alignment='center',
          prop={'family': 'Times New Roman'})  # 设置图例字体

# 设置饼图中的数字标签字体
for autotext in autotexts:
    autotext.set_fontfamily('Times New Roman')
    autotext.set_fontsize(12)  # 设置百分比标签的字体大小

ax2.set_title('Category Distribution in WindBlade-30K Dataset', fontsize=16, fontweight='bold', pad=20)

# 调整布局以适应图例
plt.tight_layout()

# 保存高质量图像
plt.savefig('distribution.png', 
            dpi=300,              # 输出图像的DPI
            bbox_inches='tight',   # 自动调整边界
            pad_inches=0.2,       # 边缘留白
            format='png')         # 使用png格式

plt.show()