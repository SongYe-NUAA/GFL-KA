import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

dist = np.load(r'D:\project\WTGFL\mmdetection-main\kresult\img_532490\best_distribution.npy')  # shape: (4, reg_max+1)

# 计算每个边的分布峰度
kurtosis_values = []
for i in range(4):
    # 计算峰度 (Fisher's definition: normal distribution has kurtosis=0)
    k = stats.kurtosis(dist[i], fisher=True)
    kurtosis_values.append(k)
    print(f"Edge {['left', 'top', 'right', 'bottom'][i]} kurtosis: {k:.4f}")

plt.figure(figsize=(10, 8))
for i, name in enumerate(['left', 'top', 'right', 'bottom']):
    plt.subplot(2, 2, i+1)
    # 绘制柱状图
    bars = plt.bar(np.arange(dist.shape[1]), dist[i], alpha=0.7)
    
    # 在柱子顶部添加点并连接
    x_values = np.arange(dist.shape[1])
    y_values = dist[i]
    
    # 添加点和折线
    plt.plot(x_values, y_values, 'o-', color='red', linewidth=1.5, markersize=4)
    
    # 设置纵坐标上限为0.5
    plt.ylim(0, 0.7)
    
    # 设置横坐标刻度为整数
    plt.xticks(np.arange(0, dist.shape[1], 1))
    
    plt.title(f"{name} (kurtosis: {kurtosis_values[i]:.4f})")
    
plt.tight_layout()
plt.show()