import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ==========================================
# 1. 设置中文字体，防止中文显示为方块
# ==========================================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Songti SC', 'Arial Unicode MS'] # 兼容Win/Mac
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 2. 准备数据
# ==========================================
# 文件地址
df = pd.read_csv(r"C:\Users\Shen Tian\Desktop\data.csv", encoding='gbk')

# ==========================================
# 3. 初始化图表并设置美观样式
# ==========================================
fig, ax = plt.subplots(figsize=(16, 8), dpi=150) # 高分辨率，宽幅图表

# 绘制折线图：调整线宽、标记点大小和颜色
color_cn = '#348ABD' # 沉稳蓝
color_us = '#F27E33' # 活力橙

x_perfect = range(1, 25) # 实际绘图的X坐标仍然是 1 到 24

ax.plot(x_perfect, df['中国评论数量'], marker='o', markersize=6, linewidth=2.5, 
        color=color_cn, label='中国', zorder=3)
ax.plot(x_perfect, df['美国评论数量'], marker='s', markersize=6, linewidth=2.5, 
        color=color_us, label='美国', zorder=3)

# ==========================================
# 4. 自动寻找最大值并添加千分位数据标签
# ==========================================
max_cn_val = df['中国评论数量'].max()
max_cn_x = df['中国评论数量'].idxmax() + 1 

max_us_val = df['美国评论数量'].max()
max_us_x = df['美国评论数量'].idxmax() + 1 

ax.annotate(f"{max_cn_val:,}", 
            xy=(max_cn_x, max_cn_val), 
            xytext=(0, 8), 
            textcoords="offset points", 
            ha='center', va='bottom', 
            fontsize=16, fontweight='bold', color=color_cn)

ax.annotate(f"{max_us_val:,}", 
            xy=(max_us_x, max_us_val), 
            xytext=(0, 8), 
            textcoords="offset points", 
            ha='center', va='bottom', 
            fontsize=16, fontweight='bold', color=color_us)

# ==========================================
# 5. 设置标题、轴标签和图例
# ==========================================
ax.set_title('中美两国评论数量月度变化趋势对比', fontsize=18, pad=20, fontweight='bold')
ax.set_ylabel('评论数量', fontsize=14)
ax.legend(fontsize=12, loc='upper right', framealpha=0.9, edgecolor='gray')

# ==========================================
# 6. 精细化坐标轴与网格 (★ 本次修改重点)
# ==========================================
# 设置X轴的刻度位置为 1 到 24
ax.set_xticks(x_perfect)

# ★ 新增：强制修改X轴的显示标签为两个 1-12 的循环
# list(range(1, 13)) 会生成[1, 2...12]
# * 2 会将其复制一次，变成[1, 2...12, 1, 2...12]
x_labels = list(range(1, 13)) * 2
ax.set_xticklabels(x_labels)

ax.set_xlim(0, 25) 

# 动态获取全局最大值，留出顶部空间
global_max = max(max_cn_val, max_us_val)
ax.set_ylim(-2000, global_max * 1.15)

# 添加浅色网格线
ax.grid(True, linestyle='-', alpha=0.3, zorder=1)

# ==========================================
# 7. 添加半年分割线
# ==========================================
split_points = [6.5, 12.5, 18.5]
for x in split_points:
    ax.axvline(x=x, color='gray', linestyle='--', linewidth=1.5, alpha=0.6, zorder=2)

# ==========================================
# 8. 添加居中的阶段标签 (H1 2024 等)
# ==========================================
stages =[
    {'label': 'H1 2024', 'x_pos': 3.5},
    {'label': 'H2 2024', 'x_pos': 9.5},
    {'label': 'H1 2025', 'x_pos': 15.5},
    {'label': 'H2 2025', 'x_pos': 21.5}
]

transform = ax.get_xaxis_transform() 

for stage in stages:
    ax.text(stage['x_pos'], -0.06, stage['label'], 
            ha='center', va='top', transform=transform, 
            fontsize=13, fontweight='bold', color='#333333', 
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))

ax.text(12.5, -0.12, '时间周期 (月份)', ha='center', va='top', transform=transform, fontsize=14)

# ==========================================
# 9. 调整边距并显示/保存
# ==========================================
plt.subplots_adjust(bottom=0.15) 

plt.savefig('trend_chart.png', bbox_inches='tight') 
plt.show()