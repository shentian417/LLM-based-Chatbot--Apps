import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ==========================================
# 1. 设置全局字体和样式
# ==========================================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Songti SC', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 2. 准备数据 (读取真实的 CSV)
# ==========================================
# 请替换为你真实的 CSV 文件路径
df = pd.read_csv(r"C:\Users\30481\Desktop\中国与美国满意度对比分析.csv")

# ==========================================
# 3. 建立全局统一的【平衡高级配色】映射
# ==========================================
unique_topics = df['指标层'].unique()

# ★ 优化点：这是一套兼顾高区分度与美观柔和的“现代商务色板”
# 摒弃了辣眼睛的纯色，保留了清晰的色相差异，整体呈现高级沉稳的质感
balanced_colors = [
    '#4A7DB8', # 稳重蓝
    '#E16D54', # 砖红色
    '#5AA874', # 清新绿
    '#F4B04D', # 明黄色
    '#9875B4', # 丁香紫
    '#62C2A6', # 薄荷青
    '#DE85B1', # 玫瑰粉
    '#836F64', # 咖啡灰
    '#5698C3', # 湖蓝色
    '#C4C255', # 橄榄绿
    '#D36B72', # 豆沙红
    '#A6C56D', # 草绿色
    '#C19875', # 卡其棕
    '#7A91CA', # 鸢尾蓝
    '#F19572', # 珊瑚橙
    '#7BAF95', # 灰绿色
    '#C2849F', # 藕粉色
    '#8A977A', # 鼠尾草绿
    '#527D8C', # 莫兰迪蓝
    '#D9B26D'  # 砂黄色
]

# 将你的指标层循环映射到上面的色板中
color_dict = {topic: balanced_colors[i % len(balanced_colors)] for i, topic in enumerate(unique_topics)}

# ==========================================
# 4. 图表结构配置映射
# ==========================================
plot_config = [
    {'raw_name': '中国24上', 'title': 'H1 2024 - 中国', 'pos': (0, 0)},
    {'raw_name': '美国24上', 'title': 'H1 2024 - 美国', 'pos': (0, 1)},
    {'raw_name': '中国24下', 'title': 'H2 2024 - 中国', 'pos': (0, 2)},
    {'raw_name': '美国24下', 'title': 'H2 2024 - 美国', 'pos': (0, 3)},
    {'raw_name': '中国25上', 'title': 'H1 2025 - 中国', 'pos': (1, 0)},
    {'raw_name': '美国25上', 'title': 'H1 2025 - 美国', 'pos': (1, 1)},
    {'raw_name': '中国25下', 'title': 'H2 2025 - 中国', 'pos': (1, 2)},
    {'raw_name': '美国25下', 'title': 'H2 2025 - 美国', 'pos': (1, 3)},
]

# ★ 优化点：因为加入了文字标签，将 X 轴最大值扩大至 1.25 倍，留出文字显示空间
global_x_max = df['综合满意度'].max() * 1.25 

# ==========================================
# 5. 开始绘制子图
# ==========================================
fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(22, 12), dpi=150)
# fig.suptitle('2024-2025年中美聊天机器人应用用户满意度主题权重时序对比', fontsize=22, fontweight='bold', y=0.96)

for cfg in plot_config:
    row, col = cfg['pos']
    ax = axes[row, col]
    
    # 提取当前数据并排序
    sub_df = df[df['国家'] == cfg['raw_name']].copy()
    
    # 防止因某个阶段没有数据而报错跳出
    if sub_df.empty:
        ax.set_title(cfg['title'], fontsize=14, pad=10)
        ax.set_xticks([])
        ax.set_yticks([])
        # 补全空白图表的边框
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('#666666')
            spine.set_linewidth(1.0)
        continue

    sub_df = sub_df.sort_values(by='综合满意度', ascending=True)
    bar_colors = [color_dict[topic] for topic in sub_df['指标层']]
    
    # 画图，将返回的对象保存给 bars 变量
    bars = ax.barh(sub_df['指标层'], sub_df['综合满意度'], color=bar_colors, height=0.6, edgecolor='none')
    
    # ★ 优化点：为柱状图添加数值标签
    # fmt='%.3f' 表示保留三位小数，padding=4 表示文字距离柱子边缘 4 个像素
    ax.bar_label(bars, fmt='%.4f', padding=4, fontsize=10, color='#333333')
    
    # 标题与刻度
    ax.set_title(cfg['title'], fontsize=14, pad=10)
    ax.set_xlabel('综合满意度权重', fontsize=11)
    ax.set_xlim(0, global_x_max)
    ax.tick_params(axis='y', labelsize=11)
    ax.tick_params(axis='x', labelsize=10)
    
    # 补全所有边框，并设置为统一的、柔和的深灰色线框
    for spine in ax.spines.values():
        spine.set_visible(True)      
        spine.set_color('#666666')   
        spine.set_linewidth(1.0)     

# ==========================================
# 6. 全局图例
# ==========================================
legend_patches = [mpatches.Patch(color=color_dict[topic], label=topic) for topic in unique_topics]

fig.legend(handles=legend_patches, 
           title='主题图例', 
           title_fontsize='13',
           fontsize='11',
           loc='center right',       
           bbox_to_anchor=(1.02,0.5), 
           framealpha=0.9,
           edgecolor='#666666')

# ==========================================
# 7. 布局调整与保存
# ==========================================
plt.subplots_adjust(left=0.05, right=0.92, top=0.90, bottom=0.08, wspace=0.45, hspace=0.25)

plt.savefig('bar_charts_comparison_final.png', bbox_inches='tight')
plt.show()