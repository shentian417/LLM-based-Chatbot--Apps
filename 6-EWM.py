import pandas as pd
import numpy as np

# 1. 读取原始数据
df = pd.read_excel("E:\\研一\\AA聊天机器人-2425年\\3-LDA\\美国\\25下\\LDA_result-yw-25下-12.xlsx")

# 使用实际列名（根据你的数据列名修改）
ai_core_indicators = [ "生活智能助手","内容准确性","智能教育支持", "实用性","AI拟人化", "AI生成质量","响应速度", "系统稳定性","品牌认同感", "功能多样性", "情感支持", "费用"]
X = df[ai_core_indicators].values

# 样本数 n 和 指标数 m
n, m = X.shape
print(f"样本数: {n}, 指标数: {m}")

# ==========================================
# 熵权法计算
# ==========================================

# 标准化
X_min = X.min(axis=0)
X_max = X.max(axis=0)
Y = (X - X_min) / (X_max - X_min + 1e-10)  # 加极小值避免除0
Y = Y + 0.0001  # 避免log(0)

# 计算熵权
P = Y / Y.sum(axis=0)
k = 1 / np.log(n)
E = -k * np.sum(P * np.log(P), axis=0)
d = 1 - E
w_ewm = d / d.sum()

print("\n--- 技术性能 4个指标的熵权 (W_EWM) ---")# 使用实际类别（根据你的分类修改）
for i, name in enumerate(ai_core_indicators):
    print(f"{name}: {w_ewm[i]:.4f}")

print(f"\n权重总和验证: {w_ewm.sum():.6f}")

# 计算综合得分
df['技术性能得分'] = X.dot(w_ewm)
df['技术性能得分'] = df['技术性能得分'].round(2)

