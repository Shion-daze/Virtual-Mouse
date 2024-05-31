import numpy as np
from scipy import stats

# あるジェスチャの成功率のデータをnumpy配列として
success_rates = np.array([1, 1, 1, 29/30, 1, 26/30, 1, 28/30, 27/30, 1, 28/30, 1, 1, 1, 29/30, 1, 1, 1, 1, 28/30])

# 平均と標準偏差を計算
mean = np.mean(success_rates)
std_dev = np.std(success_rates, ddof=1)  # 標本標準偏差

# サンプルサイズ
n = len(success_rates)

# 95%信頼区間を計算
confidence_interval = stats.norm.interval(0.95, loc=mean, scale=std_dev/np.sqrt(n))

print("信頼区間:", confidence_interval)
