import numpy as np

# 成功率のデータ (右クリックと左クリック)
right_click_data = np.array([1, 1, 1, 29/30, 1, 26/30, 1, 28/30, 27/30, 1, 28/30, 1, 1, 1, 29/30, 1, 1, 1, 1, 28/30])
left_click_data = np.array([1, 1, 1, 29/30, 1, 1, 1, 28/30, 1, 1, 1, 1, 1, 1, 29/30, 1, 1, 1, 24/30, 29/30])

# 平均値の計算
mean_right_click = np.mean(right_click_data)
mean_left_click = np.mean(left_click_data)

# 標準偏差の計算
std_dev_right_click = np.std(right_click_data, ddof=1)
std_dev_left_click = np.std(left_click_data, ddof=1)

# 標準誤差の計算
std_err_right_click = std_dev_right_click / np.sqrt(len(right_click_data))
std_err_left_click = std_dev_left_click / np.sqrt(len(left_click_data))

print(f"右クリックの平均値: {mean_right_click:.4f}")
print(f"右クリックの標準偏差: {std_dev_right_click:.4f}")
print(f"右クリックの標準誤差: {std_err_right_click:.4f}")

print(f"左クリックの平均値: {mean_left_click:.4f}")
print(f"左クリックの標準偏差: {std_dev_left_click:.4f}")
print(f"左クリックの標準誤差: {std_err_left_click:.4f}")
