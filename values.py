import numpy as np

success_rates = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

average = np.mean(success_rates)
variance = np.var(success_rates)
std_deviation = np.sqrt(variance)
coefficient_of_variation = std_deviation / average

# 結果を表示する
print("Average Success Rate:", average)
print("Variance:", variance)
print("Standard Deviation:", std_deviation)
print("Coefficient of Variation:", coefficient_of_variation)
