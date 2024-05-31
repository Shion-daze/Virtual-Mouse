import matplotlib.pyplot as plt
import numpy as np

# ユーザーからの入力データ（例）
probabilities_A = [93.25, 89.78, 78.03, 65.38, 53.35]  #先行研究の確率データ（5つの事象）
probabilities_B = [100, 100, 100, 100, 100]  # 自分の研究の確率データ（5つの事象）
# 事象の名前を改行文字を含む形式に変更
categories = ['2','3','4','5','6']

# X軸の位置を設定
x = np.arange(len(categories))

# グラフの幅
width = 0.35

# グラフを作成
fig, ax = plt.subplots()

# グループAのデータをプロット
rects1 = ax.bar(x - width/2, probabilities_A, width, label='Previous Research')

# グループBのデータをプロット
rects2 = ax.bar(x + width/2, probabilities_B, width, label='This Research')

# グラフのカスタマイズ
ax.set_xlabel('Number Of People')
ax.set_ylabel('Accuracy [%]')
ax.set_xticks(x)
ax.set_xticklabels(categories)  # 改行文字を含むカテゴリ名を表示

# ラベルの位置を調整
ax.legend(loc='upper left')

# ラベルの間隔を広げる
plt.xticks(rotation=0)

# Y軸の目盛りを設定（デフォルト）
# 目盛りを設定しないことで元に戻ります

# 'Result'の隣に 'Previous Research' と 'This Research' のラベルを配置
plt.legend(loc="upper right", bbox_to_anchor=(1.02, 0.87))

# グラフの表示
plt.tight_layout()  # ラベルがはみ出さないように調整
plt.show()
