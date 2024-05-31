import matplotlib.pyplot as plt
import pandas as pd

data = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Cursor Movement
    [1, 1, 1, 29/30, 1, 1, 1, 28/30, 1, 1, 1, 1, 1, 1, 29/30, 1, 1, 1, 24/30, 29/30],  # Left Clicking
    [1, 1, 1, 29/30, 1, 26/30, 1, 28/30, 27/30, 1, 28/30, 1, 1, 1, 29/30, 1, 1, 1, 1, 28/30],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
]

# ジェスチャの名前
gestures = ['Cursor Movement', 'Left Clicking', 'Right Clicking', 'Scrolling Up', 'Scrolling Down', 'Initiative']

# DataFrameの作成
df = pd.DataFrame(data).T  # .Tでデータの転置を行い、列が各ジェスチャに対応するようにする
df.columns = gestures  # 列名をジェスチャの名前に設定

# 箱ひげ図の作成
plt.figure(figsize=(12, 8))  # グラフのサイズを指定
df.boxplot()  # Pandasのboxplotメソッドを使用
plt.title('Gesture Recognition Accuracy')  # タイトル
plt.ylabel('Accuracy')  # Y軸ラベル
plt.show()  # グラフの表示
