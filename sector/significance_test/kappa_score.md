## pythonでカッパ値を計算する方法

1. `sklearn`というライブラリを使うので、パソコンにいなければ`pip install sklearn`でインストール
2. 関数の公式説明サイトは: `https://scikit-learn.org/stable/modules/generated/sklearn.metrics.cohen_kappa_score.html`
3. 使用例:

```py
from sklearn import metrics

a1 = [0, 1, 1, 1, 1, 0, 0]
a2 = [0, 1, 0, 1, 1, 0, 0]
score = metrics.cohen_kappa_score(a1, a2)
print(score)

# >>> 0.72
```

