## pythonでcohenカッパ値(受験者二人)を計算する方法


* `sklearn`というライブラリを使うので、パソコンにいなければ`pip install sklearn`でインストール、関数の説明サイトは: `https://scikit-learn.org/stable/modules/generated/sklearn.metrics.cohen_kappa_score.html`
* 使用例:

```py
from sklearn import metrics

a1 = [0, 1, 1, 1, 1, 0, 0]
a2 = [0, 1, 0, 1, 1, 0, 0]
score = metrics.cohen_kappa_score(a1, a2)
print(score)

# >>> 0.72
```

## pythonでfleissカッパ値(受験者複数、二人以上可)を計算する方法

* `statsmodels`というライブラリを使用、関数の説明サイト: `https://www.statsmodels.org/dev/generated/statsmodels.stats.inter_rater.fleiss_kappa.html`
* 使用例:

```py
from statsmodels.stats import inter_rater
import numpy as np

a1 = [0, 1, 1, 1, 1, 0, 0]
a2 = [0, 1, 0, 1, 1, 0, 0]
a3 = [0, 1, 0, 1, 1, 0, 1]
matrix = np.array([a1, a2, a3])
matrix = matrix.T # (subject, rater)という形にするために、転置が必要

array, categorys = inter_rater.aggregate_raters(matrix)
score = inter_rater.fleiss_kappa(array)
print(score)

# >>> 0.6181818181818183


