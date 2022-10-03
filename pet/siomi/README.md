# データセットと結果計算関数の使い方

```py
# データセットの使い方, shuffle済み
from reader import *
train_ds, test_ds = customized_ds() # train_dsの長さは448で、test_dsは106

# your_functionを完成してください
result_list = your_function(train_ds, test_ds) # result_list: 106個の0/1結果が必要、格調高い単語は1
calculate_result(result_list)
```

