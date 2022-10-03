# データセットと結果計算関数の使い方

```py
# データセットの使い方, shuffle済み
from reader import customized_ds, calculate_result
train_ds, test_ds = customized_ds() # train_dsの長さは448で、test_dsは106

# サンプル
print(train_ds[:10])
# => [('やら', 1), ('腹づもり', 1), ('やはり', 1), ('徒労', 1), ('斎場', 1), ('おみ足', 1), ('とばっちり', 0), ('今ひとつ', 1), ('泥臭い', 0), ('不本意', 1)]

# your_functionを完成してください、result_list: 106個の0/1結果が必要、格調高い単語は1
result_list = your_function(train_ds, test_ds) 
calculate_result(result_list)
```

