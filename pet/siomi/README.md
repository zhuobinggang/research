# 使い方

```py
from main import *
model = train_model(data_points = 256, batch_size = 4)
left = '手土産'
right = 'お持たせ'
print(get_predicted_word_by_word_pair(model, left, right))
```
