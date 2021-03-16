## 使用

```
import runner
ld = runner.train_ld() # OK
m = runner.gru_baseline_model()
for _ in range(3):
  runner.run(m, ld, epoch=1)
  runner.test(m)
```

## Result

1. Encoder: SelfAttention(with pos encoding), Decoder: GRU: 0.88 -> 0.917 -> 0.947
2. Encoder: SelfAttention(no pos encoding), Decoder: GRU: 0.918 -> 0.913 -> 0.962
3. Encoder: Embedding, Decoder: GRU: 0.954 -> 0.971
4. Encoder: Embedding, Decoder: TF(no pos encoding): 0.782 -> 0.895 -> 0.939
4. Encoder: Embedding, Decoder: TF(no pos encoding) (X 2):  0.742
4. Encoder: Embedding, Decoder: TF(no pos encoding) (fw = 96) (head = 4):  0.689 
4. Encoder: Embedding, Decoder: TF(no pos encoding) (fw = 96) (head = 8):  0.778
4. Encoder: Embedding, Decoder: TF(no pos encoding) (fw = 96) (head = 16):  0.736
5. Encoder: Embedding, Decoder: TF(with pos encoding) (fw = 96): 0.691 -> 0.848
6. Encoder: Embedding, Decoder: TF(with pos encoding) (fw = 32): 0.636 -> 0.831
7. Encoder: Embedding, Decoder: MyTF(with pos encoding): 0.792 -> 0.861 


## Analysis

### 1 vs 2 vs 3
1. encoder方面有没有pos encoding都一样
2. 甚至encoder方面没有selfatt都一样

### 4 vs 5

排序任务里显然只需attend到最大那个数就可以了，没有顺序的必要。需要换个测试方案
