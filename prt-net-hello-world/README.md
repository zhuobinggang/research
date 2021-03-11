## 使用

```
import runner
ld = runner.train_ld() # OK
m = runner.gru_baseline_model()
runner.run(m, ld, epoch=20)
runner.test(m)
```
