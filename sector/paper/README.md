# steps

```
platex template.tex
dvipdfmx template.dvi
```

# performance小节的计算方法

```py
from res_4method_10_m_10_testset import *
from exp2 import *
dd = cal(res)

>>> dd.shape
>>> (10, 4)
```

# 事例考察小节的计算结果方式

```py
from exp7 import *

ld = load_mld() # 长度7004

# 自动跳过记事开头的case
one_idxs = get_all_target_one_idxs(ld)
zero_idxs = get_all_target_zero_idxs(ld)

# 计算5个模型对于idxs的平均可能性输出，耗时要半个小说左右
aux_fl, aux, fl, stand = method4_model5_res(one_idxs, ld)  # (4, LENGTH, n)
aux_fl, aux, fl, stand = method4_model5_res(zero_idxs, ld)

# return: (global_idxs, case, methods_mean_posibility)
r0 = best_focus([aux_fl, aux, fl, stand], 0, ld, one_idxs, MAX = TRUE)
```
