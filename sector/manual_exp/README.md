## 使用说明

1. 首先将ld文件就那样复制过来，参考`mld.py`
2. 然后用python将它变成json输出
3. 将文件改成js，暴露datas
4. 在test.html将数据源文件改成刚才的js

## 命令记录

从10个test测试集里各自取出10个case组成mld

```py
from mainichi_paragraph import *
tlds = read_additional_test_ds()

for ld in tlds:
  random.shuffle(ld)

mld = []

for ld in tlds:
  mld += ld[:10]

f = open('manual_exp/mld2.py', 'w')
f.write(str(mld))
f.close()
```

生成js文件

```py
from mld2 import datas
import json
ds = json.dumps(datas)
>>> len(ds)
>>> 94371
f = open('datas2.txt', 'w')
f.write(str(ds))
f.close()
```

用机器跑实验

```py
from manual_exp.mld2 import datas
import torch as t
m = t.load('save/r01_fl50_0.tch')
from sec_paragraph import *
cal_f1(m, datas)
>>> 0.7142857142857143
```


