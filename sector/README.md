## Overview

Setting: 

* BERT: tohoku university whole word bert
* Architecture: Cross Segmentation Attention 128 (From paper: Text Segmentation by Cross Segment Attention)
* Weighting loss: {1: 2, 0: 1}
* Epochs: 2

## Next plan

### #1 BERT结构对比

- [ ] #1 CSG 128
- [ ] #2 CSG 64
- [ ] #3 单句子取出[CLS]Token
  - [X] Coding
- [ ] #4 单句子CSG 128
  - [X] Coding
- [ ] #5 单句子，同时训练BERT辨认分割点
- [ ] #6 CSG 256:0 (预测分割点)
- [ ] #7 Kuro Bert

#### 分析

1. #1 vs #2: 前后token量对段落分割精度的影响?
2. #3 vs #4: CSG结构(输入平衡性)对BERT精度的影响?
3. #5 vs #3: 增加限制可以让模型学习更多东西?
4. #5 vs #2#1: 辨认分割点Loss和CSG模型的内在联系?
5. #6 能否预测分割点？
6. #7 vs #1: 不同Bert精度对比

### #2 其他提高精度的小主意

- [ ] #1 将分割点前后部分倒过来训练


## Record

#### Check best epoch to stop for Simplest weight setting = 1:1

It can be epoch = 2

```
>>> losses
[1776.6383453648305, 1294.191233681282, 838.2970449654094, 644.4661880128842, 461.74606151817716]
>>> dics[0]
{'prec': 0.8119891008174387, 'rec': 0.4401772525849335, 'f1': 0.5708812260536399, 'bacc': 0.7122065152575113}
>>> dics[1]
{'prec': 0.7263157894736842, 'rec': 0.5096011816838996, 'f1': 0.5989583333333334, 'bacc': 0.7399502367181207}
>>> dics[2]
{'prec': 0.581081081081081, 'rec': 0.5716395864106352, 'f1': 0.5763216679076694, 'bacc': 0.7539486485857152}
>>> dics[3]
{'prec': 0.5940438871473355, 'rec': 0.5598227474150664, 'f1': 0.5764258555133079, 'bacc': 0.7503248989531353}
>>> dics[4]
{'prec': 0.6963562753036437, 'rec': 0.5081240768094535, 'f1': 0.5875320239111871, 'bacc': 0.7369270144156932}
```


