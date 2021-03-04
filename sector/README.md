## News

Setting: 

* BERT: tohoku university whole word bert
* Architecture: Cross Segmentation Attention 128 (From paper: Text Segmentation by Cross Segment Attention)
* Weighting loss: {1: 2, 0: 1}
* Epochs: 2

Command & Result: 

```py
import csg_bert as M
m = M.Model()
M.run(m)

# === output ===

# Model No.1

# epoch = 1, loss = 422.17002775287256, prec = 0.8083976833976834, rec = 0.8387581372058087, f1 = 0.8232981076431557
# epoch = 2, loss = 273.6536409219261, prec = 0.8530954879328436, rec = 0.814221331997997, f1 = 0.8332052267486548
# epoch = 3, loss = 166.14195720525458, prec = 0.8652906029331885, rec = 0.7976965448172258, f1 = 0.8301198540906722, bacc = 0.8854356169191592
# epoch = 4, loss = 100.65162188466638, prec = 0.8045697617890131, rec = 0.828743114672008, f1 = 0.8164775530340405, bacc = 0.8926300754539056

# Model No.2

# epoch = 1, loss = 396.53721253667027, prec = 0.8276735935124176, rec = 0.8177265898848273, f1 = 0.822670025188917, bacc = 0.8904749769326786


```

## Next plan

- [ ] Train 2 BERTs with FL loss and get results

- [ ] 实验数据收集
  - [ ] 跑16个model，每个epoch=2, 但是要同时得出dev数据集的结果 (Doing 在panther上跑着呢)
  - [ ] 跑16个kuro bert model，每个epoch=2, 但是要同时得出dev数据集的结果 (已编码, 速度慢一倍有余，只跑8个算了)

#### New Idea

- [ ] Train Bert without csg architechure 但是让他指出分割点 (理论上和csg seg的效果一样)
- [ ] 将两部分句子倒过来训练
- [ ] ETC

