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

# epoch = 0, loss = 422.17002775287256, prec = 0.8083976833976834, rec = 0.8387581372058087, f1 = 0.8232981076431557
# epoch = 1, loss = 273.6536409219261, prec = 0.8530954879328436, rec = 0.814221331997997, f1 = 0.8332052267486548

```

## Next plan

- [ ] Train 3rd epoch and get scores: (F1 & BA & Prec & Rec) of (Dev & Test) Dataset
- [ ] Train kuro BERT and get results
- [ ] Train 2 BERTs with FL loss and get results

#### New Idea

- [ ] Train Bert without csg architechure 但是让他指出分割点 (理论上和csg seg的效果一样)
- [ ] 将两部分句子倒过来训练
- [ ] ETC

