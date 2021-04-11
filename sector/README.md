## 使用每日新闻进行文本分割

这几天进行的工作和一些注意事项
- [X] DONE: 完成数据集处理
- [X] NOTE: 关于2vs2设置，lr需要从2e-5 -> 5e-6，不然会导致失败。调整以后2vs2比起1vs1，f1提高了:0.6813898704358068 -> 0.7224777675559645，是符合直观的结果，不需要再用该死的先行研究数据。后续可以实验3vs3。
- [X] NOTE: 关于epoch数，每日新闻数据集只需要跑一遍即可
- [X] NOTE: 加上ordering之后的epoch数，第三epoch也一样有性能提升(第四开始下降)，但是估计拿不到有意差所以就放弃了把
- [X] NOTE: 多个SEP + 用中间SEP来判断导致性能下降了4%左右
- [X] NOTE: mini数据集 + Standard(1sep1cls 用cls判断):  epoch1没有converge， epoch2("prec": 0.6914285714285714, "rec": 0.7289156626506024), epoch3(prec0.743859649122807, rec0.6385542168674698)
- [X] NOTE: mini数据集 + Standard 1vs1:  epoch1(prec0.759825327510917, rec0.5240963855421686), epoch2(prec0.6872964169381107, rec0.6355421686746988), epoch3(prec0.6590257879656161,rec0.6927710843373494)
- [X] CONCLUDE: mini数据集，epoch2即可
- [ ] NOTE: 多个SEP + 用CLS来判断
- [ ] NOTE: 两个SEP + 用CLS来判断
- [ ] NOTE: 单个SEP + 用CLS来判断


## 数据收集

数据收集在run.py里



