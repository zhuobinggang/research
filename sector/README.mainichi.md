## 关于每日新闻的处理规则

1. 将所有AD=01(1面), 02(2面), 03(3面), 10(特集)的去掉
2. 将不含句号的T2去掉
3. 将T2数量<3的去掉(相当于只有一段)
4. 将同时含有【 & 】的行去掉，
5. 将只有一句话的段落去掉

```
articles = [[s_with_start_mark, s2, s3], [s_with_start_mark2, s5, s6]]
```