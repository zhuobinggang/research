## 段落分割分析

段落分割领域是文本分割的一个子领域。文本分割领域XX等人用BERT获得了SOTA的成果。论文中提出的Cross Segmentation BERT架构获得了令人瞩目的成果，令人惊讶的是该架构只利用了分割点前后的局部信息。

在段落分割领域，XX等人在BERT上使用Focus Loss，对夏目的小说进行段落分割，据我们所知他们取得了日语段落分割领域的最新成果。他们的架构本质上也只利用了分割点前后的局部信息。

我们首先会想到的问题是，段落分割真的不需要长距离上下文吗？然后，即使先行研究使用BERT获得了不错的成果，可是他们并没有对段落分割这个问题进行深入剖析。

段落分割，特别是小说的段落分割，直觉上是一个很难的问题。只根据局部信息，人类真的能够在段落分割上击败机器吗？如果机器在段落分割上比人类做的更好，那么通过剖析机器的工作原理，人类是不是也能从中学到什么呢？ 这个问题就是本文的出发点。

在几年前，由于缺乏分析工具，神经网络对人类来说很大程度上是黑盒子。但是近年注意力机制的运用为我们打开了一扇分析的大门——通过辨别注意力的强弱，我们可以推知机器在思考什么。注意力机制非但可以提高神经网络的可解释性，更可贵的是直接推动了神经网络研究的边界，如今我们几乎能在所有最新成果中找到注意力机制的身影。

本文将借助注意力机制对段落分割任务进行分析。本文的注目点包括以下问题：

1. 对于1:1的段落分割配置，机器的注意力重心在前句还是后句？
2. 对手2:2的段落分割配置，机器的注意力重心在哪里？
3. 对段落分割贡献最大的元素是什么？ 根据注意力排序
4. 机器的判断基准足够健壮吗？ 我们能构思出让机器判断失效的例子吗？
5. 根据贡献排序，人类能学习到什么？
6. 将贡献最大的元素排除， 机器的准确率会被破坏吗？ 还是说会学习到更加健壮的判断方式？
7. 对失败的例子进行解析， 为什么神经网络在这里会失败？
8. 自注意层不同的头注意点在哪里？

## 对于1:1的段落分割配置，机器的注意力重心在前句还是后句？
* 得出前后句注意力的平均值
* 构建单侧神经网络，检验结果是否符合比例

## 对手2:2的段落分割配置，机器的注意力重心在哪里？
* 得出四个位置注意力的平均值
* 更宽的上下文有必要吗？（根据注意力平均值进行分析）

## 对段落分割贡献最大的元素是什么？ 根据注意力排序
* 是名词？动词？还是助词？

## 机器的判断基准足够健壮吗？ 我们能构思出让机器判断失效的例子吗？
* 人为构建侵入性句子的方法

## 根据贡献排序，人类能学习到什么？
* 机器的注意力符合我们的直觉吗？

## 将贡献最大的元素人为排除， 机器的准确率会下降吗？ 
* 还是说会学习到更加健壮的判断方式？

## 对失败的例子进行解析， 为什么神经网络在这里会失败？
* 将失败例子挑出进行分析

## 自注意层不同的头注意点在哪里?