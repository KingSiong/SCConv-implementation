# the Implementation of SCConv

he says his model is fine, while i keep silent.

these codes are based on [kuangliu: pytorch-cifar](https://github.com/kuangliu/pytorch-cifar/tree/master).

## 实验记录

在 CIFAR-10 数据集上，分别用 ResNet50 和将 $3\times 3$ 卷积替换成 SCConv 的 ResNet50，分别训练 200 个 epoch 的实验结果：

|Model|Acc($\%$)|FLOPs(G)|Params(M)|
|---|---|---|---|
|ResNet-50|94.93|2.62|23.52|
|ResNet-50 with SCConv|92.03|1.86|15.91|