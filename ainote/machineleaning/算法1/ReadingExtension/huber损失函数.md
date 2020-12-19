# Huber Loss

Huber Loss 是一个用于回归问题的带参损失函数, 优点是能增强平方误差损失函数(MSE, mean square error)对离群点的鲁棒性。

- 当预测偏差小于 δ 时，它采用平方误差,
- 当预测偏差大于 δ 时，采用的线性误差。

相比于最小二乘的线性回归，Huber Loss降低了对离群点的惩罚程度，所以 Huber Loss 是一种常用的鲁棒的回归损失函数。



Huber Loss 定义如下：

![image-20191113165350530](https://tva1.sinaimg.cn/large/006y8mN6gy1g8wiigc950j318g0483z1.jpg)

δ 是 Huber Loss 的参数，y是真实值，f(x)是模型的预测值,

通过如上公式，我们可以发现，这个函数对于小一些的  $$(y-f(x))$$  误差函数是二次的，而对大的值误差函数是线性的。



Huber Loss 参数变化图：

![huberloss-w450](https://tva1.sinaimg.cn/large/006y8mN6gy1g8wilhs3ffg30p00fwad8.gif)