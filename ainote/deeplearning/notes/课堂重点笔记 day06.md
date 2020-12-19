**深度学习的优缺点**：

- 优点：精度高，可以近似任意函数
- 缺点：计算量大（对数据量的依赖和对算力的依赖），解释性比较差，小数据集的场景效果不好，容易过拟合



**常见损失函数**：

- **分类任务**：
  - 多分类场景：交叉熵损失，tf.keras.losses.CategoricalCrossentropy()
  - 二分类场景：二分类交叉熵损失（负的对数似然损失），tf.keras.losses.BinaryCrossentropy()
- **回归任务**：
  - MAE损失，tf.keras.losses.MeanAbsoluteError()，零点不可导
  - MSE损失，tf.keras.losses.MeanSquaredError()，预测值和真实值差距很大的时候容易发生梯度爆炸
  - smooth L1损失：tf.keras.losses.Huber()

**常见的优化方法**：

- **SGD**，随机梯度下降，在TensorFlow中SGD这个API一般指的是小批量梯度下降，tf.keras.optimizers.SGD(learning_rate=0.1)

- **常用术语**：

  - epoch：表示将在整个数据集交给模型训练多少遍
  - batch_size：每次迭代交给模型训练的样本数，一般的取值：2^n
  - iteration（step）：表示训练完所有的epoch所需要的迭代的次数

- **反向传播（BP，backward propagation）**：

  - 前向反馈，得到预测值

  - 对比预测值和真实值之间的差距（损失函数）

  - 使用反向传播去进行求导，更新前面神经网络层中的参数

    - **链式求导法则**：对复合函数的求导

  - **梯度下降中（损失函数优化的过程中）经常会遇到这些问题**：

    - 在平坦区域下降很慢

    - 找到的是鞍点，非极值点

    - 找到的是非全局最优解，而是局部最优解

    - 解决方法：

      - **鞍点问题**：使用**动量算法优化**，使用指数加权平均对原始的梯度进行一个加权平均操作，将之前计算出来的梯度也考虑了进来

      - API：tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)

      - Nesterov accelerated gradient(NAG)：tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9, nesterov=True)

      - 在接近最优解时在最优解附近来回振荡，无法快速收敛：使用**AdamGrad优化方法**去解决，它的思想是越往后训练，学习率的值越小

        - API：

        ```python
        tf.keras.optimizers.Adagrad(
            learning_rate=0.001, initial_accumulator_value=0.1, epsilon=1e-07
        )
        ```

      - 解决AdaGrad后期学习率过小，无法快速收敛：**RMSpro**p进行优化（加权平均）

        - API：

        ```python
        tf.keras.optimizers.RMSprop(
            learning_rate=0.001, rho=0.9, momentum=0.0, epsilon=1e-07, centered=False,
            name='RMSprop', **kwargs
        )
        ```

      - **Adam**：综合了momentum和RMSprop的优势，即对学习率做了修正，又对梯度进行了修正

        - 没有经验性的调参建议，那么优先使用Adam
  - API：
      
  ```python
        tf.keras.optimizers.Adam(
            learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07
        )
        ```
    
- 学习率退火（在模型训练的过程中，通过callbacks回调函数调用）
    - **帮助我们在不同训练时机使用合适的学习率训练模型**
    - 分段常数衰减
    - 指数衰减
    - 1/t衰减
  
- 常用正则化方法：
  
  - **L1、L2、L1L2正则化**
  
  - **Dropout**：让模型在训练的过程中随机的让部分神经元失活（不参与到模型的计算），rate是神经元失活的概率，其它没有被失活的神经元的输入的值变成了   原始值/(1-rate)
  
  - Dropout API：layer = tf.keras.layers.Dropout(0,input_shape=(2,))
  
  - Early stopping：通过在模型训练过程中产生的指标，例如训练误差和验证误差进行一个观察，如果当训练误差不断减小的过程中，验证误差经历了达到最小后上升的过程，那么我们就可以断定当验证误差最小时，是当前条件下模型的最优表现，那么我们就可以设定对应的迭代阈值，让模型的训练及时停止
  
    - API实现：
  
      ```python
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',  patience=5
        )
        ```
  
      - 使用是在模型的训练过程中添加回调函数即可：
  
      - ```python
          model.fit(X, Y, epochs=10, batch_size=1, callbacks=[callback], verbose=1)
          ```
  
  - **批标准化（BN， batch normalization）**：
  
    - 在输入到下一层网络进行计算前，将前一层的每个神经元的输出进行标准化处理（计算每一个批次样本的均值和方差），对最后标准化的结果再做一个线性缩放，其中 $$\gamma$$ 和 $$\beta$$ 这两个参数也是需要在反向传播中和其它参数一起被优化
  
    - API：
  
      ```python
        tf.keras.layers.BatchNormalization(
            epsilon=0.001, center=True, scale=True,
            beta_initializer='zeros', gamma_initializer='ones',
        )
        ```

