# 3.2 泛娱乐特征工程与模型代码构建

## 学习目标

* 目标
  * 说明泛娱乐推荐系统的特征工程过程
* 应用
  * 应用完成泛娱乐推荐系统Wide&Deep模型的构建

### 3.2.1 特征工程

#### 3.2.1.1 定义正负样本

* 根据模型最终的预测要求：使用户产生更多的交互行为, 来定义正负样本
  * 正样本定义: 若用户A对帖子B产生交互行为, 则A的所有特征和B的所有特征连接组成的向量作为正样本特征, 1作为正样本的标签.
  * 负样本定义: 若用户A对帖子B产生负向行为(举报/不感兴趣)或用户A未对已推荐的帖子B产生任何行为, 则A的所有特征和B的所有特征连接组成的向量作为负样本特征, 0作为负样本的标签.
  * 因为我们的模型最终使用ACC作为评估标准, 我们也需要把正负样本的比例应维持在1:1左右, 实际情况中, 用户主动标记举报/不感兴趣的情况非常少, 为了补充负样本数量, 需要将每次推荐给用户但未产生交互行为的数据定义为负样本

#### 3.2.1.2 获取初始训练集

* 选取最新产生的200万条样本, 样本总数的选择依据模型参数总量，而且需要兼顾模型训练时间和模型效果，有关实验表明, 在数据集存在噪音的情况下, 样本总数应该是模型参数量的10倍左右，将得到具备拟合能力与泛化能力较强的模型。
* 训练数据样式:

```
android,23,4,32,43,32,2,54,1502378738,2,4,6,33,421,22,43,12,0,0,0
ios,47,4,16,28,32,43,56,1502408488,22,33,12,2,1,23,4,47,0,0,1
android,22,4,16,7,0,0,7,1502362845,2,34,62,2,32,221,32,4,0,0,0
android,23,21,122,223,33,23,42,1502367552,77,11,2,3,87,1,20,6,2,0,0
android,76,432,876,23,2,23,56,1502430914,2,32,1,23,54,66,33,212,199,0,1
```

* 每一列分别代表: 设备系统类型、用户的转发数、评论数、点赞数、发布帖子数、帖子被点赞数、被转发数、被评论数、帖子发布时间戳、用户关注第一位明星编号、第二位明星编号、第三位明星编号、第四位明星编号、第五位明星编号、贴子涉及第一位明星编号、第二位明星编号、第三位明星编号、第四位明星编号、第五位明星编号、目标标签
* wide-deep模型参数量的计算法分为两部分: 
  * wide模型侧：对应的稀疏特征扩展之后的维度，泛娱乐的特征大概在5万维左右, 其中包括原始稀疏特征, hash分桶的连续特征, 以及组合特征 
  * deep模型侧,：有大概1000维的特征作为输入，参数总量大致11万左右.
  * 因此我们的模型参数总量为：16万, 因此样本数量最好维持在160万以上，考虑到还应存在测试数据集。因此每次使用相对于当前最新的200万条数据

### 3.2.2 训练样本获取

* 目的：读取数据库中的数据构造成样本
  * 设备系统类型、用户的转发数、评论数、点赞数、发布帖子数、帖子被点赞数、被转发数、被评论数、帖子发布时间戳、用户关注第一位明星编号、第二位明星编号、第三位明星编号、第四位明星编号、第五位明星编号、贴子涉及第一位明星编号、第二位明星编号、第三位明星编号、第四位明星编号、第五位明星编号、目标标签。20个原始特征列：19 + 1

我们需要从neo4j当中获取训练样本，构造样本过程分为正样本和负样本两部分，正样本: 用户产生交互行为的双画像，负样本: 推荐曝光后没有产生交互行为的双画像，通过cypher语句取出特征。

1、读取用户、帖子特征进行组合cypher语句

```python
from neo4j.v1 import GraphDatabase
import numpy as np
import pandas as pd 
NEO4J_CONFIG = dict({
    "uri": "bolt://192.168.19.137:7687",
    "auth": ("neo4j", "itcast"),
    "encrypted": False
})

_driver = GraphDatabase.driver(**NEO4J_CONFIG)

# 选择有过行为关系的用户和帖子，将相关特征合并，以及目标为1
def get_positive_sample():
    cypher = "match(a:SuperfansUser)-[r:share|comment|like]-(b:SuperfansPost)  return [a.like_posts_num, a.forward_posts_num, a.comment_posts_num,a.publish_posts_num,a.follow_stars_list,b.hot_score,b.commented_num,b.forwarded_num,b.liked_num,b.related_stars_list,b.publish_time] limit 200"
    train_data_with_labels = get_train_data(cypher, '1') 
    return train_data_with_labels
# 选择没有关系的用户和帖子，将相关特征合并，以及目标为0
def get_negative_sample():
    cypher = "match(a:SuperfansUser)-[r:report|unlike]-(b:SuperfansPost)  return [a.like_posts_num, a.forward_posts_num, a.comment_posts_num,a.publish_posts_num,a.follow_stars_list,b.hot_score,b.commented_num,b.forwarded_num,b.liked_num,b.related_stars_list,b.publish_time] limit 200"
    train_data_with_labels = get_train_data(cypher, '0')
    return train_data_with_labels
```

2、获取结果之后，进行数据集的格式处理和构造：get_train_data(cypher, '1') 

```python
# [92, 47, 4, 1618, ['218960', '187579', '210958', '219148', '3116'], 549, 5, 2, 533, ['1'], 1516180431]
def _extended_length(b, index):
    """
    :param b: 传入样本
    :param index: 传入位置
    :return:
    """
    print(b, index)
    for i in index:
        if len(b[i])<5:
            k = [0]*5
            for i, value in enumerate(b[i]):
                k[i] = value
            b.extend(k)
    print(b)
    i = 0
    while i < len(index):
        b.pop(index[i] - i)
        i += 1
    return b

def get_train_data(cypher, label):
    ### 根据neo4j关系生成标注数据
    # 正样本: 用户产生交互行为的双画像
    # 负样本: 推荐曝光后没有产生交互行为的双画像
    with _driver.session() as session:
       record = session.run(cypher)
       sample = list(map(lambda x: x[0], record))

    index_list = [4,9]
    # 第一步特征处理: 列表特征处理
    train_data = list(map(lambda x: _extended_length(x, index_list) + [str(label)], sample))
    print(train_data)
    return train_data
```

最后保存到本地当前目录train_data.csv文件

```python
if __name__ == "__main__":
    p_train_data = get_positive_sample()
    n_train_data = get_negative_sample()
    print(len(p_train_data))
    print(len(n_train_data))
    train_data = p_train_data + n_train_data
    pd.DataFrame(train_data).to_csv("./train_data.csv", header=False, index=False)
```

### 3.2.3 模型构建

* 分析:
  * model.py中包含了从源数据文件到模型指定输入格式数据的全部特征工程; 
  * 其中两个重要函数build_estimator和input_fn**被task.py中的关键函数调用;**
  * 要求: build_estimator必须返回tf的分类器类型, 具体参见tf.estimator.DNNLinearCombinedClassifier()源码

* 目的：构建泛娱乐WDL模型输入、特征的处理，从而进行后续模型训练
* 步骤：
  * 1、模型输入函数构建
  * 2、 tf.feature_column特征处理
  * 3、DNNLinearCombinedClassifier模型构建

#### 3.2.3.1 模型输入函数

* 使用tf.data.TextLineDataset(filenames)解析我们的训练数据集CSV文件

1、指定读取CSV文件API，返回dataset

```python
def input_fn(filenames,
             num_epochs=None,
             shuffle=True,
             skip_header_lines=0,
             batch_size=200):
  dataset = tf.data.TextLineDataset(filenames).skip(skip_header_lines).map(
      _decode_csv)
```

2、实现_decode_csv解析文件内容特征以及目标值函数

所有CSV列，以及解析式的默认格式。

```python
CSV_COLUMNS = [
'like_posts_num', 'forward_posts_num', 'comment_posts_num', 'publish_posts_num', 'hot_score',
    'commented_num', 'forwarded_num', 'liked_num', 'publish_time', 'follow_star_1', 
    'follow_star_2', 'follow_star_3', 'follow_star_4', 'follow_star_5', 'related_star_1',
    'related_star_2', 'related_star_3', 'related_star_4', 'related_star_5', 'islike'
]
CSV_COLUMN_DEFAULTS = [[''], [0], [0], [0], [0],
                       [0], [0], [0], [0], [0], 
                       [0], [0], [0], [0], [0],
                       [0], [0], [0], [0], ['']]

LABEL_COLUMN = 'islike'
LABELS = ['1', '0']
```

使用tf.decode_csv解析


```python
def _decode_csv(line):
  #  ## ['123','321'] ---> [['123'], ['321']]
  row_columns = tf.expand_dims(line, -1)  
  # ##修改各个特征的类型 
  columns = tf.decode_csv(row_columns, record_defaults=CSV_COLUMN_DEFAULTS) 
  features = dict(zip(CSV_COLUMNS, columns))
  # Remove unused columns
  for col in UNUSED_COLUMNS:
      features.pop(col)
  return features
```

其中会对特征进行过滤，指定无用的特征列 UNUSED_COLUMNS，输入的特征以及目标标签都会需要

UNUSED_COLUMNS = set(CSV_COLUMNS) - {col.name for col in INPUT_COLUMNS} - {LABEL_COLUMN}

这里UNUSED_COLUMNS有'device_system'以及'islike'这两列

```python
# 指定StarID维度列表映射大小
STAR_ID_LIST = list(map(lambda x: x, range(0,500)))
INPUT_COLUMNS = [
    tf.feature_column.numeric_column('like_posts_num'),              
    tf.feature_column.numeric_column('forward_posts_num'),      
    tf.feature_column.numeric_column('comment_posts_num'),      
    tf.feature_column.numeric_column('publish_posts_num'),      
    tf.feature_column.numeric_column('commented_num'),      
    tf.feature_column.numeric_column('forwarded_num'),      
    tf.feature_column.numeric_column('liked_num'),      
    tf.feature_column.numeric_column('publish_time'),      
    tf.feature_column.categorical_column_with_vocabulary_list(
                'follow_star_1', STAR_ID_LIST),
    tf.feature_column.categorical_column_with_vocabulary_list(
                'follow_star_2', STAR_ID_LIST),
    tf.feature_column.categorical_column_with_vocabulary_list(
                'follow_star_3', STAR_ID_LIST),
    tf.feature_column.categorical_column_with_vocabulary_list(
                'follow_star_4', STAR_ID_LIST),
    tf.feature_column.categorical_column_with_vocabulary_list(
                'follow_star_5', STAR_ID_LIST),
    tf.feature_column.categorical_column_with_vocabulary_list(
                'related_star_1', STAR_ID_LIST),
    tf.feature_column.categorical_column_with_vocabulary_list(
                'related_star_2', STAR_ID_LIST),
    tf.feature_column.categorical_column_with_vocabulary_list(
                'related_star_3', STAR_ID_LIST),
    tf.feature_column.categorical_column_with_vocabulary_list(
                'related_star_4', STAR_ID_LIST),
    tf.feature_column.categorical_column_with_vocabulary_list(
                'related_star_5', STAR_ID_LIST)
]
```

3、dataset进行制定epoch以及Batch大小，打乱顺序，并指定目标值，将字符串编程目标0，1

使用dataset的batch,repeat相关方法进行处理，

```python
  if shuffle:
      dataset = dataset.shuffle(buffer_size=batch_size * 10)                                                  
  iterator = dataset.repeat(num_epochs).batch(
      batch_size).make_one_shot_iterator()
  features = iterator.get_next()
  return features, parse_label_column(features.pop(LABEL_COLUMN))
```

最后将目标值进行处理

```python
def parse_label_column(label_string_tensor):
  table = tf.contrib.lookup.index_table_from_tensor(tf.constant(LABELS))
  return table.lookup(label_string_tensor)
```

#### 3.2.3.2 tf.feature_column特征处理

```
(like_posts_num, forward_posts_num, comment_posts_num, publish_posts_num, hot_score,
commented_num, forwarded_num, liked_num, publish_time, follow_star_1,
follow_star_2, follow_star_3, follow_star_4, follow_star_5, related_star_1,
related_star_2, related_star_3, related_star_4, related_star_5) = INPUT_COLUMNS
```

* 数值型特征：
  * 类别特征进行数值化操作
  * ['like_posts_num', 'forward_posts_num', 'comment_posts_num', 'publish_posts_num', 'hot_score','commented_num', 'forwarded_num', 'liked_num', 'publish_time']
* 类别型特征：
  * ['device_system','follow_star_1', 'follow_star_2', 'follow_star_3', 'follow_star_4', 'follow_star_5', 'related_star_1','related_star_2', 'related_star_3', 'related_star_4', 'related_star_5']

#### wide侧特征列指定

* device_system,follow_star_1,follow_star_2,follow_star_3,follow_star_4,follow_star_5,related_star_1,related_star_2,related_star_3,related_star_4,related_star_5
  * [follow_star_1,follow_star_2]与 [related_star_1,related_star_2, related_star_3, related_star_4, related_star_5]的两两组合交叉特征
  * [follow_star_1, related_star_1, follow_star_2]
  * [follow_star_2, related_star_1, related_star_2]
  * [follow_star_3, related_star_1, related_star_2]
  * [follow_star_1, related_star_2, related_star_1]

```python
wide_columns = [
      tf.feature_column.crossed_column([follow_star_1, related_star_1],
                                       hash_bucket_size=int(1e3)),
      tf.feature_column.crossed_column([follow_star_1, related_star_2],
                                       hash_bucket_size=int(1e3)),
      tf.feature_column.crossed_column([follow_star_1, related_star_3],
                                       hash_bucket_size=int(1e3)),
      tf.feature_column.crossed_column([follow_star_1, related_star_4],
                                       hash_bucket_size=int(1e3)),
      tf.feature_column.crossed_column([follow_star_1, related_star_5],
                                       hash_bucket_size=int(1e3)),
      tf.feature_column.crossed_column([follow_star_2, related_star_1],
                                       hash_bucket_size=int(1e3)),
      tf.feature_column.crossed_column([follow_star_2, related_star_2],
                                       hash_bucket_size=int(1e3)),
      tf.feature_column.crossed_column([follow_star_2, related_star_3],
                                       hash_bucket_size=int(1e3)),
     tf.feature_column.crossed_column([follow_star_2, related_star_4],
                                       hash_bucket_size=int(1e3)),
     tf.feature_column.crossed_column([follow_star_2, related_star_5],
                                       hash_bucket_size=int(1e3)),
     tf.feature_column.crossed_column([follow_star_1, related_star_1, follow_star_2],
                                       hash_bucket_size=int(1e4)),
     tf.feature_column.crossed_column([follow_star_2, related_star_1, related_star_2],
                                       hash_bucket_size=int(1e4)),
     tf.feature_column.crossed_column([follow_star_3, related_star_1, related_star_2],
                                       hash_bucket_size=int(1e4)),
     tf.feature_column.crossed_column([follow_star_1, related_star_2, related_star_1],
                                       hash_bucket_size=int(1e4)),
     device_system,
     follow_star_1,
     follow_star_2,
     follow_star_3,
     follow_star_4,
     follow_star_5,
     related_star_1,
     related_star_2,
     related_star_3,
     related_star_4,
     related_star_5
  ]
```

#### deep侧特征指定

* 类别型特征进行indicator_column指定 + 数值型特征

```python
# 深度特征比做挖掘特征，针对稀疏+稠密的所有特征, 但由于隐层作用时将考虑大小问题，因此类别特征必须onehot编码才能作为输入
deep_columns = [
      tf.feature_column.indicator_column(follow_star_1),
      tf.feature_column.indicator_column(follow_star_2),
      tf.feature_column.indicator_column(follow_star_3),
      tf.feature_column.indicator_column(follow_star_4),
      tf.feature_column.indicator_column(follow_star_5),
      tf.feature_column.indicator_column(related_star_1),
      tf.feature_column.indicator_column(related_star_2),
      tf.feature_column.indicator_column(related_star_3),
      tf.feature_column.indicator_column(related_star_4),
      tf.feature_column.indicator_column(related_star_5),
      like_posts_num,
      forward_posts_num,
      comment_posts_num,
      publish_posts_num,
      commented_num,
      forwarded_num,
      liked_num,
      publish_time
  ]
```

####3.2.3.3 Wide&Deep模型构建

* 这里填入一个配置参数，wide和deep的特征列指定，dnn网络的神经元个数以及层数
  * [100, 70, 50, 25]只是作为我们初始化的一个值，后面在训练阶段会自动调参

```python
def build_estimator(config, embedding_size=8, hidden_units=None):
	"""
	"""
	# 特征处理
	
	# 模型构建
  return tf.estimator.DNNLinearCombinedClassifier(config=config,
                                                  linear_feature_columns=wide_columns,
                                                  dnn_feature_columns=deep_columns,
                                                  dnn_hidden_units=[embedding_size] + [100, 70, 50, 25]) 
```

### 3.2.4 小结

* 泛娱乐推荐特征工程以及样本的导出
* 泛娱乐模型构建
  * 输入数据函数构建
  * 特征指定
  * build_estimator构建