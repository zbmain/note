# 2.4 召回金字塔

## 学习目标

- 目标
  - 知道召回金字塔的作用
- 应用
  - 应用完成召回金字塔的实现

### 2.4.1 召回金字塔

![](../images/召回模块组成图.jpg)

#### 2.4.1.1 为什么用召回金字塔

因为考虑到计算资源在数据变动时的滞后性，以及如何满足个别用户快速消费的需求, 我们引入召回金字塔模型, 并为每一位用户维护一组该模型。过程如下

* 满足公共召回策略帖子构成"金字塔"的底层，满足个性化召回策略帖子构成"金字塔"的中层, 各项召回策略叠加的帖子处在上层，从塔尖到塔底曝光比率逐渐增加。塔尖数据一般会优先进入下一个流程。 用户一次一般只会消费金子塔中的部分数据，因为全局召回按照窗口时间有序进行。

因此在此期间的用户推荐只需重新计算金字塔中的数据，并重新排列，保证金字塔数据的用户最快消费时间< 全局召回时间, 就可以保证用户获取到新的数据，并及时响应了用户最新行为。

#### 2.4.1.2 召回金字塔实现

- 什么是召回金字塔: **用于存放召回数据的缓存, 一般使用redis／ES实现, 并根据召回策略命中率进行从大到小排序, 将头部数据输送给规则过滤器服务.**
- 召回金字塔作用: 
  - 进一步缩小精排序的数据规模, 以便满足性能要求
  - 减轻召回池计算压力, 在全体数据无法实时计算时, 只需将金字塔中的数据进行计算并排序进而返回
- 召回金字塔机制实现(召回策略未分配权重)

- 输入：all_data=[[1,23,2], [3,2,54], [1,2,432]]
- 输出: [2,1,3,23,54,432]，并且给每个数据进行权重计算，排序后输出pid

召回金字塔机制实现：

```python
from itertools import chain

def pyramid_array(all_data):
    result = []
    for pid in set(chain(*all_data)):
        v = 0
        for list_ in all_data:
            if pid in list_:
                v += 1
        result.append([pid, v])

    result.sort(key=lambda x: x[1])
    return list(map(lambda x: x[0], result))[::-1]
```

#### 2.4.1.3 写入缓存数据

* _get_recomm:中使用
  * r_data = j_data_write(uid, j_data)：将数据写入金子塔并返回应该推送给**规则过滤器的数据**

```python
def j_data_write(uid, j_data):
    """将该用户的所有金字塔数据写入"""
    r = redis.StrictRedis(connection_pool=pool)
    r.set('j_' + str(uid), str(j_data)) 
    
    # 与当前（即上一次推送给）规则过滤器中数据做去重处理后推给过滤器
    old = r.get('r_' + str(uid)) 
    if not old:
        r_data = j_data[:50] 
    else:
        # 集合做差集
        r_data = list(set(j_data) - set(eval(old)))[:50] 
   
    r.set('r_'+str(uid), str(r_data))
    return r_data 
```

redis连接配置

```python
REDIS_CONFIG = dict({
    "host":"192.168.19.137",
    "port":"6379",
    "decode_responses":True  
})

pool = redis.ConnectionPool(**REDIS_CONFIG)
```

#### 2.4.1.4 整体召回模块功能添加

```python
def _get_recomm(IP, uid):
    """推荐全流程"""
    # 1、获取召回数据
    #（1）获得热门召回数据
    hot_data = _get_hot()
    # 获得最近发布召回数据
    last_data = get_last()
    # 获得单位时间内增长速度最快的帖子
    v_data = get_v()
    # 获得基于用户的协同过滤召回数据
    r_data = get_r(uid)
    # 随机召回数据
    random_data = get_random()
    all_data = [hot_data] + [last_data] + [v_data] + [r_data] + [random_data]
    # 进行金字塔规则计算并写入
    # （2）不给召回策略施加任何权重的召回金字塔计算
    j_data = pyramid_array(all_data)
    # 将数据写入金子塔并返回应该推送给规则过滤器的数据
    r_data = j_data_write(uid, j_data)
    # （3）将数据推送给规则规律器做数据内部去重
    # f_data = rfilter(r_data)
    
    # 2、排序部分
    return v_get_cache(IP)
```

### 2.4.2 小结

* 了解召回金字塔的原理和作用
* 召回金字塔结构实现以及缓存实现