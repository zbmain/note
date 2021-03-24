# 2.6 用户推荐逻辑完善

## 学习目标

* 目标
  * 了解用户推荐缓存获取逻辑
  * 知道用户行为数据读取操作
* 应用
  * 应用完成用户推荐缓存获取代码
  * 应用完成用户行为数据neo4j行为读写操作

### 2.6.1 用户推荐缓存获取

之前提供了一个这样的接口，用户获取用户缓存结果推荐出去，这里会用到用户的IP进行判断

* get_cache

根据用户请求，获取用户缓存结果（进行用户是否第一次使用某IP登陆判断）

```python
@api_view(['GET', 'POST'])
def get_cache(request):
    IP = request.META.get("HTTP_X_REAL_IP")
    result = r_api.v_get_cache(str(IP))
    return HttpResponse(json.dumps(result, ensure_ascii=False))
```

实现步骤：

* 判断用户是否第一次使用这个IP登录
  * 如果是，判断用户IP对应缓存是否为空
    * 为空：进行召回、排序过程推荐出去_get_recomm
    * 不为空：直接进行缓存结果获取（会自动删除获取的这些结果）

```python
def v_get_cache(IP):
    r = redis.StrictRedis(connection_pool=pool)
    # 判断该用户是否第一次使用这个IP登录
    uid = r.get("u_" + IP)
    if not uid:
        uid = random.choice([10033736])
        r.set("u_" + IP, uid)

    # 判断IP对应的缓存是否为空
    if not r.llen(IP):
        return _get_recomm(IP, int(uid))
    else:
        pid_list = eval(r.lpop(IP))
        print(pid_list)
        with _driver.session() as session:
            cypher = "match(a:SuperfansPost{pid:%d}) return properties(a)"

            record = list(map(lambda x: session.run(cypher % x), pid_list))
            result = list(
                map(lambda y: list(map(lambda x: x[0], y))[0], record))
        return result
```

### 2.6.2 用户操作行为API

用户点赞API、用户评论API、用户转发API、用户取消点赞API、用户删除评论API。我们会对用户的操作进行数据库的节点增删改查。主要是对用户的行为关系进行修改

```python
@api_view(['GET', 'POST'])
def like(request):
    IP = request.META.get("HTTP_X_REAL_IP")
    pid = request.POST.get("pid")
    type_ = "like"
    result = api.write_to_neo4j(IP, pid, type_)
    return Response(result)


@api_view(['GET', 'POST'])
def forward(request):
    IP = request.META.get("HTTP_X_REAL_IP")
    pid = request.POST.get("pid")
    type_ = "forward"
    result = api.write_to_neo4j(IP, pid, type_)
    return Response(result)


@api_view(['GET', 'POST'])
def comment(request):
    IP = request.META.get("HTTP_X_REAL_IP")
    pid = request.POST.get("pid")
    content = request.POST.get("content")
    type_ = "comment"
    result = api.write_to_neo4j(IP, pid, type_, content)


@api_view(['GET', 'POST'])
def cancel_like(request):
    IP = request.META.get("HTTP_X_REAL_IP")
    pid = request.POST.get("pid")
    type_ = "like"
    result = api.cancel_to_neo4j(IP, pid, type_)
```

####2.6.2.1 neo4j数据库读写逻辑代码实现

* 1、用户点赞后写数据库，我们将匹配用户节点，和帖子节点，并在两者之间合并一条类似类型的边，合并是指：存在则匹配，不存在则创建。
* 2、用户评论后的写数据库，同样匹配用户和帖子节点，并在两者之间创建一条表示类型的边，并且为边置于时间和内容属性。以便在删除操作时进行索引。
* 3、用户转发操作的写数据库，同样匹配用户和帖子节点，并在两者之间创建一条共享类型的边。
* 4、用于用户的取消点赞的操作，匹配用户和帖子节点，并删除他们之间像类型的边。
* 5、用于用户的删除评论操作，匹配用户和帖子节点，根据时间戳删除指定的赞扬类型的边。

```python
cypher1 = "MATCH(a:{uid:%d}) MATCH(b:SuperfansPost{pid:%d}) with a,b MERGE(a)-[r:%s]-(b)"
cypher2 = "MATCH(a:SuperfansUser{uid:%d}) MATCH(b:SuperfansPost{pid:%d}) with a,b CREATE(a)-[r:%s]-(b) set r.time=%d, r.content=%s"
cypher3 = "MATCH(a:SuperfansUser{uid:%d}) MATCH(b:SuperfansPost{pid:%d}) with a,b CREATE(a)-[r:%s]-(b)"
cypher4 = "MATCH(a:SuperfansUser{uid:%d})-[r:%s]-(b:SuperfansPost{pid:%d}) delete r"
cypher5 = "MATCH(a:SuperfansUser{uid:%d})-[r:%s]-(b:SuperfansPost{pid:%d}) where r.time=%d delete r
```

1、写入数据行为逻辑

```python
r_result = {"msg":"Success", "code":1}
f_result = {"msg": "Fail", "code":0}

def write_to_neo4j(IP, pid, type_, content=""):
    """写入行为类型到neo4j数据库
    :param IP: 用户IP
    :param pid: 帖子ID
    :param type_: 行为类型
    :param content: 内容（评论）
    :return:
    """
    r = redis.StrictRedis(connection_pool=pool)
    uid = r.get("u_" + IP)
    # 如果类型是喜欢，写入喜欢关系
    if type_ == "like":
        # 判断是否存在该用户，存在写入，返回成功，不存在返回失败
        if uid:
            with _driver.session() as session:
                session.run(cypher1 % (int(uid), int(pid), type_))
                return r_result
        else:
            return f_result
    elif type_ == "comment":
        if uid:
            time_ = int(time.time())
            with _driver.session() as session:
                session.run(cypher2 % (int(uid), int(pid), type_, time_, content))
                return r_result
        else:
            return f_result
    else:
        if uid:
            with _driver.session() as session:
                session.run(cypher3 % (int(uid), int(pid), type_))
            return r_result
        else:
            return f_result
```

2、取消行为类型

```python
def cancel_to_neo4j(IP, pid, type_, time_):
    """
    删除行为类型
    :param IP: 用户IP
    :param pid: 帖子ID
    :param type_: 行为类型
    :param time_: 时间戳
    :return:
    """
    r = redis.StrictRedis(connection_pool=pool)
    uid = r.get("u_" + IP)
    if type_ == "like":
        if uid:
            with _driver.session() as session:
                session.run(cypher4 %(int(uid), type_, int(pid)))
                return r_result
        else:
            return f_result
    # 如果是评论取消
    elif type_ == "comment":
        if uid:
            # 根据时间戳删除
            with _driver.session() as session:
                session.run(cypher5 %(int(uid),type_, int(pid), time_))
                return r_result
        else:
            return f_result
```

### 2.6.3 小结

* 用户推荐缓存获取代码

* 用户行为数据neo4j行为读写操作