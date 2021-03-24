# 2.1 Web接口业务流介绍

## 学习目标

- 目标
  - 知道用户推荐接口运行流程
- 应用
  - 无

### 2.1.1 Web端环境启动

我们这里使用supervisor进程管理工具进行管理Web服务的启动，nginx+uwsgi(django)是web端的服务解决方案

#### 2.1.1.1 supervisor启动

配置supervisor要启动的进程服务：3个服务

* 1、uwsgi服务

```python
[program:main_server]
command=uwsgi --ini /home/zhoumingzhen/conf/uwsgi.ini --close-on-exec
stopsignal=QUIT               ; signal used to kill process (default TERM)
stopasgroup=false             ; send stop signal to the UNIX process group (default false)
killasgroup=false             ; SIGKILL the UNIX process group (def false)
stdout_logfile=/home/zhoumingzhen/log/main_server_out.log
stdout_logfile_maxbytes=1MB   ; max # logfile bytes b4 rotation (default 50MB)
stderr_logfile=/home/zhoumingzhen/log/main_server_err.log
stderr_logfile_maxbytes=1MB   ; max # logfile bytes b4 rotation (default 50MB)
```

* 2、启动nginx的web服务，在start_nginx.ini启动配置中

```ini
[program:nginx]
command=/usr/sbin/nginx -c /home/zhoumingzhen/conf/nginx/nginx.conf -g "daemon off;"
stdout_logfile=/home/zhoumingzhen/log/nginx_out.log
stderr_logfile=/home/zhoumingzhen/log/nginx_err.log
stdout_logfile_maxbytes=1MB
stderr_logfile_maxbytes=1MB
```

* 3、启动redis相关服务

```ini
[program:redis]
command=redis-server /home/zhoumingzhen/conf/redis.conf
stdout_logfile=/home/zhoumingzhen/log/redis_out.log
stderr_logfile=/home/zhoumingzhen/log/redis_err.log
stdout_logfile_maxbytes=1MB
stderr_logfile_maxbytes=1MB
```

### 2.1.2 后台业务逻辑

围绕着APP设计中，用户可能进行的那些操作，以及在什么地方提供对用户的推荐接口。

- API总览:
  - 用于用户获取推荐：
    - 首次推荐API
    - 个性化推荐API
  - 推荐结果操作行为API：
    - 用户点赞API
    - 用户评论API
    - 用户转发API
    - 用户取消点赞API
    - 用户删除评论API

* 视图函数逻辑

相关django框架模块，以及推荐模块导入

```python
from django.http import HttpResponse
from rest_framework import viewsets
from rest_framework.response import Response
from rest_framework.decorators import api_view
from rest_framework.authentication import SessionAuthentication, BasicAuthentication
from rest_framework.permissions import IsAuthenticated
from rest_framework.decorators import authentication_classes
from rest_framework.decorators import permission_classes
from . import api
from recomm import api as r_api
import json
import logging
```

- 实现路由：

```python
from django.conf.urls import url
from django.contrib import admin
from . import views

urlpatterns = [
    url(r'^api/first_show[/]?$', views.first_show), # 用户第一次请求
    url(r'^api/get_cache[/]?$', views.get_cache), # 用户请求，获取缓存结果
    url(r'^api/like[/]?$', views.like),
    url(r'^api/forward[/]?$', views.forward),
    url(r'^api/commend[/]?$', views.commend)
    url(r'^api/cancel_like[/]?$', views.cancel_like),
    url(r'^api/delete_commend[/]?$', views.delete_commend)
]
```

#### 2.1.2.1 接口演示

访问对应接口可以获得推荐结果：

http://211.103.136.242/api/get_cache?nsukey=UUZr%2BpRKZ%2B%2BWB1E3esC2%2FNXU9D9hZYDv04OMUrpJDGblaTaOROg%2FqxptH%2F%2FJIcrB9h4j72WcFGnxgsbMH%2BYtprjYxdd83ESgV1v3k07LnGPFJ67EEkiTX%2FLMTljmFlCsSAkypx8mLqlIw%2FFFLaczLJoyOLBWM3BxrYxqaZkTJWlewON300Nw4zGVu0RPVIZv2BTHldCLAVXn0jvXAYZ6zw%3D%3D

```
[{"hot_score": 0, "text_info": "😉", "iv_url": "http://p.upcdn.pengpengla.com/star/p/u/2017/8/10/2f452e28-3e63-48ed-87e3-b7e4ad508785.jpg", "publish_time": 1502339444, "commented_num": 0, "liked_num": 0, "forwarded_num": 0, "pid": 675117, "related_stars_list": ["2", "4", "5"]}, {"hot_score": 71, "text_info": "不愿让暴暴受伤害的铲屎官……", "iv_url": "http://p.upcdn.pengpengla.com/star/p/u/2017/7/4/63563717-4e4d-4500-8bf3-5d4e332132fd.png", "publish_time": 1499161824, "commented_num": 0, "liked_num": 71, "pid": 690410, "forwarded_num": 0, "related_stars_list": ["2", "4", "5"]}, {"hot_score": 108, "text_info": "这几张最帅好吗", "iv_url": "http://p.upcdn.pengpengla.com/star/p/u/2017/8/7/a97755a5-6867-49bf-8525-ab932fcc2006.jpg", "publish_time": 1502088406, "commented_num": 0, "liked_num": 108, "pid": 674069, "forwarded_num": 0, "related_stars_list": ["2", "4", "5"]}, {"hot_score": 72, "text_info": "大爱谦谦", "iv_url": "http://p.upcdn.pengpengla.com/star/p/u/2017/8/7/795b5e28-ca55-477a-80be-321748f01aa1.jpg", "publish_time": 1502090884, "commented_num": 0, "liked_num": 72, "pid": 674085, "forwarded_num": 0, "related_stars_list": ["2", "4", "5"]}, {"hot_score": 20, "text_info": "李东学语音素材第一版回顾", "iv_url": "http://g.cdn.pengpengla.com/starfantuan/fanquan/149760691453.mp3", "publish_time": 1497606612, "commented_num": 2, "liked_num": 16, "pid": 260409, "forwarded_num": 0, "related_stars_list": ["2", "4", "5"]}]
```

#### 2.1.2.2 用户推荐接口介绍

主要由两个通过用户首次推荐和个性化推荐的接口

- first_show

用户第一次进行请求，获取热门召回推荐

```python
@api_view(['GET', 'POST'])
def first_show(request):
    IP = request.META.get("HTTP_X_REAL_IP")
    result = r_api.get_hot(str(IP))
    return HttpResponse(json.dumps(result, ensure_ascii=False))
```

* get_cache

根据用户请求，获取用户缓存结果（进行用户是否第一次使用某IP登陆判断）

```python
@api_view(['GET', 'POST'])
def get_cache(request):
    IP = request.META.get("HTTP_X_REAL_IP")
    result = r_api.v_get_cache(str(IP))
    return HttpResponse(json.dumps(result, ensure_ascii=False))
```

* get_recomm

获取用户推荐结果（直接获取一次召回推荐结果）

```python
@api_view(['GET', 'POST'])
def get_recomm(request):
    IP = request.META.get("HTTP_X_REAL_IP")
    result = r_api._get_recomm(str(IP))
    return HttpResponse(json.dumps(result, ensure_ascii=False))
```

其中都会通过from recomm import api as r_api这个包的相关函数进行推荐，我们推荐逻辑主要都在recomm模块中，这是自定义命名的，当做推荐模块使用。

### 2.1.3 小结

* Web端环境启动
  * uwsgi服务
  * nginx服务
  * redis服务
* 用户推荐接口
  * first_show、get_cache

