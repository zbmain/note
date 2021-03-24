# 1.3 开发环境介绍

## 学习目标

- 目标
  - 了解泛娱乐推荐系统的开发环境组成
- 应用
  - 无

### 1.3.1 整体环境介绍

- 基础组件:
  - 后台服务: Django, uwsgi, supervisor, Nginx
  - 数据库服务: Neo4j, Redis
  - AI训练: tensorflow, gcp ML-engine（GCP）
  - AI预测: gcp ML-engine（GCP）

### 1.3.2 搭建步骤

- 1、使用服务器系统:centos7, 安装必备环境包

```python
yum install supervisor
yum install nginx
yum install redis
```

还有Centos 安装neo4j图数据库安装与使用。请参见：http://yum.neo4j.org/stable/, yum install neo4j-3.3.5

- 2、创建一个虚拟环境安装:

```python
conda create -n recreation python=3.6
source activate recreation

# 安装的包, 在requirements.txt文件中，pip install -r requirements.txt
Django>=1.11.7
djangorestframework>=3.7.3
django-filter>=1.1.0
flower>=0.9.2
requests>=2.18.4
django-cors-headers
uwsgi
neo4j-driver==1.7.2
numpy
redis
```

### 1.3.3 启动配置文件介绍

因为我们这里需要将web服务以及数据库redis服务启动，才能后续对接推荐系统接口。

这里介绍的是配置nginx+django,以及redis启动服务。使用supervisor进行管理的三个服务。整体需要启动的服务

#### django后台服务

- 1、uwsgi配置在项目config目录下

uwsgi配置文件uwsgi.ini:

```
#uWSGI configuration
[uwsgi]
#chdir=
http=0.0.0.0:5000
module=server.wsgi
master=True
vacuum=True
max-requests=5000
# can also be a file
socket=0.0.0.0:3001
processes=2
#wsgi-file=server/wsgi.py
pcre=True
#pidfile=/tmp/project-master.pid
#daemonize=/var/log/uwsgi/yourproject.log
```

配置分析: 设置监听http协议的5000端口来替代django自带runserver测试级服务，Django配置：配置文件setting.py在server目录下

* 2、nginx配置文件nginx.conf在conf目录下
  * 配置分析:设置监听8087端口作为对外服务端口，并反向代理至uwsgi的5000端口，同时可根据使用现状进行合适的负载均衡

```
#user  nobody;
worker_processes  4;

error_log   /home/zhoumingzhen/log/nginx/error.log;
pid         /home/zhoumingzhen/log/nginx/nginx.pid;


events {
    worker_connections  1024;
}


http {
    include       mime.types;
    default_type  application/octet-stream;

    sendfile        on;
    #keepalive_timeout  0;
    keepalive_timeout  65;

    #gzip  on;
    upstream test {
        server 0.0.0.0:5000;
        #server 0.0.0.0:8718 weight=1;
        #server 0.0.0.0:8000 weight=1;
     }

    server {
        listen       8087;
        server_name  0.0.0.0;

        #charset koi8-r;

        #access_log  log/host.access.log  main;

        location /static/ {
            alias /home/zhoumingzhen/static/;
        }

        location / {
            proxy_pass     http://test;
            #uwsgi_pass      127.0.0.1:3001;
            include      /home/zhoumingzhen/conf/nginx/uwsgi_params;

            proxy_set_header X-Real-IP $remote_addr;
        }

        error_page   500 502 503 504  /50x.html;
        location = /50x.html {
                    root   html;
        }

    include servers/*;
}
```

####supervisor启动配置

supervisor配置文件supervisor.conf，执行命令

```shell
supervisord -c /root/recreation_project/supervisord.conf
```

* 配置分析:

1、开启http9001端口作为可视化监控界面的访问端口
2、监控uwsgi, nginx, redis等组件, 并指定日志打印位置和容量限制

```python
[unix_http_server]
file=/tmp/supervisor.sock   ; the path to the socket file

[inet_http_server]          ; inet (TCP) server disabled by default
port=0.0.0.0:9001           ; ip_address:port specifier, *:port for all iface


[supervisord]
logfile=/root/recreation_project/log/supervisord.log ; main log file; default $CWD/supervisord.log
logfile_maxbytes=50MB        ; max main logfile bytes b4 rotation; default 50MB
logfile_backups=10           ; # of main logfile backups; 0 means none, default 10
loglevel=info                ; log level; default info; others: debug,warn,trace
pidfile=/root/recreation_project/log/supervisord.pid ; supervisord pidfile; default supervisord.pid
nodaemon=false               ; start in foreground if true; default false
minfds=1024                  ; min. avail startup file descriptors; default 1024
minprocs=200                 ; min. avail process descriptors;default 200

[rpcinterface:supervisor]
supervisor.rpcinterface_factory = supervisor.rpcinterface:make_main_rpcinterface


[supervisorctl]
serverurl=unix:///tmp/supervisor.sock ; use a unix:// URL  for a unix socket
serverurl=http://0.0.0.0:9001 ; use an http:// url to specify an inet socket

[program:main_server]
command=uwsgi --ini /root/recreation_project/conf/uwsgi.ini --close-on-exec
stopsignal=QUIT               ; signal used to kill process (default TERM)
stopasgroup=false             ; send stop signal to the UNIX process group (default false)
killasgroup=false             ; SIGKILL the UNIX process group (def false)
stdout_logfile=/root/recreation_project/log/main_server_out.log       ; stdout log path, NONE for none; default AUTO
stdout_logfile_maxbytes=1MB   ; max # logfile bytes b4 rotation (default 50MB)
stderr_logfile=/root/recreation_project/log/main_server_err.log        ; stderr log path, NONE for none; default AUTO
stderr_logfile_maxbytes=1MB   ; max # logfile bytes b4 rotation (default 50MB)

[include]
files = /root/recreation_project/supervisord.conf.d/*.ini

```

#### nginx的启动配置

/root/recreation_project/supervisord.conf.d/start_nginx.ini

```
[program:nginx]
command=/usr/sbin/nginx -c /root/recreation_project/conf/nginx/nginx.conf -g "daemon off;"
stdout_logfile=/root/recreation_project/log/nginx_out.log
stderr_logfile=/root/recreation_project/log/nginx_err.log
stdout_logfile_maxbytes=1MB
stderr_logfile_maxbytes=1MB
```

#### redis数据库服务

* 配置redis的启动命令输出日志位置

/root/recreation_project/supervisord.conf.d/start_redis.ini

```
[program:redis]
command=redis-server /root/recreation_project/conf/redis.conf
stdout_logfile=/root/recreation_project/log/redis_out.log
stderr_logfile=/root/recreation_project/log/redis_err.log
stdout_logfile_maxbytes=1MB
stderr_logfile_maxbytes=1MB
```

### 1.3.4 小结

* 开发环境组件
  * 后台：django+uwsgi+nginx+supervisor
  * 数据库服务：redis+neo4j
  * 模型训练与预测：GCP
* 环境结构
  * 安装系统环境包与python虚拟化境包
  * nginx+uwsgi配置，supervisor三个启动服务的配置