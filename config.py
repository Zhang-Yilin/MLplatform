#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import gevent.monkey
import multiprocessing
 
gevent.monkey.patch_all()

#监听本机的5000端口
bind='0.0.0.0:5000'
 
preload_app = True

#进程
workers = 5
 
#线程
threads = 10
 
backlog=2048
 
#工作模式为gevent
worker_class="gevent"
 
# debug=True
 
#如果不使用supervisord之类的进程管理工具可以是进程成为守护进程，否则会出问题
daemon = True
 
#进程名称
proc_name='gunicorn.pid'
 
#进程pid记录文件
pidfile='app_pid.log'
 
loglevel='debug'
logfile = 'debug.log'
accesslog = 'access.log'
access_log_format = '%(h)s %(t)s %(U)s %(q)s'


