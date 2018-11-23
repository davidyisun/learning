#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    脚本名: 串口编程
    原文：https://blog.csdn.net/zjsxxzh/article/details/53726083
Created on 2018-11-20
@author:David Yisun
@group:data
"""
import serial
import threading
import time
x=serial.Serial('com12',9600)
# i=0
def fasong():
    while True:
        #print("wocao")
        myinput=input('shuru>')
        myinput=myinput.encode(encoding="utf-8")
        #print("you input "+myinput)
        #time.sleep(1)
        x.write(myinput)


def jieshou():
    myout=""
    while True:
       while x.inWaiting()>0:
           myout+=x.read(1).decode()
       if myout!="":
            print(myout)
            myout=""
       #myout=x.read(14)
      # myout="lll"
       #time.sleep(1)

if __name__== '__main__':
     t1 = threading.Thread(target=jieshou,name="jieshou")
     t2= threading.Thread(target=fasong, name="fasong")
     t2.start()
     t1.start()
