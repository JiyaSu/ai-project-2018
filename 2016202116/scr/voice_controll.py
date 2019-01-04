#!/usr/bin/env python
#-*- coding:utf-8 -*-
import serial 
import threading 
import time 
from aip import AipSpeech


""" 你的 APPID AK SK """
APP_ID = '10760396'
API_KEY = 'xTaIuxBBLU3LvwZPKjQWCf0B'
SECRET_KEY = '70d7aa785081ed00ae7c808691df8b62'

client = AipSpeech(APP_ID, API_KEY, SECRET_KEY)



def get_file_content(filePath):
    with open(filePath, 'rb') as fp:
        return fp.read()

def context():
    res=client.asr(get_file_content(r'C:\Users\86135\Desktop\AIcar\go.wav'), 'wav', 16000, {'dev_pid': 1536,})
    re=res.get('result','无')
    for i in re[0]:
        print(i)
        if i=='前'or i=='直':
            return '1'
        elif i=='后':
            return '2'
        elif i=='左':
            return '5'
        elif i=='右':
            return '6'
        elif i=='停':
            return '7'
        elif i=='终':
            return '8'
    return '0'       




if __name__ == '__main__':

    ser = serial.Serial("COM4", 9600, timeout=0.5)
    print (ser.name)
    print (ser.port)

    if ser.isOpen():
        print("succeed")
    else:
        print("fail")


    d=context()
    print(d)

    while d <'7':
        ser.write(d.encode())
        ser.readline()
        d=context()
        print(d)










#ser.write('1'.encode())
#st=ser.readline()



#ser.write('2'.encode())
#st=ser.readline()

#ser.write('3'.encode())
#st=ser.readline()

#ser.write('4'.encode())
#st=ser.readline()

#print(st)

