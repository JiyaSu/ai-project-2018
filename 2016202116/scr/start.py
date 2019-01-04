import wave
import threading
import tkinter
import tkinter.filedialog
import tkinter.messagebox

import pyaudio

import wave
import numpy as np
import math
from scipy import signal
from scipy.fftpack import dct
from sklearn.externals import joblib
from sklearn.mixture import GaussianMixture
import serial
import threading
import time
import os

#API##################################################################################

from aip import AipSpeech

""" 你的 APPID AK SK """
APP_ID = '10760396'
API_KEY = 'xTaIuxBBLU3LvwZPKjQWCf0B'
SECRET_KEY = '70d7aa785081ed00ae7c808691df8b62'

client = AipSpeech(APP_ID, API_KEY, SECRET_KEY)


d=0
first=0
allowRecording = False

def readWAV(wave_path):  # 读wav文件
    f = wave.open(wave_path, "rb")
    params = f.getparams()
    nchannels, sampwidth, framerate, nframes = params[:4]  # 切片，取下标为0-3的四个元素
    # print (nchannels, sampwidth, framerate, nframes,'\n')

    str_data = f.readframes(nframes)  # 读取音频，字符串格式
    f.close()
    wave_data = np.fromstring(str_data, dtype=np.short)  # 将字符串转化为int数组
    # wave_data = wave_data.T   #转置
    time = np.arange(0, nframes) * (1.0 / framerate)  # arange用于创建等差数组

    return framerate, nframes, wave_data, time


def enframe(wave_data, nw, inc, winfunc):   #分帧并加窗
    '''将音频信号转化为帧。
    参数含义：
    wave_data:原始音频型号
    nw:每一帧的长度(这里指采样点的长度，即采样频率乘以时间间隔)
    inc:相邻帧的间隔（同上定义）
    '''
    wlen=len(wave_data) #信号总长度
    # print('信号长',wlen,'帧长',nw,'移动步长',inc)
    if wlen<=nw: #若信号长度小于一个帧的长度，则帧数定义为1
        nf=1
    else: #否则，计算帧的总长度
        nf=int(np.ceil((1.0*wlen-nw+inc)/inc))
        # print('帧数',nf)
    pad_length=int((nf-1)*inc+nw) #所有帧加起来总的铺平后的长度
    zeros=np.zeros((pad_length-wlen,)) #不够的长度使用0填补，类似于FFT中的扩充数组操作
    pad_signal=np.concatenate((wave_data,zeros)) #填补后的信号记为pad_signal
    indices=np.tile(np.arange(0,nw),(nf,1))+np.tile(np.arange(0,nf*inc,inc),(nw,1)).T  #相当于对所有帧的时间点进行抽取，得到nf*nw长度的矩阵
    indices=np.array(indices,dtype=np.int32) #将indices转化为矩阵
    frames=pad_signal[indices] #得到帧信号
    #print(frames)
    win=np.tile(winfunc,(nf,1))  #window窗函数，这里默认取1
    return frames*win   #返回帧信号矩阵


def mfcc(wave_data, framerate, nframes):  # mfcc特征提取
    # 预加重
    # b,a = signal.butter(1,1-0.97,'high')
    # emphasized_signal = signal.filtfilt(b,a,wave_data)
    # 归一化倒谱提升窗口
    wave_data = wave_data * 1.0 / (max(abs(wave_data)))  # 音频信号归一化
    lifts = []
    for n in range(1, 13):
        lift = 1 + 6 * np.sin(np.pi * n / 12)
        lifts.append(lift)
    # print(lifts)

    # 分帧、加窗
    nw = int(np.ceil(0.025 * framerate))  # 帧的长度
    inc = int(np.ceil(0.01 * framerate))
    winfunc = signal.hamming(nw)
    X = enframe(wave_data, nw, inc, winfunc)  # 转置的原因是分帧函数enframe的输出矩阵是帧数*帧长
    frameNum = X.shape[0]  # 返回矩阵行数18，获取帧数
    mfccs = []
    for i in range(frameNum):
        y = X[i, :]
        # fft
        yf = np.abs(np.fft.fft(y))
        # print(yf.shape)
        # 谱线能量
        yf = yf ** 2
        # 梅尔滤波器系数
        nfilt = 24
        low_freq_mel = 0
        NFFT = 256
        high_freq_mel = (2595 * np.log10(1 + (framerate / 2) / 700))  # 把 Hz 变成 Mel
        mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # 将梅尔刻度等间隔
        hz_points = (700 * (10 ** (mel_points / 2595) - 1))  # 把 Mel 变成 Hz
        bin = np.floor((NFFT + 1) * hz_points / framerate)
        fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
        for m in range(1, nfilt + 1):
            f_m_minus = int(bin[m - 1])  # left
            f_m = int(bin[m])  # center
            f_m_plus = int(bin[m + 1])  # right
            for k in range(f_m_minus, f_m):
                fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
            for k in range(f_m, f_m_plus):
                fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
        filter_banks = np.dot(yf[0:129], fbank.T)
        filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # 数值稳定性
        filter_banks = 10 * np.log10(filter_banks)  # dB
        filter_banks -= (np.mean(filter_banks, axis=0) + 1e-8)
        # print(filter_banks)
        # DCT系数
        num_ceps = 12
        c2 = dct(filter_banks, type=2, axis=-1, norm='ortho')[1: (num_ceps + 1)]  # Keep 2-13
        c2 *= lifts
        # print(c2)
        mfccs.append(c2)
    mfccs = np.array(mfccs, dtype=np.float)
    wave_data = wave_data * (max(abs(wave_data)))  # 还原信号
    # print(mfccs)
    # print(mfccs.shape)
    return mfccs

def owner():
    test_path=r'C:\Users\86135\Desktop\AIcar\go.wav'
    result=11
    Pmin=float("inf")
    #print('test file', test_path)
    framerate, nframes, wdata, time = readWAV(test_path)
    # print ('采样频率：',framerate,'\n')
    # print ('采样点数: ',nframes,'\n')
    test = mfcc(wdata, framerate, nframes)
    test = np.array(test, dtype=np.float)
    print(test.shape)
    for i in range(1,4):
        Pt=1.0  #the probability of a wave file
        model_path = "model" +str(i) + ".m"
        gmm=joblib.load(model_path)
        for frameMFCC in test:
            P = gmm.score_samples([frameMFCC])[0]
            P=-1*(1*P)/10   #寻找-log(P)的最小值,除以10是为了方便计算,防止数据溢出
            Pt=Pt*P  #由事件概率可知整个音频属于某个数字的概率由其每帧属于某个数字的概率相乘得到
        # print("P is here:",P)
        if Pt<=Pmin:
            result=i
            Pmin=Pt
    return result

def context():
    res=client.asr(get_file_content(r'C:\Users\86135\Desktop\AIcar\go.wav'), 'wav', 16000, {'dev_pid': 1536,})
    re=res.get('result','无')
    print(re)
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
        elif i=='听':
            return '7'
        elif i=='退':
            return '8'
        elif i=='畅':
            return '9'
    return '0'       




def record():
    CHUNK_SIZE = 1024
    CHANNELS = 1
    FORMAT = pyaudio.paInt16
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=16000,
                    input=True,
                    frames_per_buffer=CHUNK_SIZE)

    fileName = r'C:\Users\86135\Desktop\AIcar\go.wav'
    wf = wave.open(fileName, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(16000)

    while allowRecording:
        data = stream.read(CHUNK_SIZE)
        wf.writeframes(data)
    wf.close()

    stream.stop_stream()
    stream.close()
    p.terminate()
    print("record sucees")
	#调API部分

   
    d=context()
    print(owner())
#    global first
    if d== '9' and  owner()==2:
        print("unlock")
        os.system(r'start python1 C:\Users\86135\Desktop\AIcar\voice_controll.py')
#    else:
#         print(owner())
#    if first==2:
#        os.system(r'start python1 C:\Users\86135\Desktop\SINGLE.py')
#        first=0
    if d=='7':
        print("listening")
#        first=2
        os.system(r'start python1 C:\Users\86135\Desktop\AIcar\SINGLE.py')
#        first=0
   

 #   d=context();



def start():
    global allowRecording

    allowRecording = True
    lbStatus['text'] = 'recording...'
    threading.Thread(target=record).start()


def stop():
    global allowRecording
    allowRecording = False
    lbStatus['text'] = 'ready'


def play():
    CHUNK_SIZE = 2000
    CHANNELS = 1
    FORMAT = pyaudio.paInt16
    #RATE = 44100
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=16000,
                    output=True)

    fileName =r'C:\Users\86135\Desktop\AIcar\go.wav'
    wf = wave.open(fileName, 'rb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)

    while True:
        data = wf.readframes(CHUNK_SIZE)
        if data == "":
            break
        stream.write(data)
    stream.close()
    p.terminate()

def closeWindow():
    if allowRecording:
        tkinter.messagebox.showerror('is reacording', 'please end recording')
        return
    root.destroy()


# 读取文件
def get_file_content(filePath):
    with open(filePath, 'rb') as fp:
        return fp.read()
		
		
if __name__ == '__main__':
	#录音部分  保存在test.wav
    root = tkinter.Tk()
    root.title("demo")
    root.geometry('280x80+400+300')
    root.resizable(False, False)

    lbStatus = tkinter.Label(root, text='ready', anchor='w', fg='green')
    lbStatus.place(x=30, y=50, width=200, height=20)

    btnStart = tkinter.Button(root, text='start', command=start)
    btnStart.place(x=30, y=20, width=60, height=20)

    btnStop = tkinter.Button(root, text='end', command=stop)
    btnStop.place(x=85, y=20, width=60, height=20)

    btnPlay = tkinter.Button(root, text='play', command=play)
    btnPlay.place(x=140, y=20, width=60, height=20)

    root.protocol('WM_DELETE_WINDOW', closeWindow)
    root.mainloop()


