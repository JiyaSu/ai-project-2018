import numpy as np
from sklearn.externals import joblib
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

#feature get

import os
import numpy
import pydub  # 似乎可去掉 第24行
from scipy.fftpack import fft
from scipy.fftpack.realtransforms import dct

#特征提取start

import numpy as np
from sklearn.externals import joblib
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import os
import numpy
import pydub

import serial
import threading 
import time 

import wave
from pyaudio import PyAudio,paInt16

'''
from pyAudioAnalysis import audioFeatureExtraction
import time
import os
import glob
import numpy
import math
from scipy.fftpack import fft
from scipy.fftpack.realtransforms import dct
import matplotlib.pyplot as plt
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import utilities
from scipy.signal import lfilter
'''

from scipy.fftpack import fft
from scipy.fftpack.realtransforms import dct


framerate=44100
NUM_SAMPLES=2000
channels=1
sampwidth=2
TIME=2
def save_wave_file(filename,data):
    '''save the date to the wavfile'''
    wf=wave.open(filename,'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(sampwidth)
    wf.setframerate(framerate)
    wf.writeframes(b"".join(data))
    wf.close()

def my_record():
    pa=PyAudio()
    stream=pa.open(format = paInt16,channels=1,
                   rate=framerate,input=True,
                   frames_per_buffer=NUM_SAMPLES)
    my_buf=[]
    count=0
    while count<TIME*40:   #控制录音时间
        string_audio_data = stream.read(NUM_SAMPLES)
        my_buf.append(string_audio_data)
        count+=1
#        print('.')
    save_wave_file(r'C:\Users\86135\Desktop\AIcar\voice.wav',my_buf)
    stream.close()

chunk=2014
def play():
    wf=wave.open(r'C:\Users\86135\Desktop\AIcar\voice.wav','rb')
    p=PyAudio()
    stream=p.open(format=p.get_format_from_width(wf.getsampwidth()),channels=
    wf.getnchannels(),rate=wf.getframerate(),output=True)
    while True:
        data=wf.readframes(chunk)
        if data=="":break
        stream.write(data)
    stream.close()
    p.terminate()

def record():
    # while True:
    isPass = input("开始录音（时间5秒）：")
    if isPass == '1':
        print("录音")
        my_record()
        print("录音识别")

def readAudioFile(path):
    '''
    This function returns a numpy array that stores the audio samples of a specified WAV of AIFF file
    '''
    extension = os.path.splitext(path)[1]

    try:
        # if extension.lower() == '.wav':
        # [Fs, x] = wavfile.read(path)
        if extension.lower() == '.mp3' or extension.lower() == '.wav' or extension.lower() == '.au' or extension.lower() == '.ogg':
            try:
                audiofile = pydub.AudioSegment.from_file(path)
            # except pydub.exceptions.CouldntDecodeError:
            except:
                print("Error: file not found or other I/O error. "
                      "(DECODING FAILED)")
                return (-1, -1)

            if audiofile.sample_width == 2:
                data = numpy.frombuffer(audiofile._data, numpy.int16)
            elif audiofile.sample_width == 4:
                data = numpy.fromstring(audiofile._data, numpy.int32)
            else:
                return (-1, -1)
            Fs = audiofile.frame_rate
            x = []
            for chn in list(range(audiofile.channels)):
                x.append(data[chn::audiofile.channels])
            x = numpy.array(x).T
        else:
            print("Error in readAudioFile(): Unknown file type!")
            return (-1, -1)
    except IOError:
        print("Error: file not found or other I/O error.")
        return (-1, -1)

    if x.ndim == 2:
        if x.shape[1] == 1:
            x = x.flatten()

    return (Fs, x)


eps = 0.00000001


def stZCR(frame):
    """Computes zero crossing rate of frame"""
    count = len(frame)
    countZ = numpy.sum(numpy.abs(numpy.diff(numpy.sign(frame)))) / 2
    return (numpy.float64(countZ) / numpy.float64(count - 1.0))


def stEnergy(frame):
    """Computes signal energy of frame"""
    return numpy.sum(frame ** 2) / numpy.float64(len(frame))


def mfccInitFilterBanks(fs, nfft):
    """
    Computes the triangular filterbank for MFCC computation
    (used in the stFeatureExtraction function before the stMFCC function call)
    This function is taken from the scikits.talkbox library (MIT Licence):
    https://pypi.python.org/pypi/scikits.talkbox
    """

    # filter bank params:
    lowfreq = 133.33
    linsc = 200 / 3.
    logsc = 1.0711703
    numLinFiltTotal = 13
    numLogFilt = 27

    if fs < 8000:
        nlogfil = 5

    # Total number of filters
    nFiltTotal = numLinFiltTotal + numLogFilt

    # Compute frequency points of the triangle:
    freqs = numpy.zeros(nFiltTotal + 2)
    freqs[:numLinFiltTotal] = lowfreq + numpy.arange(numLinFiltTotal) * linsc
    freqs[numLinFiltTotal:] = freqs[numLinFiltTotal - 1] * logsc ** numpy.arange(1, numLogFilt + 3)
    heights = 2. / (freqs[2:] - freqs[0:-2])

    # Compute filterbank coeff (in fft domain, in bins)
    fbank = numpy.zeros((nFiltTotal, nfft))
    nfreqs = numpy.arange(nfft) / (1. * nfft) * fs

    for i in range(nFiltTotal):
        lowTrFreq = freqs[i]
        cenTrFreq = freqs[i + 1]
        highTrFreq = freqs[i + 2]

        lid = numpy.arange(numpy.floor(lowTrFreq * nfft / fs) + 1,
                           numpy.floor(cenTrFreq * nfft / fs) + 1,
                           dtype=numpy.int)
        lslope = heights[i] / (cenTrFreq - lowTrFreq)
        rid = numpy.arange(numpy.floor(cenTrFreq * nfft / fs) + 1,
                           numpy.floor(highTrFreq * nfft / fs) + 1,
                           dtype=numpy.int)
        rslope = heights[i] / (highTrFreq - cenTrFreq)
        fbank[i][lid] = lslope * (nfreqs[lid] - lowTrFreq)
        fbank[i][rid] = rslope * (highTrFreq - nfreqs[rid])

    return fbank, freqs


def stChromaFeaturesInit(nfft, fs):
    """
    This function initializes the chroma matrices used in the calculation of the chroma features
    """
    freqs = numpy.array([((f + 1) * fs) / (2 * nfft) for f in range(nfft)])
    Cp = 27.50
    nChroma = numpy.round(12.0 * numpy.log2(freqs / Cp)).astype(int)

    nFreqsPerChroma = numpy.zeros((nChroma.shape[0],))

    uChroma = numpy.unique(nChroma)
    for u in uChroma:
        idx = numpy.nonzero(nChroma == u)
        nFreqsPerChroma[idx] = idx[0].shape

    return nChroma, nFreqsPerChroma


def stEnergyEntropy(frame, n_short_blocks=10):
    """Computes entropy of energy"""
    Eol = numpy.sum(frame ** 2)  # total frame energy
    L = len(frame)
    sub_win_len = int(numpy.floor(L / n_short_blocks))  # cannot reshape array of size 4400 into shape (220,10)
    if L != sub_win_len * n_short_blocks:
        frame = frame[0:sub_win_len * n_short_blocks]
    # sub_wins is of size [n_short_blocks x L]
    sub_wins = frame.reshape(sub_win_len, n_short_blocks,
                             order='F').copy()  # 原为sub_wins = frame.reshape(sub_win_len, n_short_blocks, order='F').copy()

    # Compute normalized sub-frame energies:
    s = numpy.sum(sub_wins ** 2, axis=0) / (Eol + eps)

    # Compute entropy of the normalized sub-frame energies:
    Entropy = -numpy.sum(s * numpy.log2(s + eps))
    return Entropy


""" Frequency-domain audio features """


def stSpectralCentroidAndSpread(X, fs):
    """Computes spectral centroid of frame (given abs(FFT))"""
    ind = (numpy.arange(1, len(X) + 1)) * (fs / (2.0 * len(X)))

    Xt = X.copy()
    Xt = Xt / Xt.max()
    NUM = numpy.sum(ind * Xt)
    DEN = numpy.sum(Xt) + eps

    # Centroid:
    C = (NUM / DEN)

    # Spread:
    S = numpy.sqrt(numpy.sum(((ind - C) ** 2) * Xt) / DEN)

    # Normalize:
    C = C / (fs / 2.0)
    S = S / (fs / 2.0)

    return (C, S)


def stSpectralEntropy(X, n_short_blocks=10):
    """Computes the spectral entropy"""
    L = len(X)  # number of frame samples
    Eol = numpy.sum(X ** 2)  # total spectral energy

    sub_win_len = int(numpy.floor(L / n_short_blocks))  # length of sub-frame
    if L != sub_win_len * n_short_blocks:
        X = X[0:sub_win_len * n_short_blocks]

    sub_wins = X.reshape(sub_win_len, n_short_blocks, order='F').copy()  # define sub-frames (using matrix reshape)
    s = numpy.sum(sub_wins ** 2, axis=0) / (Eol + eps)  # compute spectral sub-energies
    En = -numpy.sum(s * numpy.log2(s + eps))  # compute spectral entropy

    return En


def stSpectralFlux(X, X_prev):
    """
    Computes the spectral flux feature of the current frame
    ARGUMENTS:
        X:            the abs(fft) of the current frame
        X_prev:        the abs(fft) of the previous frame
    """
    # compute the spectral flux as the sum of square distances:
    sumX = numpy.sum(X + eps)
    sumPrevX = numpy.sum(X_prev + eps)
    F = numpy.sum((X / sumX - X_prev / sumPrevX) ** 2)

    return F


def stSpectralRollOff(X, c, fs):
    """Computes spectral roll-off"""
    totalEnergy = numpy.sum(X ** 2)
    fftLength = len(X)
    Thres = c * totalEnergy
    # Ffind the spectral rolloff as the frequency position
    # where the respective spectral energy is equal to c*totalEnergy
    CumSum = numpy.cumsum(X ** 2) + eps
    [a, ] = numpy.nonzero(CumSum > Thres)
    if len(a) > 0:
        mC = numpy.float64(a[0]) / (float(fftLength))
    else:
        mC = 0.0
    return (mC)


def stHarmonic(frame, fs):
    """
    Computes harmonic ratio and pitch
    """
    M = numpy.round(0.016 * fs) - 1
    R = numpy.correlate(frame, frame, mode='full')

    g = R[len(frame) - 1]
    R = R[len(frame):-1]

    # estimate m0 (as the first zero crossing of R)
    [a, ] = numpy.nonzero(numpy.diff(numpy.sign(R)))

    if len(a) == 0:
        m0 = len(R) - 1
    else:
        m0 = a[0]
    if M > len(R):
        M = len(R) - 1

    Gamma = numpy.zeros((M), dtype=numpy.float64)
    CSum = numpy.cumsum(frame ** 2)
    Gamma[m0:M] = R[m0:M] / (numpy.sqrt((g * CSum[M:m0:-1])) + eps)

    ZCR = stZCR(Gamma)

    if ZCR > 0.15:
        HR = 0.0
        f0 = 0.0
    else:
        if len(Gamma) == 0:
            HR = 1.0
            blag = 0.0
            Gamma = numpy.zeros((M), dtype=numpy.float64)
        else:
            HR = numpy.max(Gamma)
            blag = numpy.argmax(Gamma)

        # Get fundamental frequency:
        f0 = fs / (blag + eps)
        if f0 > 5000:
            f0 = 0.0
        if HR < 0.1:
            f0 = 0.0

    return (HR, f0)


def stChromaFeatures(X, fs, nChroma, nFreqsPerChroma):
    # TODO: 1 complexity
    # TODO: 2 bug with large windows

    chromaNames = ['A', 'A#', 'B', 'C', 'C#', 'D',
                   'D#', 'E', 'F', 'F#', 'G', 'G#']
    spec = X ** 2
    if nChroma.max() < nChroma.shape[0]:
        C = numpy.zeros((nChroma.shape[0],))
        C[nChroma] = spec
        C /= nFreqsPerChroma[nChroma]
    else:
        I = numpy.nonzero(nChroma > nChroma.shape[0])[0][0]
        C = numpy.zeros((nChroma.shape[0],))
        C[nChroma[0:I - 1]] = spec
        C /= nFreqsPerChroma
    finalC = numpy.zeros((12, 1))
    newD = int(numpy.ceil(C.shape[0] / 12.0) * 12)
    C2 = numpy.zeros((newD,))
    C2[0:C.shape[0]] = C
    C2 = C2.reshape(int(C2.shape[0] / 12), 12)
    # for i in range(12):
    #    finalC[i] = numpy.sum(C[i:C.shape[0]:12])
    finalC = numpy.matrix(numpy.sum(C2, axis=0)).T
    finalC /= spec.sum()

    #    ax = plt.gca()
    #    plt.hold(False)
    #    plt.plot(finalC)
    #    ax.set_xticks(range(len(chromaNames)))
    #    ax.set_xticklabels(chromaNames)
    #    xaxis = numpy.arange(0, 0.02, 0.01);
    #    ax.set_yticks(range(len(xaxis)))
    #    ax.set_yticklabels(xaxis)
    #    plt.show(block=False)
    #    plt.draw()

    return chromaNames, finalC


def stMFCC(X, fbank, n_mfcc_feats):
    """
    Computes the MFCCs of a frame, given the fft mag
    ARGUMENTS:
        X:        fft magnitude abs(FFT)
        fbank:    filter bank (see mfccInitFilterBanks)
    RETURN
        ceps:     MFCCs (13 element vector)
    Note:    MFCC calculation is, in general, taken from the
             scikits.talkbox library (MIT Licence),
    #    with a small number of modifications to make it more
         compact and suitable for the pyAudioAnalysis Lib
    """

    mspec = numpy.log10(numpy.dot(X, fbank.T) + eps)
    ceps = dct(mspec, type=2, norm='ortho', axis=-1)[:n_mfcc_feats]
    return ceps


def stFeatureExtraction(signal, fs, win, step):
    """
    This function implements the shor-term windowing process. For each short-term window a set of features is extracted.
    This results to a sequence of feature vectors, stored in a numpy matrix.
    ARGUMENTS
        signal:       the input signal samples
        fs:           the sampling freq (in Hz)
        win:          the short-term window size (in samples)
        step:         the short-term window step (in samples)
    RETURNS
        st_features:   a numpy array (n_feats x numOfShortTermWindows)
    """

    win = int(win)
    step = int(step)

    # Signal normalization
    signal = numpy.double(signal)

    signal = signal / (2.0 ** 15)
    DC = signal.mean()
    MAX = (numpy.abs(signal)).max()
    signal = (signal - DC) / (MAX + 0.0000000001)

    N = len(signal)  # total number of samples
    cur_p = 0
    count_fr = 0
    nFFT = int(win / 2)

    [fbank, freqs] = mfccInitFilterBanks(fs, nFFT)  # compute the triangular filter banks used in the mfcc calculation
    nChroma, nFreqsPerChroma = stChromaFeaturesInit(nFFT, fs)

    n_time_spectral_feats = 8
    n_harmonic_feats = 0
    n_mfcc_feats = 13
    n_chroma_feats = 13
    n_total_feats = n_time_spectral_feats + n_mfcc_feats + n_harmonic_feats + n_chroma_feats
    #    n_total_feats = n_time_spectral_feats + n_mfcc_feats + n_harmonic_feats
    feature_names = []
    feature_names.append("zcr")
    feature_names.append("energy")
    feature_names.append("energy_entropy")
    feature_names += ["spectral_centroid", "spectral_spread"]
    feature_names.append("spectral_entropy")
    feature_names.append("spectral_flux")
    feature_names.append("spectral_rolloff")
    feature_names += ["mfcc_{0:d}".format(mfcc_i)
                      for mfcc_i in range(1, n_mfcc_feats + 1)]
    feature_names += ["chroma_{0:d}".format(chroma_i)
                      for chroma_i in range(1, n_chroma_feats)]
    feature_names.append("chroma_std")
    st_features = []
    while (cur_p + win - 1 < N):  # for each short-term window until the end of signal
        count_fr += 1
        x = signal[cur_p:cur_p + win]  # get current window
        cur_p = cur_p + step  # update window position
        X = abs(fft(x))  # get fft magnitude
        X = X[0:nFFT]  # normalize fft
        X = X / len(X)
        if count_fr == 1:
            X_prev = X.copy()  # keep previous fft mag (used in spectral flux)
        curFV = numpy.zeros((n_total_feats, 1))
        curFV[0] = stZCR(x)  # zero crossing rate
        curFV[1] = stEnergy(x)
        curFV[2] = stEnergyEntropy(x)  # short-term entropy of energy
        [curFV[3], curFV[4]] = stSpectralCentroidAndSpread(X, fs)  # spectral centroid and spread
        curFV[5] = stSpectralEntropy(X)  # spectral entropy
        curFV[6] = stSpectralFlux(X, X_prev)  # spectral flux
        curFV[7] = stSpectralRollOff(X, 0.90, fs)  # spectral rolloff

        curFV[n_time_spectral_feats:n_time_spectral_feats + n_mfcc_feats, 0] = \
            stMFCC(X, fbank, n_mfcc_feats).copy()  # MFCCs

        chromaNames, chromaF = stChromaFeatures(X, fs, nChroma, nFreqsPerChroma)
        curFV[n_time_spectral_feats + n_mfcc_feats:
              n_time_spectral_feats + n_mfcc_feats + n_chroma_feats - 1] = \
            chromaF
        curFV[n_time_spectral_feats + n_mfcc_feats + n_chroma_feats - 1] = \
            chromaF.std()

        st_features.append(curFV)
        # delta features
        '''
        if count_fr>1:
            delta = curFV - prevFV
            curFVFinal = numpy.concatenate((curFV, delta))            
        else:
            curFVFinal = numpy.concatenate((curFV, curFV))
        prevFV = curFV
        st_features.append(curFVFinal)        
        '''
        # end of delta
        X_prev = X.copy()

    st_features = numpy.concatenate(st_features, 1)
    return st_features, feature_names

#特征提取end

if __name__ == '__main__':
    record()
    ser = serial.Serial("COM4", 9600, timeout=0.5)
    print (ser.name)
    print (ser.port)
    
    if ser.isOpen():
        print("succeed")
    else:
        print("fail")

    # use the wav file recorded by speeker 1 and 2 as test set
    # use the wav file recorded by speeker 3 to 10 as train set

    # type = [9, 11, 21, 43, 44]  # 43:siren 9:sheep 11:rain 21:crying baby 44:car horn
    # TTYPE = [11, 44, 43, 21, 9]  # 43:siren 9:sheep 11:rain 21:crying baby 44:car horn
    # trD = range(2, 10)  # train part 确定某种类型中用于训练的音频名称
    # teD = range(0, 1)  # test part 确定某种类型中用于测试的音频名称
    test_path=r'C:\Users\86135\Desktop\AIcar\voice.wav'
    [Fs, x] = readAudioFile(test_path);
    #x = x[:, 1]   #这个音频是2通道的
    F, f_names = stFeatureExtraction(x, Fs, 0.025 * Fs, 0.010 * Fs);  # 使用50 msecs的帧大小和25 msecs的帧步长（50％重叠）,现为25与10
    # print f_names[0]

    row=F
    count=0
    for i in range(len(row[0])):
        column = []  # 定义列数组
        count = count + 1
        for col in row:
            column.append(col[i])
        if (count == 1):
            test = column
        else:
            test = np.vstack((test, column))
    test = np.array(test, dtype=np.float)
    # print(test.shape)
    # 一段测试音频数据读取完毕

    vote = np.zeros(51)  # 每帧投一次票，统计用于预测
    model_path = "modelSVM.m"
    clf = joblib.load(model_path)
    for frameMFCC in test:
        P = clf.predict([frameMFCC])[0]
        vote[P] += 1
    result = np.argmax(vote)
    print(result)

    i=0
    if result==43:
        for i in range(5):
            if i==0:
                ser.write('a'.encode())
                ser.readline()
            else:
                ser.write('1'.encode())
                ser.readline()
            i=i+1
    #     #遇到了警报，先靠右，再直行，发信号a
    elif result==9:
        for i in range(5):
            ser.write('a'.encode())
            ser.readline()
    #     #遇到羊，扭动
    elif result==11:
        for i in range(10):
            ser.write('b'.encode())
            ser.readline()
    #     #下雨了，慢行，发信号b
    elif result==21:
        ser.write('c'.encode())
        ser.readline()
    #     #遇到宝宝哭，跳个舞，发信号c
    else:
        for i in range(10):
            ser.write('d'.encode())
            ser.readline()
    #     #遇到车鸣笛，加速，发信号d


