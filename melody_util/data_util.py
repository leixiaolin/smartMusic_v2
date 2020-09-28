from numpy import dstack
from pandas import read_csv
from keras.utils import to_categorical
import numpy as np
from collections import Counter
import random
import librosa
from melody_util.config import get_config
import matplotlib.pyplot as plt
import librosa.display
import os

def get_mean_and_std():
    # data stats for normalization
    stats = np.load('./data/means_stds_ourself.npy')
    means = stats[0]
    stds = stats[1]
    return means,stds

#function to zero pad ends of spectrogram
def zeropad2d(x,n_frames):
	y=np.hstack((np.zeros([x.shape[0],n_frames]), x))
	y=np.hstack((y,np.zeros([x.shape[0],n_frames])))
	return y

def int_random(a, b, n) :
    # 定义一个空列表存储随机数
    a_list = []
    while len(a_list) < n :
        d_int = random.randint(a, b)
        if(d_int not in a_list and d_int < b) :
            a_list.append(d_int)
        else :
            pass
    # 将生成的随机数列表转换成元组并返回
    return a_list

def int_random_v2(begin, end, needcount) :
    # 定义一个空列表存储随机数
    # step = int((b-a)/n)
    # a_list = range(a,b,step)
    # t1 = np.max(a_list)
    # result = []
    result = random.sample(range(begin, end), needcount)
    # for x in a_list:
    #     d_int = random.randint(0, step-1)
    #     tmp = x + d_int
    #     if tmp > b:
    #         tmp -= step
    #     result.append(tmp)
    # t2 = np.max(result)
    return result

def get_nearly_peaks_with_textgrid_times(onsets,peaks_in_time):
    result = []
    if len(peaks_in_time) < 1:
        return onsets
    for p in onsets:
        offset = [np.abs(o - p) for o in peaks_in_time]
        min_index = offset.index(np.min(offset))
        select_time = peaks_in_time[min_index]
        if select_time not in result:result.append(select_time)
    return result

def get_data_from_npy(data_path, label_path):
    # data_path = 'F:/项目/花城音乐项目/音符起始点检测/onset_db/data_pt_single_file/melgram1All.npy'

    # 梅尔频谱图数据
    melgram1All = np.load(data_path, allow_pickle=True)
    means, stds = get_mean_and_std()
    # 标签数据
    labels = np.load(label_path, allow_pickle=True).astype(int)
    # onset_times = [i*10e-3 for i in range(len(labels)) if labels[i] == 1]
    # librosa.display.specshow(melgram1All, x_axis='time', y_axis='mel', sr=44100, hop_length=441, fmax=16000)
    # plt.vlines(onset_times, 0, 4000, colors="c", linestyles="dashed")
    # plt.show()


    true_labels = [i for i,x in enumerate(labels) if x == 1] # 正样本的下标
    # print(true_labels)
    false_labels = [i for i,x in enumerate(labels) if x == 0] # 负样本的下标
    random_indexs = int_random_v2(0,len(false_labels),len(labels)-len(false_labels))
    selected_false_labels = [false_labels[i] for i in random_indexs]  # 欠采样后的负样本下标
    selected_labels = true_labels + selected_false_labels  #正负样本下标
    selected_labels = sorted(selected_labels)
    np.random.shuffle(selected_labels)
    # print(selected_labels)

    contextlen, duration, window_len = get_config()
    trainX = makechunks(melgram1All,duration)

    # start =90
    # lens = 15
    # display_pic(trainX[start:start + lens], labels[start:start + lens])
    # print(trainX.shape)
    # trainX.transpose((2,1))
    trainX =  np.transpose(trainX, (0,2,1))
    trainY = labels.reshape(len(labels),)
    # print(trainX.shape)
    # print(trainY.shape)
    # start = 100
    # length = 25
    # print(trainY[start:start + length])
    # display_pic_T(trainX[start:start + length], None)
    selected_trainX = np.zeros((len(selected_labels),trainX.shape[1],trainX.shape[2]),dtype=float)
    selected_trainY = np.zeros((len(selected_labels)),dtype=int)
    for i,x in enumerate(selected_labels):
        selected_trainX[i,:,:] = trainX[x,:,:]
        selected_trainY[i] = trainY[x]
    # print(selected_trainX.shape)
    # print(selected_trainY.shape)

    trainy_tmp = np.squeeze(selected_trainY)
    # print(Counter(trainy_tmp))
    return selected_trainX,selected_trainY

#
# 这方法是标签数据块是否有起始点
def get_data_from_npy_v2(data_path, label_path):
    # data_path = 'F:/项目/花城音乐项目/音符起始点检测/onset_db/data_pt_single_file/melgram1All.npy'

    # 梅尔频谱图数据
    melgram1All = np.load(data_path, allow_pickle=True)

    # 标签数据
    labels = np.load(label_path, allow_pickle=True).astype(int)

    contextlen, duration, window_len = get_config()
    trainX,trainY = makechunks_v2(melgram1All,labels,window_len)
    print(trainX.shape,trainY.shape)
    labels = trainY

    true_labels = [i for i,x in enumerate(labels) if x == 1] # 正样本的下标
    false_labels = [i for i,x in enumerate(labels) if x == 0] # 负样本的下标
    random_indexs = int_random_v2(0,len(false_labels),len(labels)-len(false_labels))
    selected_false_labels = [false_labels[i] for i in random_indexs]  # 欠采样后的负样本下标
    selected_labels = true_labels + selected_false_labels  #正负样本下标
    sorted(selected_labels)
    np.random.shuffle(selected_labels)
    print(selected_labels)

    # trainX.transpose((2,1))
    trainX =  np.transpose(trainX, (0,2,1))
    trainY = labels.reshape(len(labels),)
    print(trainX.shape)
    print(trainY.shape)
    selected_trainX = np.zeros((len(selected_labels),trainX.shape[1],trainX.shape[2]),dtype=float)
    selected_trainY = np.zeros((len(selected_labels)),dtype=int)
    for i,x in enumerate(selected_labels):
        selected_trainX[i,:,:] = trainX[x,:,:]
        selected_trainY[i] = trainY[x]
    print(selected_trainX.shape)
    print(selected_trainY.shape)

    trainy_tmp = np.squeeze(selected_trainY)
    print(Counter(trainy_tmp))
    return selected_trainX,selected_trainY

def makechunks(x, duration):
    y = np.zeros([x.shape[1], x.shape[0], duration])
    for i_frame in range(x.shape[1] - duration):
        y[i_frame] = x[:, i_frame:i_frame + duration]
    return y

def makechunks_v2(x, y,window_len):
    hop_len = int(window_len * 0.5)
    chunk_total = int((x.shape[1] - window_len)/hop_len)
    contextlen, duration, window_len = get_config()
    trainX = np.zeros([chunk_total, x.shape[0], window_len])
    start = 0
    trainY = np.zeros([chunk_total])
    for i_frame in range(0,chunk_total):
        trainX[i_frame] = x[:, start:start + window_len]
        tmp = y[start:start + window_len]
        if np.max(tmp) > 0:
            trainY[i_frame] = 1
        start += hop_len
    return trainX,trainY

def makechunks_for_predict(x, window_len):
    chunk_total = int((x.shape[1] - window_len)/window_len)
    contextlen, duration, window_len = get_config()
    trainX = np.zeros([chunk_total, x.shape[0], window_len])
    start = 0
    for i_frame in range(0,chunk_total):
        trainX[i_frame] = x[:, start:start + window_len]
        start += window_len
    return trainX

# load the dataset, returns train and test X and y elements
def load_dataset(type=1,data_path = './data/melgram1All.npy', label_path = './data/labels_all.npy'):
    # load all train
    if type == 1:
        trainXall, trainYall = get_data_from_npy(data_path, label_path)
    else:
        trainXall, trainYall = get_data_from_npy_v2(data_path, label_path)
    trainYall = trainYall.reshape(len(trainYall),-1)
    # print(trainXall.shape, trainYall.shape)
    # print(trainXall.shape, trainYall.shape)
    # trainXall, trainYall = underSampler(trainXall, trainYall)
    # trainy_tmp = np.squeeze(trainYall)
    # sorted(trainy_tmp)
    # print(Counter(trainy_tmp))
    rate = 0.9

    index = int(trainYall.shape[0] * rate)
    trainX = trainXall[:index,:,:]
    trainY = trainYall[:index,:]

    # print("===============================")
    # print(sorted(Counter(np.squeeze(trainY)).items()))
    # print("===============================")

    testX = trainXall[index:, :, :]
    testY = trainYall[index:,:]

    # print(testX.shape, testY.shape)
    # zero-offset class values
    # trainY = trainY - 1
    # testY = testY - 1
    # one hot encode y
    trainY = to_categorical(trainY)
    testY = to_categorical(testY)
    print(trainX.shape, trainY.shape, testX.shape, testY.shape)
    return trainX, trainY, testX, testY

def underSampler(trainX,trainy):
    trainX_temp = trainX.reshape(trainX.shape[0], -1)

    from imblearn.under_sampling import RandomUnderSampler

    rus = RandomUnderSampler(random_state=0)
    X_resampled, y_resampled = rus.fit_sample(trainX_temp, trainy)
    X_resampled = X_resampled.reshape(X_resampled.shape[0], trainX.shape[1], -1)
    return X_resampled, y_resampled

def get_chunks_from_file(filename):
    root_path = os.path.abspath(os.path.dirname(__file__))
    data_path = os.path.join(root_path, './data/means_stds_ourself.npy')
    stats = np.load(data_path)
    means = stats[0]
    stds = stats[1]
    contextlen, duration, window_len = get_config()

    x, fs = librosa.load(filename, sr=44100)
    melgram1 = librosa.feature.melspectrogram(x, sr=fs, n_fft=1024, hop_length=441, n_mels=80, fmin=27.5, fmax=16000)
    melgram1 = 10 * np.log10(1e-10 + melgram1)
    melgram1 = (melgram1 - np.atleast_2d(means[0]).T) / np.atleast_2d(stds[0]).T  #normalize
    melgram1 = zeropad2d(melgram1, contextlen)  #zero pad ends
    trainX = makechunks(melgram1, duration)
    return trainX

# 切分数据块
def get_chunks_from_file_v2(filename):
    stats = np.load('./data/means_stds_jj.npy')
    means = stats[0]
    stds = stats[1]
    contextlen, duration, window_len = get_config()

    x, fs = librosa.load(filename, sr=44100)
    melgram1 = librosa.feature.melspectrogram(x, sr=fs, n_fft=1024, hop_length=441, n_mels=80, fmin=27.5, fmax=16000)
    melgram1 = 10 * np.log10(1e-10 + melgram1)
    melgram1 = (melgram1 - np.atleast_2d(means[0]).T) / np.atleast_2d(stds[0]).T  #normalize
    melgram1 = zeropad2d(melgram1, contextlen)  #zero pad ends
    contextlen, duration, window_len = get_config()
    trainX = makechunks_for_predict(melgram1, window_len)
    return trainX

def get_cqt_chunks_from_file(filename):
    stats = np.load('./data/cqt_means_stds_ourself.npy',allow_pickle=True)
    means = stats[0]
    stds = stats[1]

    contextlen, duration, window_len = get_config()
    x, fs = librosa.load(filename, sr=44100)
    melgram1 = librosa.amplitude_to_db(librosa.cqt(x, sr=fs,hop_length=512,n_bins=80), ref = np.max)
    melgram1 = (melgram1 - np.atleast_2d(means[0]).T) / np.atleast_2d(stds[0]).T  #normalize
    melgram1 = zeropad2d(melgram1, contextlen)  #zero pad ends
    trainX = makechunks(melgram1, duration)
    return trainX

def display_pic(data,labels):
    plt.figure(figsize=(10, 4))
    num = data.shape[0]
    for i in range(num):
        plt.subplot(1, num, i+1)
        librosa.display.specshow(data[i], x_axis='time',sr=51200, hop_length=512)
        if labels is not None:
            plt.title(labels[i])
    # librosa.display.specshow(CQT ,y_axis='cqt_note',x_axis='time')
    plt.show()

def display_pic_T(data,labels):
    # means, stds = get_mean_and_std()
    plt.figure(figsize=(10, 4))
    num = data.shape[0]
    for i in range(num):
        plt.subplot(1, num, i+1)
        tmp = data[i].T
        # tmp = (tmp - np.atleast_2d(means[0]).T) / np.atleast_2d(stds[0]).T
        librosa.display.specshow(tmp, x_axis='time',sr=51200, hop_length=512)
        if labels is not None:
            plt.title(labels[i])
    # librosa.display.specshow(CQT ,y_axis='cqt_note',x_axis='time')
    plt.show()

if __name__ == '__main__':
    data_path, label_path = './data/test.npy', './data/labels_test.npy'
    # data_path, label_path = './data/melgram1All_jj.npy', './data/labels_all_jj.npy'
    # data_path, label_path = './data/melgram1All_ourself35.npy', './data/labels_all_ourself_35.npy'
    # data_path, label_path = './data/cqt_v2_win30.npy', './data/cqt_labels_v2_win30.npy'
    trainX, trainy, testX, testy = load_dataset(1,data_path, label_path)
    print(trainX.shape, trainy.shape, testX.shape, testy.shape)
    start = 0
    len = 15
    max_y_pred_test = np.argmax(trainy[start:start + len], axis=1)
    print(max_y_pred_test)
    display_pic_T(trainX[start:start + len],max_y_pred_test)

    # trainX_temp = trainX.reshape(trainX.shape[0],-1)
    #
    # from imblearn.under_sampling import RandomUnderSampler
    #
    # rus = RandomUnderSampler(random_state=0)
    # X_resampled, y_resampled = rus.fit_sample(trainX_temp, trainy)
    # X_resampled = X_resampled.reshape(X_resampled.shape[0],trainX.shape[1],-1)
    # print(X_resampled.shape,y_resampled.shape)
    #
    # print(sorted(Counter(np.squeeze(y_resampled)).items()))