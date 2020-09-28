import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # 预测时强制使用cpu，用时比GPU还快些

from keras.models import load_model
import numpy as np
from melody_util.data_util import get_chunks_from_file,get_cqt_chunks_from_file,get_chunks_from_file_v2
import librosa
from matplotlib import pyplot as plt
import librosa.display
from melody_util.spectral_flux_util import get_peaks_from_file
import time
import os
from melody_util.spectral_flux_util import get_troughs_from_file
import gc
from melody_util.parselmouth_sc_util import get_pitch_by_parselmouth,del_onset_frames_for_too_short
from melody_util.data_util import get_nearly_peaks_with_textgrid_times
from melody_util.parselmouth_sc_util import get_pitch_derivative_from_file

contextlen = 15  # +- frames
duration = 2 * contextlen + 1

def show_plt(filename,true_indexs):
    y, sr = librosa.load(filename, sr=44100)
    # y1 = y[0:2048]
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=441, n_mels=80, fmin=27.5, fmax=16000)
    # S = librosa.feature.melspectrogram(y=y1, sr=sr,n_fft=2048, hop_length=441,n_mels=80, fmin=27.5, fmax=16000)
    # print(S.shape)

    plt.figure(figsize=(10, 4))
    S_dB = librosa.power_to_db(S, ref=np.max)
    S_dB = 10 * np.log10(1e-10 + S)
    librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, hop_length=441,fmax=16000)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-frequency spectrogram with onset')
    plt.tight_layout()
    hop_dur = 10e-3
    onset_times = [x* hop_dur for x in true_indexs]
    # print(onset_times)
    # onset_times = [8.48, 11.82, 12.79, 13.06, 13.13, 13.77, 15.11, 15.86, 17.23, 17.86, 18.51, 18.61, 19.25, 19.59, 22.63, 23.32, 24.68, 25.34, 27.99, 28.08, 29.37, 29.42, 29.68, 29.77, 30.14, 33.53, 34.9, 35.55, 35.89, 36.88, 37.6, 38.94, 39.61, 39.65, 40.35, 40.99, 41.34, 44.41, 45.07, 45.41, 46.4, 46.77, 47.14, 47.8, 49.8, 49.9, 51.2, 51.68, 51.93, 52.6]
    plt.vlines(onset_times, 0, 4000, colors="c", linestyles="dashed")
    # plt.vlines(troughs, 0, 2000, colors='r', linestyles='dotted')
    #plt.show()
    return plt
def predict_onset_frames_from_single_file_with_model(filename,model):
    start = time.time()
    testX = get_chunks_from_file(filename)
    # testX = get_cqt_chunks_from_file(filename)
    # testX = get_chunks_from_file_v2(filename)
    # print(testX.shape)
    testX = np.transpose(testX, (0, 2, 1))
    # peaks = get_peaks_from_file(filename)
    # peaks_in_time = [p*10e-3 for p in peaks]
    # # print("peaks_in_time is {}".format(peaks_in_time))
    # # 以极值点为准星，前后扩充
    # peaks_augs = [[i for i in range(p - 2, p + 2)] for p in peaks]
    # peaks_augs = np.array(peaks_augs).flatten()
    # peaks_augs = np.unique(peaks_augs, axis=0)
    # peaks_augs = [p for p in peaks_augs if p < testX.shape[0]]
    # peaks = peaks_augs
    # # print("peaks is {}".format(peaks))
    # selected_testX = np.zeros([len(peaks), testX.shape[1], testX.shape[2]])
    # for i, p in enumerate(peaks):
    #     selected_testX[i, :, :] = testX[p, :, :]
    # testX = selected_testX
    # testX = testX[:int(testX.shape[0]*0.5),:,:]

    # model = load_model(model_path)
    y_pred_test = model.predict(testX)
    # Take the class with the highest probability from the test predictions
    max_y_pred_test = np.argmax(y_pred_test, axis=1)

    # print(max_y_pred_test.shape)
    pred_indexs = [i for i, x in enumerate(max_y_pred_test) if x == 1]
    # print("pred_indexs is {}".format(pred_indexs))
    # tmp = [p for i, p in enumerate(peaks) if i in pred_indexs]
    tmp = pred_indexs
    select_tmp = []
    if len(tmp) > 0:
        # select_tmp = [tmp[0]]
        # for i in range(1,len(tmp)):
        #     if tmp[i] - tmp[i-1] > 4 and tmp[i] not in select_tmp:
        #         select_tmp.append(tmp[i]) # 过密的取第一个
        for i in range(len(tmp)-1):
            if tmp[i+1] - tmp[i] > 4 and tmp[i] not in select_tmp:
                select_tmp.append(tmp[i]) # 过密的取最后一个
        select_tmp.append(pred_indexs[-1])
        # print("select_tmp is {}".format(select_tmp))
        select_tmp = del_first(select_tmp) # 第一种删除方法：当前段的前后段都远大于当前段
        # print("select_tmp is {}".format(select_tmp))
    onsets = [t*10e-3 for t in select_tmp]
    # select_tmp = get_nearly_peaks_with_textgrid_times(onsets, peaks_in_time) # 每个取附近的频谱能量变化极值点
    # select_tmp = [int(t/10e-3) for t in select_tmp]
    # print("select_tmp is {}".format(select_tmp))
    pitch = get_pitch_by_parselmouth(filename)
    select_tmp = del_onset_frames_for_too_short(pitch, select_tmp)
    # print("select_tmp is {}".format(select_tmp))
    # pitch_derivative = get_pitch_derivative_from_file(filename)
    # select_tmp = del_second_type(select_tmp,pitch_derivative)
    end = time.time()
    running_time = end - start
    print('time cost : %.5f sec' % running_time)
    # pic = show_plt(filename, pred_indexs)
    return select_tmp

def predict_onset_frames_from_single_file(filename,model_path = './model/best_model.02-0.95-ourself-1.h5'):
    # print(os.path.abspath(os.path.dirname(__file__)))
    # model_path = './model/best_model.03-0.97-ourself-1.h5'
    # print(model_path)
    root_path = os.path.abspath(os.path.dirname(__file__))
    model_path = os.path.join(root_path,model_path)
    m = load_model(model_path, compile=False)
    onset_frames = predict_onset_frames_from_single_file_with_model(filename, m)
    return onset_frames

def get_predict_pic_from_single_file(filename,model):
    start = time.time()
    testX = get_chunks_from_file(filename)
    # testX = get_cqt_chunks_from_file(filename)
    # testX = get_chunks_from_file_v2(filename)
    # print(testX.shape)
    testX = np.transpose(testX, (0, 2, 1))
    peaks = get_peaks_from_file(filename)
    peaks_in_time = [p*10e-3 for p in peaks]
    # print("peaks_in_time is {}".format(peaks_in_time))
    # 以极值点为准星，前后扩充
    peaks_augs = [[i for i in range(p - 2, p + 2)] for p in peaks]
    peaks_augs = np.array(peaks_augs).flatten()
    peaks_augs = np.unique(peaks_augs, axis=0)
    peaks_augs = [p for p in peaks_augs if p < testX.shape[0]]
    peaks = peaks_augs
    # print("peaks is {}".format(peaks))
    selected_testX = np.zeros([len(peaks), testX.shape[1], testX.shape[2]])
    for i, p in enumerate(peaks):
        selected_testX[i, :, :] = testX[p, :, :]
    # testX = selected_testX
    # testX = testX[:int(testX.shape[0]*0.5),:,:]

    # model = load_model(model_path)
    y_pred_test = model.predict(testX)
    # Take the class with the highest probability from the test predictions
    max_y_pred_test = np.argmax(y_pred_test, axis=1)

    # print(max_y_pred_test.shape)
    pred_indexs = [i for i, x in enumerate(max_y_pred_test) if x == 1]
    # print("pred_indexs is {}".format(pred_indexs))
    # tmp = [p for i, p in enumerate(peaks) if i in pred_indexs]
    tmp = pred_indexs
    select_tmp = []
    if len(tmp) > 0:
        # select_tmp = [tmp[0]]
        # for i in range(1,len(tmp)):
        #     if tmp[i] - tmp[i-1] > 4 and tmp[i] not in select_tmp:
        #         select_tmp.append(tmp[i]) # 过密的取第一个
        for i in range(len(tmp)-1):
            if tmp[i+1] - tmp[i] > 4 and tmp[i] not in select_tmp:
                select_tmp.append(tmp[i]) # 过密的取最后一个
        select_tmp.append(pred_indexs[-1])
        # print("select_tmp is {}".format(select_tmp))
        select_tmp = del_first(select_tmp) # 第一种删除方法：当前段的前后段都远大于当前段
        # print("select_tmp is {}".format(select_tmp))
    onsets = [t*10e-3 for t in select_tmp]
    # select_tmp = get_nearly_peaks_with_textgrid_times(onsets, peaks_in_time) # 每个取附近的频谱能量变化极值点
    # select_tmp = [int(t/10e-3) for t in select_tmp]
    # print("select_tmp is {}".format(select_tmp))
    pitch = get_pitch_by_parselmouth(filename)
    select_tmp = del_onset_frames_for_too_short(pitch, select_tmp)
    # print("select_tmp is {}".format(select_tmp))
    # pitch_derivative = get_pitch_derivative_from_file(filename)
    # select_tmp = del_second_type(select_tmp,pitch_derivative)
    end = time.time()
    running_time = end - start
    print('time cost : %.5f sec' % running_time)
    pic = show_plt(filename, select_tmp)
    # pic = show_plt(filename, pred_indexs)
    return pic

# 第一种删除方法：当前段的前后段都远大于当前段，
def del_first(onset_frames):
    onset_duration = np.diff(onset_frames)
    ratio = 2.5
    del_indexs = []
    # 第一段：如果远小于第二段
    if len(onset_duration) < 2:
        return onset_frames
    if ratio * onset_duration[0] < onset_duration[1]:
        del_indexs.append(1)
    # 其他段
    onset_duration = np.append(onset_duration,np.max(onset_duration))
    for i in range(1,len(onset_duration)-1):
        if onset_duration[i-1] > ratio * onset_duration[i] < onset_duration[i+1]:
            del_indexs.append(i)
    select_onset_frames = [ f for i,f in enumerate(onset_frames) if i not in del_indexs]
    return select_onset_frames

# 第二种删除方法：对于短距（间距小于25）的线，如果后面紧挨空白音高，或者音高无变化的。
def del_second_type(onset_frames,pitch_derivative):
    if len(pitch_derivative) < 1:
        return onset_frames
    delete_frames = []
    onset_frames_tmp = onset_frames.copy()
    onset_frames_tmp.append(onset_frames[-1] + 20) # 假设最后一个节奏时长为20
    onset_duration = np.diff(onset_frames_tmp)
    for i,o in enumerate(onset_frames):
        if onset_duration[i] < 20:
            if o + 3 < len(pitch_derivative):
                tmp = pitch_derivative[o:o+3]
                if np.min(tmp) < -100: # 如果后面紧挨空白音高,则音高差值小于-100
                    delete_frames.append(o)
                    continue
                tmp = pitch_derivative[o-3:o+3]
                if len(tmp) > 0 and np.max(np.diff(tmp)) < 1: # 音高无变化的
                    delete_frames.append(o)
    select_onset_frames = [f for f in onset_frames if f not in delete_frames]
    return select_onset_frames

troughs = []
if __name__ == "__main__":
    filename = 'F:/项目/花城音乐项目/样式数据/20.06.19MP3/20200603-4611.wav'
    filename = 'F:/项目/花城音乐项目/音符起始点检测/jingju/part1/wav/all/daeh-Bie_yuan_zhong-Mei_fei-qm.wav'
    filename = 'F:/项目/花城音乐项目/音符起始点检测/ourself/wav/10.wav'
    filename = 'F:/项目/花城音乐项目/样式数据/20.03.16MP3/wav/2adcc430-0dc8-4c7e-bb24-ec36dd0f82ae.wav'
    filename = 'F:/项目/花城音乐项目/样式数据/20.03.16MP3/wav/4c1590c4-8a8d-4920-80dd-e4fc9e7a1d95.wav'
    filename = 'F:/项目/花城音乐项目/样式数据/20.03.16MP3/wav/c6e90530-1ea0-40d4-8994-0a58e6c8258b.wav'
    filename = 'F:/项目/花城音乐项目/样式数据/20.03.16MP3/wav/42d3b2f6-4fd9-4a72-a242-a2da0053e61d.wav'
    filename = 'F:/项目/花城音乐项目/样式数据/8.12MP3/旋律/小学8题20190809-3492-6.wav'
    filename = 'F:/项目/花城音乐项目/音符起始点检测/ourself/wav/小学8题20190717-6249-7_aug2-2.5.wav'
    filename = 'F:/项目/花城音乐项目/音符起始点检测/ourself/wav/小学8题20190717-6249-7_aug1-2.wav'
    filename = 'F:/项目/花城音乐项目/音符起始点检测/ourself/wav/x3_aug2--1.5.wav'
    filename = 'F:/项目/花城音乐项目/样式数据/1.31MP3/wav/节奏/节奏1.20分.wav'
    filename = 'F:/项目/花城音乐项目/样式数据/1.31MP3/wav/旋律/done/旋律1_40分.wav'
    filename = 'F:/项目/花城音乐项目/样式数据/1.31MP3/wav/旋律/done/旋律1_92分.wav'
    filename = 'F:/项目/花城音乐项目/样式数据/6.18MP3/旋律/2；83.wav'
    filename = 'F:/项目/花城音乐项目/样式数据/6.18MP3/旋律/旋律3，55.wav'
    filename = 'F:/项目/花城音乐项目/样式数据/6.18MP3/旋律/五年级1，98.wav'
    filename = 'F:/项目/花城音乐项目/样式数据/1.31MP3/wav/旋律/done/旋律1_90分.wav'
    filename = 'F:/项目/花城音乐项目/样式数据/1.31MP3/wav/旋律/done/旋律4_70分.wav'
    filename = 'F:/项目/花城音乐项目/样式数据/12.05MP3/wav/旋律/B-10.wav'
    filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/9.08MP3/旋律/zx1.wav', '[500,250,250,500,500;1500,500;1000,1000;2000]', '[3,3,1,3,4,5,5,6,7,1+]'
    filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/视唱1-02（90）.wav'
    # filename = 'F:/项目/花城音乐项目/样式数据/1.31MP3/wav/旋律/done/旋律9_25分.wav'
    # filename = 'F:/项目/花城音乐项目/样式数据/1.31MP3/wav/旋律/done/旋律9.30分.wav'
    # filename = 'F:/项目/花城音乐项目/样式数据/1.31MP3/wav/旋律/done/旋律1_92分.wav'
    # filename = 'F:/项目/花城音乐项目/音符起始点检测/ourself/wav/z1_aug1-1.wav'
    # filename = 'F:/项目/花城音乐项目/样式数据/20.05.01MP3/wav/6749.wav'
    model_path = './model/best_model.03-0.97-ourself-1.h5'
    model_path = './model/best_model.01-0.95-ourself-1.h5'
    model_path = './model/best_model.02-0.95-ourself-1.h5'
    # model_path = './models_bak/best_model.01-0.18ourself.h5'
    savepath = 'E:/t/'

    model = load_model(model_path, compile=False)
    troughs = get_troughs_from_file(filename)
    troughs = [t*10e-3 for t in troughs]
    # pic = get_predict_pic_from_single_file(filename,model)
    # pic.show()
    onset_frames = predict_onset_frames_from_single_file(filename)
    # onset_frames = predict_onset_frames_from_single_file_with_model(filename,model)
    print("onset_frames is {}, size is {}".format(onset_frames,len(onset_frames)))

    exit()
    dir_list = ['F:/项目/花城音乐项目/样式数据/20.03.16MP3/wav/','F:/项目/花城音乐项目/样式数据/20.03.18MP3/wav/']
    dir_list = ['F:/项目/花城音乐项目/样式数据/20.03.16MP3/wav/']
    dir_list = ['F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/']
    dir_list = ['F:/项目/花城音乐项目/样式数据/6.18MP3/旋律/']
    # dir_list = ['F:/项目/花城音乐项目/样式数据/8.12MP3/旋律/']
    # dir_list = ['F:/项目/花城音乐项目/音符起始点检测/ourself/wav/']
    # dir_list = ['F:/项目/花城音乐项目/样式数据/1.31MP3/wav/节奏/']
    # dir_list = ['F:/项目/花城音乐项目/样式数据/1.31MP3/wav/旋律/']
    dir_list = ['F:/项目/花城音乐项目/样式数据/1.31MP3/wav/旋律/done/']
    # dir_list = []
    for dir in dir_list:
        file_list = os.listdir(dir)
        # shuffle(file_list)  # 将语音文件随机排列
        #file_list = ['节2罗（75）.wav']
        file_total = len(file_list)
        for filename in file_list:
            if filename.find('wav') >= 0:
                print(filename)
                pic = get_predict_pic_from_single_file(os.path.join(dir,filename),model)
                pic.savefig(savepath + filename.split('.wav')[0] + '.png', bbox_inches='tight', pad_inches=0)
                pic.clf()
                plt.close('all')
                gc.collect()
