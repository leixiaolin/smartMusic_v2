import librosa
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
from create_base import *
# 1. Get the file path to the included audio example
# Sonify detected beat events
# 定义加载语音文件并去掉两端静音的函数
def load_and_trim(path):
    audio, sr = librosa.load(path)
    energy = librosa.feature.rmse(audio)
    frames = np.nonzero(energy >= np.max(energy) / 5)
    indices = librosa.core.frames_to_samples(frames)[1]
    audio = audio[indices[0]:indices[-1]] if indices.size else audio[0:0]

    return audio, sr

def get_note_with_cqt_rms(filename):
    y, sr = librosa.load(filename)
    rms = librosa.feature.rmse(y=y)[0]
    rms = [x / np.std(rms) for x in rms]
    time = librosa.get_duration(filename=filename)
    print("time is {}".format(time))
    CQT = librosa.amplitude_to_db(librosa.cqt(y, sr=16000), ref=np.max)
    w, h = CQT.shape
    print("w.h is {},{}".format(w, h))
    onsets_frames = get_real_onsets_frames_rhythm(y)
    CQT = np.where(CQT > -20, np.max(CQT), np.min(CQT))
    result = []
    last = 0
    is_ok = 0
    for i in range(10, h - 10):
        is_ok = 0
        last_j = 100
        for j in range(w-1, 15,-1):
            if np.min(CQT[j, i:i + 5]) == np.max(CQT) and np.max(CQT[j, i - 4:i - 1]) == np.min(CQT) and i - last > 5:
                if last_j - j > 10:
                    is_ok += 1
                    last_j = j
            elif np.min(CQT[j, i:i + 5]) == np.max(CQT) and is_ok == 1:
                is_ok += 1
            elif np.min(CQT[j, i:i + 15]) == np.max(CQT):
                is_ok += 2
                break
        if rms[i + 1] > rms[i] and is_ok > 1:
            if len(result) == 0:
                result.append(i)
                last = i
            elif i - result[-1] > 5:
                result.append(i)
                last = i


    rms_on_frames = [rms[x] for x in result]
    mean_rms_on_frames = np.mean(rms_on_frames)
    onstm = librosa.frames_to_time(result, sr=sr)
    # CQT[:,onsets_frames[1]:h] = -100
    plt.subplot(3, 1, 1)
    total_frames_number = get_total_frames_number(filename)
    print("total_frames_number is {}".format(total_frames_number))
    # librosa.display.specshow(CQT)
    librosa.display.specshow(CQT, y_axis='cqt_note', x_axis='time')
    print(np.max(y))
    # onstm = librosa.frames_to_time(onsets_frames, sr=sr)
    plt.vlines(onstm, 0, sr, color='y', linestyle='solid')

    plt.subplot(3, 1, 2)
    max_rms = np.max(rms)
    # rms = np.diff(rms)
    times = librosa.frames_to_time(np.arange(len(rms)))
    # rms_on_onset_frames_cqt = [rms[x] for x in onset_frames_cqt]
    # min_rms_on_onset_frames_cqt = np.min(rms_on_onset_frames_cqt)
    # rms = [1 if x >=min_rms_on_onset_frames_cqt else 0 for x in rms]
    plt.plot(times, rms)
    plt.axhline(mean_rms_on_frames,color='r')
    # plt.axhline(min_rms_on_onset_frames_cqt)

    # plt.vlines(onsets_frames_rms_best_time, 0,np.max(rms), color='y', linestyle='solid')
    plt.vlines(onstm, 0, np.max(rms), color='y', linestyle='solid')
    # plt.vlines(base_onsets, 0, np.max(rms), color='r', linestyle='solid')
    plt.xlim(0, np.max(times))

    plt.subplot(3, 1, 3)
    librosa.display.waveplot(y, sr=sr)


    return plt

if __name__ == "__main__":
    #y, sr = load_and_trim('F:/项目/花城音乐项目/样式数据/ALL/旋律/1.31MP3/旋律1.100分.wav')
    filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律2.1(80).wav'
    filename = 'F:/项目/花城音乐项目/样式数据/ALL/旋律/1.31MP3/旋律3.100分.wav'
    #filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律一（9）（100）.wav'
    #filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律一（14）（95）.wav'
    filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋律五（3）（63）.wav'
    #filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏一（4）（96）.wav'

    #y, sr = load_and_trim(filename)

    # plt.colorbar(format='%+2.0f dB')
    # plt.title('Constant-Q power spectrogram (note)')
    plt = get_note_with_cqt_rms(filename)
    plt.show()

    dir_list = ['F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/']
    # dir_list = ['e:/test_image/m1/A/']
    dir_list = []
    total_accuracy = 0
    total_num = 0
    result_path = 'e:/test_image/n/'
    # clear_dir(result_path)
    # 要测试的数量
    test_num = 100
    score = 0
    for dir in dir_list:
        file_list = os.listdir(dir)
        # shuffle(file_list)  # 将语音文件随机排列
        # file_list = ['视唱1-01（95）.wav']
        for filename in file_list:
            # clear_dir(image_dir)
            # wavname = re.findall(pattern,filename)[0]
            print(dir + filename)
            # plt = draw_start_end_time(dir + filename)
            # plt = draw_baseline_and_note_on_cqt(dir + filename, False)
            plt = get_note_with_cqt_rms(dir + filename)
            # tmp = os.listdir(result_path)

            if filename.find("tune") > 0 or filename.find("add") > 0 or filename.find("shift") > 0:
                score = re.sub("\D", "", filename.split("-")[0])  # 筛选数字
            else:
                score = re.sub("\D", "", filename)  # 筛选数字

            if str(score).find("100") > 0:
                score = 100
            else:
                score = int(score) % 100

            if int(score) >= 90:
                grade = 'A'
            elif int(score) >= 75:
                grade = 'B'
            elif int(score) >= 60:
                grade = 'C'
            elif int(score) >= 1:
                grade = 'D'
            else:
                grade = 'E'
            # result_path = result_path + grade + "/"
            # plt.savefig(result_path + filename + '.jpg', bbox_inches='tight', pad_inches=0)
            grade = 'A'
            plt.savefig(result_path + grade + "/" + filename + '.jpg', bbox_inches='tight', pad_inches=0)
            plt.clf()