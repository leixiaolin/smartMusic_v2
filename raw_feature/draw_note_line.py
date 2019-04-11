import librosa
import matplotlib.pyplot as plt
import numpy as np
import librosa.display
from create_base import *




# 2. Load the audio as a waveform `y`
#    Store the sampling rate as `sr`

# 定义加载语音文件并去掉两端静音的函数
def load_and_trim(path):
    audio, sr = librosa.load(path)
    energy = librosa.feature.rmse(audio)
    frames = np.nonzero(energy >= np.max(energy) / 5)
    indices = librosa.core.frames_to_samples(frames)[1]
    audio = audio[indices[0]:indices[-1]] if indices.size else audio[0:0]

    return audio, sr

def draw_start_end_time(filename):
    y, sr = librosa.load(filename)
    print("y min :{}".format(np.min(y)))
    print("y max :{}".format(np.max(y)))
    plt.subplot(3,1,1)
    librosa.display.waveplot(y, sr=sr)
    rms = librosa.feature.rmse(y=y)[0]
    print(np.std(rms))
    zero_crossings = librosa.zero_crossings(y, pad=False)
    print(sum(zero_crossings))
    plt.subplot(3,1,2)
    rms = librosa.feature.rmse(y=y)[0]
    rms = [x / np.std(rms) for x in rms]
    rms = [1 if x > np.max(rms)*0.20 else 0 for x in rms]
    times = librosa.frames_to_time(np.arange(len(rms)))
    plt.plot(times, rms)
    plt.xlim(0,np.max(times))
    start,end = get_start_and_end_for_note(filename)
    start_time = librosa.frames_to_time([start,end])
    #plt.axvline(start_time[0],color="r")
    plt.axvline(start_time[1],color="r")
    print("start,end is {},{}".format(start,end))
    base_frames = onsets_base_frames_for_note(filename)
    base_frames = [x + start - base_frames[0] for x in base_frames]
    base_time = librosa.frames_to_time(base_frames, sr=sr)
    print("base_frames is {}".format(base_frames[0]))
    plt.vlines(base_time, 0, np.max(rms), color='r', linestyle='dashed')
    plt.subplot(3,1,3)
    CQT = librosa.amplitude_to_db(librosa.cqt(y, sr=16000), ref=np.max)
    w, h = CQT.shape
    CQT[0:20, :] = np.min(CQT)
    base_notes = base_note(filename)
    base_notes = [x + 5 - np.min(base_notes) for x in base_notes]
    print("base_notes is {}".format(base_notes))
    #CQT[base_notes[0], :] = -20
    CQT = add_base_note_to_cqt(CQT, base_notes, base_frames,end)
    #plt.axhline(base_frames[0],color="w")
    librosa.display.specshow(CQT, y_axis='cqt_note', x_axis='time')
    plt.vlines(base_time, 0, sr, color='r', linestyle='dashed')
    return plt
#plt.axis('off')
# plt.axes().get_xaxis().set_visible(False)
# plt.axes().get_yaxis().set_visible(False)
if __name__ == "__main__":
    # 波形幅度包络图
    filepath = 'F:\项目\花城音乐项目\样式数据\音乐样本2019-01-29\节奏九\\'
    filename = 'F:/项目/花城音乐项目/样式数据/ALL/旋律/1.31MP3/旋律2.100分.wav'
    filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏1_40441（96）.wav'
    # filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏1周(95).wav'
    # filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏1怡(90).wav'
    filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律二（9）（92）.wav'
    filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/视唱1-03（10）.wav'
    filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋1.1(96).wav'
    filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋律八（2）（60）.wav'
    filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋8录音4(93).wav'
    filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律10_40231（20）.wav'
    filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律2_40330（60）.wav'
    filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律6.1(80).wav'
    filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋3.3(96).wav'
    filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律一（12）（95）.wav'
    #filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律一（12）（95）-shift--1.wav'
    filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律一（12）（95）-add.wav'

    #filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律3_40302（95）.wav'



    #plt = draw_start_end_time(filename)
    plt = draw_baseline_and_note_on_cqt(filename)
    plt.show()

    dir_list = ['F:/项目/花城音乐项目/样式数据/3.06MP3/节奏/']
    dir_list = ['F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/']
    dir_list = []
    total_accuracy = 0
    total_num = 0
    result_path = 'e:/test_image/t/'
    # clear_dir(result_path)
    # 要测试的数量
    test_num = 100
    score = 0
    for dir in dir_list:
        file_list = os.listdir(dir)
        #shuffle(file_list)  # 将语音文件随机排列
        # file_list = ['视唱1-01（95）.wav']
        for filename in file_list:
            # clear_dir(image_dir)
            # wavname = re.findall(pattern,filename)[0]
            print(dir + filename)
            #plt = draw_start_end_time(dir + filename)
            plt = draw_baseline_and_note_on_cqt(dir + filename)
            tmp = os.listdir(result_path)

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
            #result_path = result_path + grade + "/"
            #plt.savefig(result_path + filename + '.jpg', bbox_inches='tight', pad_inches=0)
            plt.savefig(result_path + grade + "/" + filename + '.jpg', bbox_inches='tight', pad_inches=0)
            plt.clf()