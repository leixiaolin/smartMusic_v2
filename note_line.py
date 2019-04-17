import librosa
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
from create_base import *
import itertools
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

def find_all_note_lines(filename):
    y, sr = librosa.load(filename)
    rms = librosa.feature.rmse(y=y)[0]
    rms = [x / np.std(rms) for x in rms]
    #time = librosa.get_duration(filename=filename)
    #print("time is {}".format(time))
    CQT = librosa.amplitude_to_db(librosa.cqt(y, sr=16000), ref=np.max)
    w, h = CQT.shape
    print("w.h is {},{}".format(w, h))
    CQT = np.where(CQT > -20, np.max(CQT), np.min(CQT))
    result = []
    last = 0

    # print("max is {}".format(np.max(CQT)))
    for i in range(15, h - 10):
        is_ok = 0
        last_j = 100
        for j in range(w - 1, 15, -1):
            if CQT[j, i] == np.max(CQT) and CQT[j, i - 1] == np.min(CQT):
                if np.min(CQT[j, i:i + 5]) == np.max(CQT) and np.max(CQT[j, i - 4:i - 1]) == np.min(
                        CQT) and i - last > 5:
                    if np.min(CQT[j, i:i + 10]) == np.max(CQT) and np.mean(CQT[j, i - 5:i - 1]) == np.min(CQT):
                        # print("3... is {},{},{}".format(CQT[j, i - 4:i - 3],CQT[j, i - 3:i-2],i))
                        is_ok += 2
                        break
                    if last_j - j > 10:
                        is_ok += 1
                        last_j = j
                elif np.min(CQT[j, i:i + 5]) == np.max(CQT) and is_ok == 1:
                    is_ok += 1
                # elif np.min(CQT[j, i+1:i + 2]) == np.max(CQT):
                #     result.append(i)
        if rms[i + 1] > rms[i] and is_ok > 1:
            if len(result) == 0:
                result.append(i)
                last = i
            elif i - result[-1] > 10:
                result.append(i)
                last = i
        elif rms[i + 1] - rms[i - 1] > 0.75 and i > 50 and i < len(rms) - 45:
            if len(result) == 0:
                result.append(i)
                last = i
            elif i - result[-1] > 8:
                result.append(i)
                last = i

    # print("result is {}".format(result))
    longest_note = []
    for i in range(len(result)):
        x = result[i]
        if i < len(result) - 1:
            next_frame = result[i + 1]
        else:
            next_frame = result[-1] + 20 if result[-1] + 20 < CQT.shape[1] else CQT.shape[1]
        #note_line = get_note_line_by_block_for_frames(x, CQT)
        # print("x,note_line is {},{}".format(x,note_line))
        longest_note_line = find_the_longest_note_line(x, next_frame, CQT)
        longest_note.append(longest_note_line)
    return result,longest_note

def get_note_with_cqt_rms(filename):
    y, sr = librosa.load(filename)
    rms = librosa.feature.rmse(y=y)[0]
    rms = [x / np.std(rms) for x in rms]
    time = librosa.get_duration(filename=filename)
    print("time is {}".format(time))
    CQT = librosa.amplitude_to_db(librosa.cqt(y, sr=16000), ref=np.max)
    result, longest_note = find_all_note_lines(filename)
    print("result is {}".format(result))
    print("longest_note is {}".format(longest_note))
    onstm = librosa.frames_to_time(result, sr=sr)
        #print("x,longest_note_line is {},{}".format(x, longest_note_line))
    #print("longest_note is {}".format(longest_note))
    # CQT[:,onsets_frames[1]:h] = -100
    plt.subplot(3, 1, 1)
    total_frames_number = get_total_frames_number(filename)
    #print("total_frames_number is {}".format(total_frames_number))
    # librosa.display.specshow(CQT)
    base_frames = onsets_base_frames_for_note(filename)
    print("base_frames is {}".format(base_frames))
    base_notes = base_note(filename)
    base_notes = [x - (base_notes[0] - longest_note[0]) for x in base_notes]
    print("base_notes is {}".format(base_notes))
    CQT,base_notes = add_base_note_to_cqt_for_filename_by_base_notes(filename,result,result[0],CQT,longest_note)
    base_notes = [x + int(np.mean(longest_note) - np.mean(base_notes)) for x in base_notes]
    #print("base_notes is {}".format(base_notes))
    librosa.display.specshow(CQT, y_axis='cqt_note', x_axis='time')
    print(np.max(y))
    # onstm = librosa.frames_to_time(onsets_frames, sr=sr)
    plt.vlines(onstm, 0, sr, color='y', linestyle='solid')

    plt.subplot(3, 1, 2)

    plt.text(onstm[0],1,result[0])
    max_rms = np.max(rms)
    # rms = np.diff(rms)
    times = librosa.frames_to_time(np.arange(len(rms)))
    rms_on_onset_frames_cqt = [rms[x] for x in result]
    mean_rms_on_frames = np.mean(rms_on_onset_frames_cqt)
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

def get_note_line_by_block_for_frames(note_frame,cqt):
    w,h = cqt.shape
    cqt_max = np.max(cqt)
    start = note_frame
    for i in range(3):
        start +=i
        if np.max(cqt[15:,start]) == cqt_max:
            break
    sub_cqt = cqt[15:,start:start + 3]
    #sub_cqt = cqt[15:, note_frame+2:note_frame + 5]
    note_line = 0
    for i in range(15,w-15):
        if np.min(sub_cqt[i]) == cqt_max and np.min(sub_cqt[i+1]) == cqt_max or np.min(cqt[i,start:start+10]) == cqt_max :
            note_line = i
            return note_line
    return note_line

def find_the_longest_note_line(note_frame,next_frame,cqt):
    w,h = cqt.shape
    cqt_max = np.max(cqt)
    cqt_min = np.min(cqt)
    sub_cqt = cqt[:,note_frame:next_frame]
    longest = 0
    best_note_line = 0
    for i in range(20,w -20):
        a = sub_cqt[i]
        if list(a).count(cqt_max) > (next_frame - note_frame)*0.2:
            if list(a).count(cqt_max) > (next_frame - note_frame)*0.40:
                best_note_line = i
                break
            n_max = max([len(list(v)) for k, v in itertools.groupby(a)])
            b = dict([(k, len(list(v))) for k, v in itertools.groupby(a)])
            c = [k for k, v in b.items() if v == n_max and k == cqt_max]
            if len(c)>0 and b.get(c[0]) > longest:
                best_note_line = i
                longest = b.get(c[0])

    if best_note_line == 0:
        for i in range(20, w - 20):
            a = sub_cqt[i]
            if list(a).count(cqt_max) > 3:
                best_note_line = i
                break
    return best_note_line

if __name__ == "__main__":
    #y, sr = load_and_trim('F:/项目/花城音乐项目/样式数据/ALL/旋律/1.31MP3/旋律1.100分.wav')
    filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律2.1(80).wav'
    filename = 'F:/项目/花城音乐项目/样式数据/ALL/旋律/1.31MP3/旋律3.100分.wav'
    #filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律一（9）（100）.wav'
    #filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律一（14）（95）.wav'
    filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋律五（3）（63）.wav'
    filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏一（4）（96）.wav'
    filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋1录音4(78).wav'
    filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋3王（80）.wav'
    filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋4谭（95）.wav'
    filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋4文(75).wav'
    filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律6.4(90).wav'
    filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋1录音4(78).wav'




    #y, sr = load_and_trim(filename)

    # plt.colorbar(format='%+2.0f dB')
    # plt.title('Constant-Q power spectrogram (note)')
    plt = get_note_with_cqt_rms(filename)
    plt.show()

    dir_list = ['F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/']
    #dir_list = ['F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/']
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
            #grade = 'A'
            plt.savefig(result_path + grade + "/" + filename + '.jpg', bbox_inches='tight', pad_inches=0)
            plt.clf()