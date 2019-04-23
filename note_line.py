import librosa
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
from create_base import *
import itertools
from dtw import dtw
from myDtw import *
from create_labels_files import *
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
                if np.min(CQT[j, i:i + 5]) == np.max(CQT) and np.max(CQT[j, i - 4:i - 1]) == np.min(CQT) and i - last > 5:
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

    print("1. result is {}".format(result))
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

    # 音高有偏离的情场（下偏离或上偏离）
    # (np.median(longest_note) > 25 and np.min(longest_note) == 20)  下偏离
    # (np.median(longest_note) < 23 and np.max(longest_note) > 30) 上偏离
    if (np.median(longest_note) > 25 and (np.min(longest_note) == 20 or np.max(longest_note) - np.median(longest_note) > 8)) or (np.median(longest_note) < 23 and np.max(longest_note) > 30):
        low_check = False
        high_check = False
        if np.median(longest_note) > 25 and (np.min(longest_note) == 20 or np.max(longest_note) - np.median(longest_note) > 8) :
            low_check = True
            high_check = False
        if np.median(longest_note) < 23 and np.max(longest_note) > 30:
            low_check = False
            high_check = True

        longest_note = []
        for i in range(len(result)):
            x = result[i]
            if i < len(result) - 1:
                next_frame = result[i + 1]
            else:
                next_frame = result[-1] + 20 if result[-1] + 20 < CQT.shape[1] else CQT.shape[1]
            # note_line = get_note_line_by_block_for_frames(x, CQT)
            # print("x,note_line is {},{}".format(x,note_line))
            longest_note_line = find_the_longest_note_line(x, next_frame, CQT,low_check,high_check)
            longest_note.append(longest_note_line)
    longest_note = check_by_median(longest_note)
    print("2. result is {}".format(result))
    return result,longest_note

def check_by_median(longest_note):
    result = []
    note_median = np.median(longest_note)
    for x in longest_note:
        if x - note_median >= 12:
            result.append(x -12)
        elif note_median - x >= 12:
            result.append(x + 12)
        else:
            result.append(x)
    return result

def augmention_by_shift(filename,shift):
    y, sr = librosa.load(filename)
    filepath, fullflname = os.path.split(filename)

    # 通过移动音调变声，14是上移14个半步，如果是-14，则是下移14个半步
    b = librosa.effects.pitch_shift(y, sr, n_steps=shift)
    new_file = fullflname.split(".")[0] + '-shift-' + str(shift)
    save_path_file = filepath + "/" + new_file + '.wav'
    librosa.output.write_wav(save_path_file, b, sr)
    return save_path_file

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
        #print("x,longest_note_line is {},{}".format(x, longest_note_line))
    #print("longest_note is {}".format(longest_note))
    # CQT[:,onsets_frames[1]:h] = -100
    total_frames_number = get_total_frames_number(filename)
    #print("total_frames_number is {}".format(total_frames_number))
    # librosa.display.specshow(CQT)
    base_frames = onsets_base_frames_for_note(filename)
    print("base_frames is {}".format(base_frames))
    total_score, onsets_score, notes_score = get_score(filename,result,longest_note,base_frames)
    old_filename = filename
    old_result = result
    old_longest_note = longest_note
    if notes_score < 40:
        old_total_scre = total_score
        old_onsets_score = onsets_score
        old_notes_score = notes_score
        filename = augmention_by_shift(filename, 6)
        result, longest_note = find_all_note_lines(filename)
        total_score, onsets_score, notes_score = get_score(filename, result, longest_note, base_frames)
        os.remove(filename)
        filename = old_filename
        if total_score < old_total_scre:
            result = old_result
            longest_note = old_longest_note
            total_score = old_total_scre
            onsets_score = old_onsets_score
            notes_score = old_notes_score

    onstm = librosa.frames_to_time(result, sr=sr)
    plt.subplot(3, 1, 1)
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


    return plt,total_score,onsets_score,notes_score

def get_score(filename,result,longest_note,base_frames):
    off = int(np.mean(base_frames) - np.mean(result))
    # off = int((base_notes[0] - longest_note[0]))
    base_frames = [x - off for x in base_frames]
    min_d, best_y, _ = get_dtw_min(result, base_frames, 65)
    onsets_score = 40 - int(min_d)
    if len(result)<len(base_frames)*0.75:
        onsets_score = onsets_score - int(40 * (len(base_frames) - len(result))/len(base_frames))
    print("onsets_score is {}".format(onsets_score))
    base_notes = base_note(filename)
    off = int(np.mean(base_notes) - np.mean(longest_note))
    #off = int((base_notes[0] - longest_note[0]))
    base_notes = [x - off for x in base_notes]
    print("base_notes is {}".format(base_notes))
    euclidean_norm = lambda x, y: np.abs(x - y)
    d, cost_matrix, acc_cost_matrix, path = dtw(longest_note, base_notes, dist=euclidean_norm)
    notes_score = 60 - int(d * np.sum(acc_cost_matrix.shape))
    if len(longest_note)<len(base_notes)*0.75 and notes_score < 55:
        notes_score = notes_score - int(60 * (len(base_notes) - len(longest_note))/len(base_notes))
    if notes_score <= 0:
        onsets_score = int(onsets_score / 2)
        notes_score = 0
    if notes_score >= 40 and onsets_score <= 5:
        onsets_score = int(40 * notes_score / 60)
    print("notes_score is {}".format(notes_score))
    total_score = onsets_score + notes_score
    print("total_score is {}".format(total_score))
    return total_score,onsets_score,notes_score

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

def find_the_longest_note_line(note_frame,next_frame,cqt,low_check=False,high_check=False):
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
    if best_note_line == 20 and low_check:
        best_note_line = 0
        for i in range(25, w - 20):
            a = sub_cqt[i]
            if list(a).count(cqt_max) > (next_frame - note_frame) * 0.2:
                if list(a).count(cqt_max) > (next_frame - note_frame) * 0.40:
                    best_note_line = i
                    break
                n_max = max([len(list(v)) for k, v in itertools.groupby(a)])
                b = dict([(k, len(list(v))) for k, v in itertools.groupby(a)])
                c = [k for k, v in b.items() if v == n_max and k == cqt_max]
                if len(c) > 0 and b.get(c[0]) > longest:
                    best_note_line = i
                    longest = b.get(c[0])

        if best_note_line == 0:
            for i in range(25, w - 20):
                a = sub_cqt[i]
                if list(a).count(cqt_max) > 3:
                    best_note_line = i
                    break

    if best_note_line >= 30 and high_check:
        best_note_line = 0
        for i in range(20, 30):
            a = sub_cqt[i]
            if list(a).count(cqt_max) > (next_frame - note_frame) * 0.2:
                if list(a).count(cqt_max) > (next_frame - note_frame) * 0.40:
                    best_note_line = i
                    break
                n_max = max([len(list(v)) for k, v in itertools.groupby(a)])
                b = dict([(k, len(list(v))) for k, v in itertools.groupby(a)])
                c = [k for k, v in b.items() if v == n_max and k == cqt_max]
                if len(c) > 0 and b.get(c[0]) > longest:
                    best_note_line = i
                    longest = b.get(c[0])

        if best_note_line == 0:
            for i in range(20, 30):
                a = sub_cqt[i]
                if list(a).count(cqt_max) > 3:
                    best_note_line = i
                    break
    old_best_note_line = best_note_line
    if best_note_line >= 35 and low_check:
        best_note_line = 0
        for i in range(20, 30):
            a = sub_cqt[i]
            if list(a).count(cqt_max) > (next_frame - note_frame) * 0.2:
                if list(a).count(cqt_max) > (next_frame - note_frame) * 0.40:
                    best_note_line = i
                    break
                n_max = max([len(list(v)) for k, v in itertools.groupby(a)])
                b = dict([(k, len(list(v))) for k, v in itertools.groupby(a)])
                c = [k for k, v in b.items() if v == n_max and k == cqt_max]
                if len(c) > 0 and b.get(c[0]) > longest:
                    best_note_line = i
                    longest = b.get(c[0])

        if best_note_line == 0:
            for i in range(20, 30):
                a = sub_cqt[i]
                if list(a).count(cqt_max) > 3:
                    best_note_line = i
                    break
        if best_note_line == 0:
            best_note_line = old_best_note_line
    return best_note_line

if __name__ == "__main__":

    # 保存新文件名与原始文件的对应关系
    files_list = []
    files_list_a = []
    files_list_b = []
    files_list_c = []
    files_list_d = []

    #y, sr = load_and_trim('F:/项目/花城音乐项目/样式数据/ALL/旋律/1.31MP3/旋律1.100分.wav')
    filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律2.1(80).wav'
    filename = 'F:/项目/花城音乐项目/样式数据/ALL/旋律/1.31MP3/旋律3.100分.wav'
    #filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律一（9）（100）.wav'
    #filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律一（14）（95）.wav'
    filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋律五（3）（63）.wav'
    filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏一（4）（96）.wav'
    filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋1录音4(78).wav'
    #filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋3王（80）.wav'
    #filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋4谭（95）.wav'

    filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律七（2）（90）.wav'
    filename = 'F:/项目/花城音乐项目/样式数据/4.18MP3/旋律/旋律3.1.wav'
    filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋2录音1(90).wav'
    filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋10罗（92）.wav'
    #filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律1_40422（95）.wav'
    #filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律七（2）（90）-shift-6.wav'
    filename = 'F:/项目/花城音乐项目/样式数据/4.18MP3/旋律/旋律1.1.wav'
    filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋律四（1）（20）.wav'
    filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋10熙(98).wav'
    filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋4卉(35).wav'




    #y, sr = load_and_trim(filename)

    # plt.colorbar(format='%+2.0f dB')
    # plt.title('Constant-Q power spectrogram (note)')
    plt,total_score,onsets_score,notes_score = get_note_with_cqt_rms(filename)

    plt.show()

    dir_list = ['F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/']
    #dir_list = ['F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/']
    # dir_list = ['e:/test_image/m1/A/']
    #dir_list = []
    total_accuracy = 0
    total_num = 0
    result_path = 'e:/test_image/n/'
    date = '3.06'
    new_old_txt = './onsets/' + date + 'best_dtw.txt'
    # clear_dir(result_path)
    # 要测试的数量
    test_num = 100
    score = 0
    for dir in dir_list:
        file_list = os.listdir(dir)
        # shuffle(file_list)  # 将语音文件随机排列
        #file_list = ['旋4.1(70).wav']
        for filename in file_list:
            # clear_dir(image_dir)
            # wavname = re.findall(pattern,filename)[0]
            print(dir + filename)
            # plt = draw_start_end_time(dir + filename)
            # plt = draw_baseline_and_note_on_cqt(dir + filename, False)
            plt,total_score,onsets_score,notes_score = get_note_with_cqt_rms(dir + filename)

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
                files_list_a.append([filename + ' - ' + grade, total_score, onsets_score, notes_score])
            elif int(score) >= 75:
                grade = 'B'
                files_list_b.append([filename + ' - ' + grade, total_score, onsets_score, notes_score])
            elif int(score) >= 60:
                grade = 'C'
                files_list_c.append([filename + ' - ' + grade, total_score, onsets_score, notes_score])
            elif int(score) >= 1:
                grade = 'D'
                files_list_d.append([filename + ' - ' + grade, total_score, onsets_score, notes_score])
            else:
                grade = 'E'
            # result_path = result_path + grade + "/"
            # plt.savefig(result_path + filename + '.jpg', bbox_inches='tight', pad_inches=0)
            #grade = 'A'
            plt.savefig(result_path + grade + "/" + filename + "-" + str(total_score) + '.jpg', bbox_inches='tight', pad_inches=0)
            plt.clf()

    t1 = np.append(files_list_a, files_list_b).reshape(len(files_list_a) + len(files_list_b), 4)
    t2 = np.append(files_list_c, files_list_d).reshape(len(files_list_c) + len(files_list_d), 4)
    files_list = np.append(t1, t2).reshape(len(t1) + len(t2), 4)

    write_txt(files_list, new_old_txt, mode='w')