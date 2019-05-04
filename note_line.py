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
    result = find_loss_by_rms_mean(result, rms,CQT)
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

def find_all_note_lines_v2(filename):
    y, sr = librosa.load(filename)
    rms = librosa.feature.rmse(y=y)[0]
    rms = [x / np.std(rms) for x in rms]
    #time = librosa.get_duration(filename=filename)
    #print("time is {}".format(time))
    CQT = librosa.amplitude_to_db(librosa.cqt(y, sr=16000), ref=np.max)
    w, h = CQT.shape
    print("w.h is {},{}".format(w, h))
    CQT = np.where(CQT > -22, np.max(CQT), np.min(CQT))
    onsets_frames, end_position = get_note_line_start(CQT)
    onsets_frames = find_loss_by_rms_mean(onsets_frames, rms, CQT)
    #onsets_frames = get_loss(onsets_frames, rms)
    onsets_frames,all_note_lines,longest_numbers = get_note_lines(CQT, onsets_frames)
    onsets_frames, all_note_lines, longest_numbers = get_loss_at_begin(CQT, onsets_frames, all_note_lines, longest_numbers)
    #onsets_frames, all_note_lines = del_false_note_lines(onsets_frames, all_note_lines, rms,CQT)
    all_note_lines = check_all_note_lines(onsets_frames, all_note_lines, CQT)
    onsets_frames, all_note_lines = del_end_range(onsets_frames, all_note_lines, rms)
    # base_notes = base_note(filename)
    # match_start, match_end, loss_before, loss_after = get_match_notes_trend(all_note_lines,base_notes)
    # if loss_before >= 3:
    #     onsets_frames = get_loss_before(onsets_frames,rms,int(np.median(onsets_frames)))
    #     onsets_frames, all_note_lines,longest_numbers = get_note_lines(CQT, onsets_frames)
    #     all_note_lines = check_all_note_lines(onsets_frames, all_note_lines, CQT)
    # if loss_after >= 3:
    #     onsets_frames = get_loss_after(onsets_frames, rms, int(np.median(onsets_frames)))
    #     onsets_frames, all_note_lines,longest_numbers = get_note_lines(CQT, onsets_frames)
    #     all_note_lines = check_all_note_lines(onsets_frames, all_note_lines, CQT)
    return onsets_frames,all_note_lines,longest_numbers

def get_loss_at_begin(cqt,result,all_note_lines,longest_numbers):
    min_cqt = np.min(cqt)
    first = result[0]
    best_longest, best_begin,best_row = 0,0,0
    w,h = cqt.shape
    if first > 60:
        f_cqt = cqt[:,:first]
        for row in range(10, w - 10):
            row_cqt = f_cqt[row]
            row_cqt = [1 if row_cqt[i] > min_cqt else 0 for i in range(len(row_cqt))]
            longest,begin = getLongestLine(row_cqt)
            if longest > best_longest:
                best_longest = longest
                best_begin = begin
                best_row = row

        if best_longest > 30:
            result.insert(0,best_begin)
            all_note_lines.insert(0,best_row)
            longest_numbers.insert(0, best_longest)
    return result,all_note_lines,longest_numbers

def getLongestLine(nums):
    sum1, res,begin,index = 0, 0,0,0
    for i in nums:
        #遇1加1，遇0置0
        sum1 = sum1*i + i
        if sum1 > res:
            #记录连续1的长度
            res = sum1
            begin = index - sum1
        index += 1
    return res,begin

def get_loss(result,rms):
    select_result = result.copy()

    rms_diff = [i if rms[i+2] - rms[i] > 0.2 else 0 for i in range(2, len(rms) - 2) ]
    for i in rms_diff:
        off = [np.abs(x - i) for x in select_result]
        min_off = np.min(off)

        if min_off > 20:
            select_result.append(i)
    select_result.sort()
    return select_result

def get_loss_before(result,rms,match_start):
    select_result = result.copy()
    rms_on_onset_frames_cqt = [rms[x] for x in result]
    mean_rms_on_frames = np.mean(rms_on_onset_frames_cqt)
    print("mean_rms_on_frames is {}".format(mean_rms_on_frames))
    rms_diff = np.diff(rms)
    rms_diff = [i for i in range(2, len(rms) - 2) if rms[i - 2] > rms[i - 1] and rms[i - 1] > rms[i] and rms[i + 1] > rms[i] and rms[i + 2] > rms[i + 1]  and rms[i-2] - rms[i] > 0.18 and rms[i+2] - rms[i] > 0.18]
    for i in rms_diff:
        off = [np.abs(x - i) for x in select_result]
        min_off = np.min(off)

        if min_off > 10 and i < match_start:
            select_result.append(i)
    select_result.sort()
    rms_on_onset_frames_cqt = [rms[x] for x in select_result]
    mean_rms_on_frames = np.mean(rms_on_onset_frames_cqt)
    print("mean_rms_on_frames is {}".format(mean_rms_on_frames))
    return select_result

def get_loss_after(result,rms,match_end):
    select_result = result.copy()
    rms_on_onset_frames_cqt = [rms[x] for x in result]
    mean_rms_on_frames = np.mean(rms_on_onset_frames_cqt)
    print("mean_rms_on_frames is {}".format(mean_rms_on_frames))
    rms_diff = np.diff(rms)
    rms_diff = [i for i in range(2, len(rms) - 2) if rms[i - 2] > rms[i - 1] and rms[i - 1] > rms[i] and rms[i + 1] > rms[i] and rms[i + 2] > rms[i + 1]  and rms[i-2] - rms[i] > 0.08 and rms[i+2] - rms[i] > 0.08]
    for i in rms_diff:
        off = [np.abs(x - i) for x in select_result]
        min_off = np.min(off)

        if min_off > 10 and i > match_end and i < len(rms)-50:
            select_result.append(i)
    select_result.sort()
    rms_on_onset_frames_cqt = [rms[x] for x in select_result]
    mean_rms_on_frames = np.mean(rms_on_onset_frames_cqt)
    print("mean_rms_on_frames is {}".format(mean_rms_on_frames))
    return select_result

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

'''
去掉位于音高线中间的节拍
'''
def check_false_by_rms_mean(result,rms,longest_note):
    select_result = []
    select_longest_note = []
    select_result.append(result[0])
    select_longest_note.append(longest_note[0])
    for i in range(1,len(result)):
        start = result[i] - 1
        end = result[i] + 2
        sub_rms = rms[start:end]
        #print("i,std is {},{}".format(i,np.std(sub_rms)))
        if i < len(result)-1:
            if longest_note[i] != longest_note[i+1] or (longest_note[i] == longest_note[i+1] and np.std(sub_rms) > 0.15):
                select_result.append(result[i])
                select_longest_note.append(longest_note[i])
        else:
            if longest_note[i-1] != longest_note[i] or (longest_note[i-1] == longest_note[i] and np.std(sub_rms) > 0.15):
                select_result.append(result[i])
                select_longest_note.append(longest_note[i])
    return select_result,select_longest_note

'''
找漏的
'''
def find_loss_by_rms_mean(result,rms,CQT):
    select_result = result.copy()
    rms_on_onset_frames_cqt = [rms[x] for x in result]
    mean_rms_on_frames = np.mean(rms_on_onset_frames_cqt)
    print("mean_rms_on_frames is {}".format(mean_rms_on_frames))
    rms_diff = np.diff(rms)
    for i in range(5, len(rms) - 5):
        off = [np.abs(x - i) for x in select_result]
        min_off = np.min(off)
        start = i - 1
        end = i + 2
        # 条件一：振幅有增加
        sub_rms = [rms[start + 1] - rms[start], rms[start + 2] - rms[start], rms[start + 3] - rms[start]]
        cond1 = np.max(sub_rms) > 0.1

        # 条件一：跨过均值线
        # cond2 = (rms[i] <= mean_rms_on_frames and rms[i+1]>mean_rms_on_frames) or (rms[i-1] <= mean_rms_on_frames and rms[i]>mean_rms_on_frames)
        cond2 = rms_diff[i] > 0.3

        if cond2 and min_off > 10:
            # print("np.std(sub_rms) is {}".format(np.std(sub_rms)))
            print("np.max(sub_rms) is {}".format(np.max(sub_rms)))
            if rms[i - 3] < rms[i]:
                select_result.append(i - 3)
            elif rms[i - 2] < rms[i]:
                select_result.append(i - 2)
            elif rms[i - 1] < rms[i]:
                select_result.append(i - 1)
            else:
                select_result.append(i)
    select_result.sort()
    rms_on_onset_frames_cqt = [rms[x] for x in select_result]
    mean_rms_on_frames = np.mean(rms_on_onset_frames_cqt)
    print("mean_rms_on_frames is {}".format(mean_rms_on_frames))
    return select_result

def get_note_with_cqt_rms(filename):
    y, sr = librosa.load(filename)
    rms = librosa.feature.rmse(y=y)[0]
    rms = [x / np.std(rms) for x in rms]
    time = librosa.get_duration(filename=filename)
    print("time is {}".format(time))
    CQT = librosa.amplitude_to_db(librosa.cqt(y, sr=16000), ref=np.max)
    result, longest_note = find_all_note_lines_v2(filename)
    #result, longest_note = check_false_by_rms_mean(result, rms, longest_note)
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
    if notes_score < 40 and total_score > 60:
        old_total_scre = total_score
        old_onsets_score = onsets_score
        old_notes_score = notes_score
        filename = augmention_by_shift(filename, 6)
        result, longest_note = find_all_note_lines(filename)
        #result,longest_note = check_false_by_rms_mean(result, rms, longest_note)
        total_score, onsets_score, notes_score = get_score(filename, result, longest_note, base_frames)
        os.remove(filename)
        filename = old_filename
        if total_score < old_total_scre:
            result = old_result
            longest_note = old_longest_note
            total_score = old_total_scre
            onsets_score = old_onsets_score
            notes_score = old_notes_score

    print("std is {},{}".format(np.std(result[0:5]),np.std(base_frames[0:5])))
    # if np.abs(np.std(result[0:5]) - np.std(base_frames[0:5]))/np.std(result[0:5]) > 0.3:
    #     total_score, onsets_score, notes_score = 0,0,0
    #     print("total_score, onsets_score, notes_score is {},{},{}".format(total_score, onsets_score, notes_score))
        #result = find_loss_by_rms_mean(result, rms)
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

def get_note_with_cqt_rms_v2(filename):
    y, sr = librosa.load(filename)
    rms = librosa.feature.rmse(y=y)[0]
    rms = [x / np.std(rms) for x in rms]
    time = librosa.get_duration(filename=filename)
    print("time is {}".format(time))
    CQT = librosa.amplitude_to_db(librosa.cqt(y, sr=16000), ref=np.max)
    result, longest_note,longest_numbers = find_all_note_lines_v2(filename)
    base_notes = base_note(filename)
    #result, longest_note = del_false_same(base_notes, result, longest_note)
    #result, longest_note = check_false_by_rms_mean(result, rms, longest_note)
    print("result is {},size is {}".format(result,len(result)))
    print("longest_note is {},size is {}".format(longest_note,len(longest_note)))
    print("longest_numbers is {},size is {}".format(longest_numbers,len(longest_numbers)))
        #print("x,longest_note_line is {},{}".format(x, longest_note_line))
    #print("longest_note is {}".format(longest_note))
    # CQT[:,onsets_frames[1]:h] = -100
    total_frames_number = get_total_frames_number(filename)
    #print("total_frames_number is {}".format(total_frames_number))
    # librosa.display.specshow(CQT)
    base_frames = onsets_base_frames_for_note(filename)
    print("base_frames is {},size is {}".format(base_frames,len(base_frames)))
    total_score, onsets_score, notes_score,trend_number,base_notes_number = get_score(filename,result,longest_note,base_frames)

    print("before is {}".format(result))
    result2, longest_note2 = del_false_same(base_notes, result, longest_note,longest_numbers)
    print("aafter is {}".format(result2))
    total_score2, onsets_score2, notes_score2, trend_number2, base_notes_number2 = get_score(filename, result2, longest_note2, base_frames)

    print("total_score2, total_score is {},{}".format(total_score2, total_score))
    if total_score2 != total_score:
        total_score, onsets_score, notes_score, trend_number, base_notes_number = total_score2, onsets_score2, notes_score2, trend_number2, base_notes_number2
        result, longest_note = result2, longest_note2
    print("total_score, onsets_score, notes_score is {},{},{}".format(total_score, onsets_score, notes_score))
    #print("std is {},{}".format(np.std(result[0:5]),np.std(base_frames[0:5])))
    # if np.abs(np.std(result[0:5]) - np.std(base_frames[0:5]))/np.std(result[0:5]) > 0.3:
    #     total_score, onsets_score, notes_score = 0,0,0
    #     print("total_score, onsets_score, notes_score is {},{},{}".format(total_score, onsets_score, notes_score))
        #result = find_loss_by_rms_mean(result, rms)
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


    return plt,total_score,onsets_score,notes_score,trend_number,base_notes_number

def get_score(filename,result,longest_note,base_frames):

    base_notes = base_note(filename)
    #result, longest_note = del_false_same(base_notes, result, longest_note)
    off = int(np.mean(base_frames) - np.mean(result))
    # off = int((base_notes[0] - longest_note[0]))
    base_frames = [x - off for x in base_frames]
    min_d, best_y, _ = get_dtw_min(result, base_frames, 65)
    onsets_score = 40 - int(min_d)
    if len(result)<len(base_frames)*0.75:
        onsets_score = onsets_score - int(40 * (len(base_frames) - len(result))/len(base_frames))
    print("onsets_score is {}".format(onsets_score))
    off = int(np.mean(base_notes) - np.mean(longest_note))
    #off = int((base_notes[0] - longest_note[0]))
    base_notes = [x - off for x in base_notes]
    print("base_notes is {}".format(base_notes))
    euclidean_norm = lambda x, y: np.abs(x - y)
    if(len(longest_note) != len(base_notes)):
        d, cost_matrix, acc_cost_matrix, path = dtw(longest_note, base_notes, dist=euclidean_norm)
        notes_score = 60 - int(d * np.sum(acc_cost_matrix.shape))
    else:
        each_note_score = 60 / len(longest_note)
        notes_score = 0
        for i in range(len(longest_note)):
            if np.abs(longest_note[i] - base_notes[i]) <= 1:
                notes_score += each_note_score
            elif np.abs(longest_note[i] - base_notes[i]) <= 2:
                notes_score += each_note_score * 0.5
        notes_score = int(notes_score)
    if len(longest_note)<len(base_notes)*0.75 and notes_score < 55:
        notes_score = notes_score - int(60 * (len(base_notes) - len(longest_note))/len(base_notes))
    if notes_score <= 0:
        onsets_score = int(onsets_score / 2)
        notes_score = 0
    if notes_score >= 40 and onsets_score <= 5:
        onsets_score = int(40 * notes_score / 60)
    total_score = onsets_score + notes_score
    trend_score,trend_number = check_notes_trend(longest_note,base_notes)
    print("trend_score is {}".format(60*trend_score))
    if 60*trend_score > notes_score:
        notes_score = int(60*trend_score)
        total_score = notes_score + onsets_score

    print("1.onsets_score is {}".format(onsets_score))
    print("1.notes_score is {}".format(notes_score))
    print("1.total_score is {}".format(total_score))
    #如果动态时间规整得分不高，且总分及格的，重新按相对音高计算得分
    if notes_score < 45 and onsets_score < 35:
            onsets_score = int(40 * trend_number / len(base_notes))
            notes_score = int(60 * trend_number/len(base_notes))
            total_score = notes_score + onsets_score
    # if trend_score/len(base_notes)<0.25 and np.max([onsets_score,notes_score]) < 30 :
    #     total_score = 0
    print("2.notes_score is {}".format(notes_score))
    print("2.total_score is {}".format(total_score))

    list_intersect_before, list_intersect, list_intersect_after = check_all_notes_trend(longest_note, base_notes)
    print("list_intersect_before,list_intersect,list_intersect_after is {},{},{}".format(list_intersect_before,list_intersect,list_intersect_after))
    # 如果总分小于85，则整体偏差分 30/85*total_score
    # if total_score < 85:
    #     gap = int(30/85*total_score)
    #     o_gap = int(gap* 0.4)
    #     n_gap = int(gap*0.6)
    #     total_score -= gap
    #     onsets_score -= o_gap
    #     notes_score -= n_gap
    return total_score,onsets_score,notes_score,trend_number,len(base_notes)

def del_false_same(base_notes,onset_frames,notes_lines,longest_numbers):
    select_onset_frames = [onset_frames[0]]
    select_notes_lines = [notes_lines[0]]
    same_index = get_the_index_of_same(base_notes)
    print("dddddd onset_frames is {}".format(onset_frames))
    if len(same_index) == 0:
        for i in range(1,len(notes_lines)):
            if notes_lines[i] != notes_lines[i-1]:
                select_onset_frames.append(onset_frames[i])
                select_notes_lines.append(notes_lines[i])
    else:
        for i in range(1, len(notes_lines)):
            if notes_lines[i] != notes_lines[i - 1]:
                select_onset_frames.append(onset_frames[i])
                select_notes_lines.append(notes_lines[i])
            else:
                #print("i ,longest_numbers[i-1] < onset_frames[i] - onset_frames[i-1] is {},{},{}".format(i,longest_numbers[i-1] ,onset_frames[i] - onset_frames[i-1]))
                if i in same_index or longest_numbers[i - 1] < onset_frames[i] - onset_frames[i - 1] - 2:
                    select_onset_frames.append(onset_frames[i])
                    select_notes_lines.append(notes_lines[i])
    print("dddddd onset_frames is {}".format(select_onset_frames))
    return select_onset_frames,select_notes_lines
def get_the_index_of_same(base_notes):
    same_index = []
    for i in range(1, len(base_notes)):
        if base_notes[i] == base_notes[i - 1]:
            same_index.append(i)
    return same_index

def check_notes_trend(longest_note,base_notes):
    diff_longest_note = []
    diff_base_notes = []
    diff_longest_note_str = ''
    diff_base_notes_str = ''
    for i in range(1,len(longest_note)):
        if longest_note[i] > longest_note[i-1]:
            tmp = 1
        if longest_note[i] == longest_note[i-1]:
            tmp = 0
        if longest_note[i] < longest_note[i-1]:
            tmp = 2
        diff_longest_note.append(str(tmp))
        diff_longest_note_str = ''.join(diff_longest_note)
    for i in range(1,len(base_notes)):
        if base_notes[i] > base_notes[i-1]:
            tmp = 1
        if base_notes[i] == base_notes[i-1]:
            tmp = 0
        if base_notes[i] < base_notes[i-1]:
            tmp = 2
        diff_base_notes.append(str(tmp))
        diff_base_notes_str = ''.join(diff_base_notes)

    list_intersect,number = getNumofCommonSubstr(diff_base_notes_str, diff_longest_note_str)
    print("diff_longest_note_str, diff_base_notes_str,intersect is {}==={}==={}".format(diff_longest_note_str, diff_base_notes_str,list_intersect))
    print("find intersect index is {}".format(diff_longest_note_str.find(list_intersect)))
    start = diff_longest_note_str.find(list_intersect)
    end = start + len(list_intersect)
    intersect_longest_note = longest_note[start:end+1]
    print("sub longest_note is {}".format(intersect_longest_note))
    start = diff_base_notes_str.find(list_intersect)
    end = start + len(list_intersect)
    intersect_base_notes = base_notes[start:end+1]
    print("sub base_notes is {}".format(intersect_base_notes))
    intersect_score = [1 if np.abs(intersect_longest_note[i] - intersect_base_notes[i])<1 else 0 for i in range(len(intersect_base_notes))]
    score = len(list_intersect)/len(diff_base_notes_str)

    return score,len(list_intersect)

def get_match_notes(diff_base_notes_str,diff_longest_note_str):
    list_intersect,number = getNumofCommonSubstr(diff_base_notes_str, diff_longest_note_str)
    print("diff_longest_note_str, diff_base_notes_str,intersect is {}==={}==={}".format(diff_longest_note_str, diff_base_notes_str,list_intersect))
    return list_intersect

def check_all_notes_trend(longest_note,base_notes):
    diff_longest_note = []
    diff_base_notes = []
    diff_longest_note_str = ''
    diff_base_notes_str = ''
    for i in range(1, len(longest_note)):
        if longest_note[i] > longest_note[i - 1]:
            tmp = 1
        if longest_note[i] == longest_note[i - 1]:
            tmp = 0
        if longest_note[i] < longest_note[i - 1]:
            tmp = 2
        diff_longest_note.append(str(tmp))
        diff_longest_note_str = ''.join(diff_longest_note)
    for i in range(1, len(base_notes)):
        if base_notes[i] > base_notes[i - 1]:
            tmp = 1
        if base_notes[i] == base_notes[i - 1]:
            tmp = 0
        if base_notes[i] < base_notes[i - 1]:
            tmp = 2
        diff_base_notes.append(str(tmp))
        diff_base_notes_str = ''.join(diff_base_notes)
    list_intersect = get_match_notes(diff_base_notes_str,diff_longest_note_str)
    start1 = diff_longest_note_str.find(list_intersect)
    end1 = start1 + len(list_intersect)

    start2 = diff_base_notes_str.find(list_intersect)
    end2 = start2 + len(list_intersect)
    list_intersect_before = []
    list_intersect_after = []
    if start1 >= 2 and start2 >=2:
        list_intersect_before = get_match_notes(diff_longest_note_str[0:start1],diff_base_notes_str[0:start2])
    if end1 >= 2 and end2 >=2:
        list_intersect_after = get_match_notes(diff_longest_note_str[end1+1:],diff_base_notes_str[end1+1:])

    return list_intersect_before,list_intersect,list_intersect_after

def get_match_notes_trend(longest_note,base_notes):
    diff_longest_note = []
    diff_base_notes = []
    for i in range(1,len(longest_note)):
        if longest_note[i] > longest_note[i-1]:
            tmp = 1
        if longest_note[i] == longest_note[i-1]:
            tmp = 0
        if longest_note[i] < longest_note[i-1]:
            tmp = 2
        diff_longest_note.append(str(tmp))
        diff_longest_note_str = ''.join(diff_longest_note)
    for i in range(1,len(base_notes)):
        if base_notes[i] > base_notes[i-1]:
            tmp = 1
        if base_notes[i] == base_notes[i-1]:
            tmp = 0
        if base_notes[i] < base_notes[i-1]:
            tmp = 2
        diff_base_notes.append(str(tmp))
        diff_base_notes_str = ''.join(diff_base_notes)

    list_intersect,number = getNumofCommonSubstr(diff_base_notes_str, diff_longest_note_str)
    print("diff_longest_note_str, diff_base_notes_str,intersect is {}==={}==={}".format(diff_longest_note_str, diff_base_notes_str,list_intersect))
    print("find intersect index is {}".format(diff_longest_note_str.find(list_intersect)))
    start = diff_longest_note_str.find(list_intersect)
    end = start + len(list_intersect)
    match_start = start
    match_end = end
    intersect_longest_note = longest_note[start:end+1]
    print("sub longest_note is {}".format(intersect_longest_note))
    start = diff_base_notes_str.find(list_intersect)
    end = start + len(list_intersect)
    loss_before = start
    loss_after = len(diff_base_notes_str)-end


    return match_start,match_end,loss_before,loss_after

"""
:type nums1: List[int]
:type nums2: List[int]
:rtype: List[int]
"""
def intersect(nums1, nums2):
  import collections
  a, b = map(collections.Counter, (nums1, nums2))
  return list((a & b).elements())


'''
求两个字符串的最长公共子串
思想：建立一个二维数组，保存连续位相同与否的状态
'''


def getNumofCommonSubstr(str1, str2):
    lstr1 = len(str1)
    lstr2 = len(str2)
    record = [[0 for i in range(lstr2 + 1)] for j in range(lstr1 + 1)]  # 多一位
    maxNum = 0  # 最长匹配长度
    p = 0  # 匹配的起始位

    for i in range(lstr1):
        for j in range(lstr2):
            if str1[i] == str2[j]:
                # 相同则累加
                record[i + 1][j + 1] = record[i][j] + 1
                if record[i + 1][j + 1] > maxNum:
                    # 获取最大匹配长度
                    maxNum = record[i + 1][j + 1]
                    # 记录最大匹配长度的终止位置
                    p = i + 1
    return str1[p - maxNum:p], maxNum


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

def get_note_line_start(cqt):
    result = []
    end_result = []
    w, h = cqt.shape
    cqt_max = np.max(cqt)
    cqt_min = np.min(cqt)
    end_position = 0
    for i in range(5, h - 15):
        # 该列存在亮点
        if np.max(cqt[:, i]) == cqt_max and i > end_position:
            sub_cqt = cqt[:, i:]
            # 从上往下逐行判断
            for j in range(w - 20, 10, -1):
                row_cqt = sub_cqt[j]
                # 如果该行存在连续8个亮点
                if np.min(row_cqt[0:8]) == cqt_max:
                    # 判断上下区域是否有音高线
                    hight_cqt = cqt[j + 10:j + 20, i:i + 10]
                    low_cqt = np.zeros(hight_cqt.shape)
                    if j - 20 > 0:
                        low_cqt = cqt[j - 20:j - 10, i:i + 10]
                    check_nioce_cqt_start = j - 6 if j - 6 > 0 else 0
                    check_nioce_cqt_end = j + 6 if j + 6 < w else w
                    check_nioce_cqt = cqt[check_nioce_cqt_start:check_nioce_cqt_end, i:i + 10]
                    check_nioce_low_result = False
                    check_nioce_high_result = False
                    for n in range(0, int((check_nioce_cqt_end - check_nioce_cqt_start - 1) / 2)):
                        if np.max(check_nioce_cqt[n]) == cqt_min:
                            check_nioce_low_result = True
                            break
                    for n in range(check_nioce_cqt_end - check_nioce_cqt_start - 1,
                                   int((check_nioce_cqt_end - check_nioce_cqt_start - 1) / 2), -1):
                        if np.max(check_nioce_cqt[n]) == cqt_min:
                            check_nioce_high_result = True
                            break

                    # 如果上下区域存在音高线，则说明不是噪声，则将起点作为节拍起点，同时找出连续区域的结束点
                    if check_nioce_low_result and check_nioce_high_result and (
                            np.max(hight_cqt) == cqt_max or np.max(low_cqt) == cqt_max):
                        if len(result) == 0:
                            result.append(i)
                            # print("i,j is {}==={}".format(i,j))
                        else:
                            offset = [np.abs(x - i) for x in result]
                            if np.min(offset) > 10:
                                result.append(i)
                                # print("i,j is {}==={}".format(i, j))
                        longest_end_position = 0
                        # 找出该连通块最大的长度
                        for k in range(j - 10, j):
                            k_cqt = sub_cqt[k]
                            # print("k_cqt shape is {},{}".format(k_cqt.shape[0],np.min(k_cqt)))
                            indexs_min = [i for i in range(len(k_cqt)) if k_cqt[i] == cqt_min]
                            index_min = 0
                            if len(indexs_min) > 0:
                                index_min = indexs_min[0]
                            end_position = i + index_min
                            if end_position > longest_end_position:
                                longest_end_position = end_position
                        end_result.append(longest_end_position)
                        check_nioce_high_result = False
                        check_nioce_low_result = False
                        break
    return result, end_result

def get_note_lines(cqt,result):
    note_lines = []
    w, h = cqt.shape
    cqt_max = np.max(cqt)
    cqt_min = np.min(cqt)
    selected_result = []
    longest_numbers = []
    for i in range(len(result)):
        x = result[i]
        if i < len(result) - 1:
            next = result[i + 1]
        else:
            next = h - 10
        sub_cqt = cqt[:, x:next]
        note_line, longest_num = get_longest_note_line(sub_cqt)
        if longest_num > 5 and longest_num < 30:
            note_lines.append(note_line)
            longest_numbers.append(longest_num)
            selected_result.append(x)
        elif longest_num >= 30:
            # 如果音高线以上有没有可分点
            best_other_onset_frames, best_note_numbers = find_other_note_line(x, note_line + 4, sub_cqt)
            if len(best_other_onset_frames) > 1:
                for i in range(len(best_other_onset_frames)):
                    note_lines.append(note_line)
                    longest_numbers.append(best_note_numbers[i])
                    selected_result.append(best_other_onset_frames[i])
            else:
                # 一分为二
                note_lines.append(note_line)
                longest_numbers.append(int(longest_num / 2))
                selected_result.append(x)

                note_lines.append(note_line)
                longest_numbers.append(int(longest_num / 2))
                selected_result.append(x + int(longest_num / 2))

        # # 从下往上逐行判断
        # for j in range(10,w-10):
        #     row_cqt = sub_cqt[j]
        #     #如果存在连续的亮点，长度大于8
        #     max_acount = np.sum(row_cqt == cqt_max)
        #     min_acount = np.sum(row_cqt == cqt_min)
        #     if max_acount > min_acount and np.sum(sub_cqt[j-1] == cqt_min) > min_acount:
        #         note_lines.append(j)
        #         selected_result.append(x)
        #         #print("x,j is {},{}".format(x,j))
        #         break
        # if j == w -11:
        #     if len(note_lines)>0:
        #         note_lines.append(note_lines[-1])
        #         selected_result.append(x)
        #     else:
        #         # 从下往上逐行判断
        #         for j in range(10, w - 10):
        #             row_cqt = sub_cqt[j]
        #             max_acount = np.sum(row_cqt == cqt_max)
        #             if max_acount > 5:
        #                 note_lines.append(j)
        #                 selected_result.append(x)
        #                 # print("x,j is {},{}".format(x,j))
        #                 break

    return selected_result, note_lines, longest_numbers

# 如果音高线以上有没有可分点
def find_other_note_line(onset_frame,note_line,sub_cqt):
    w,h = sub_cqt.shape
    print("w,h is {},{}".format(w,h))
    min_cqt = np.min(sub_cqt)
    max_cqt = np.max(sub_cqt)
    longest_num = 0
    best_note_numbers = []
    best_other_onset_frames = []

    for row in range(note_line+1,w-10):
        row_cqt = sub_cqt[row]
        if np.max(row_cqt) == max_cqt:
            row_cqt = [1 if row_cqt[i] > min_cqt else 0 for i in range(len(row_cqt))]
            other_onset_frames,note_numbers = get_all_small_note_line(onset_frame,row_cqt)
            lenght_other_onset_frmaes = len(other_onset_frames)
            if lenght_other_onset_frmaes > longest_num:
                longest_num = lenght_other_onset_frmaes
                best_note_numbers = note_numbers
                best_other_onset_frames = other_onset_frames
    return best_other_onset_frames,best_note_numbers

def get_all_small_note_line(start,nums):
    sum1, res = 0, 0
    index = 0
    onset_frames = []
    note_numbers = []
    start_index = 0
    last_sum1 = 0
    for i in nums:
        #遇1加1，遇0置0
        sum1 = sum1*i + i

        if sum1 == 1: # 记录开始点
            start_index = index
        index +=1

        if sum1 == 0 and last_sum1 > 10:
            onset_frames.append(start + start_index)
            note_numbers.append(last_sum1)
        last_sum1 = sum1

    return onset_frames,note_numbers

def get_longest_note_line(sub_cqt):
    w, h = sub_cqt.shape
    #print("w,h is {},{}".format(w, h))

    longest_num = 0
    note_line = 0
    if h > 0:
        min_cqt = np.min(sub_cqt)
        for row in range(10, w - 10):
            row_cqt = sub_cqt[row]
            row_cqt = [1 if row_cqt[i] > min_cqt else 0 for i in range(len(row_cqt))]
            total_continue = continueOne(row_cqt)
            if total_continue > longest_num:
                longest_num = total_continue
                note_line = row
    return note_line, longest_num


def continueOne(nums):
    sum1, res = 0, 0
    for i in nums:
        #遇1加1，遇0置0
        sum1 = sum1*i + i
        if sum1 > res:
            #记录连续1的长度
            res = sum1
    return res

def del_false_note_lines(onset_frames,all_note_lines,rms,CQT):
    select_note_lines = []
    select_note_lines.append(all_note_lines[0])
    select_onset_frames = []
    select_onset_frames.append(onset_frames[0])
    #print("max rms is {}".format(np.max(rms)))
    for i in range(1,len(onset_frames)):
        current_onset = onset_frames[i]
        last_onset = onset_frames[i-1]
        current_note = all_note_lines[i]
        last_note = all_note_lines[i-1]
        # 如果当前音高线等于前一个音高线
        if current_note == last_note:
            if np.max(CQT[current_note:current_note+4,current_onset-1]) == np.min(CQT):
                select_note_lines.append(all_note_lines[i])
                select_onset_frames.append(onset_frames[i])
                #print("np.abs(rms[i+1] - rms[i-1]) is {},{},{}".format(rms[current_onset+1] , rms[current_onset-1],np.abs(rms[current_onset+1] - rms[current_onset-1])))
            elif np.abs(rms[current_onset+1] - rms[current_onset-1]) > 0.3:
                select_note_lines.append(all_note_lines[i])
                select_onset_frames.append(onset_frames[i])
        else:
            select_note_lines.append(all_note_lines[i])
            select_onset_frames.append(onset_frames[i])
    return select_onset_frames,select_note_lines

def check_all_note_lines(onset_frames,all_note_lines,cqt):
    note_lines_median = np.median(all_note_lines)
    print("note_lines_median is {}".format(note_lines_median))
    selected_note_lines = []
    for i in range(len(all_note_lines)):
        x = all_note_lines[i]
        onset_frame = onset_frames[i]
        if np.abs(x - note_lines_median)>10:
            low_start = x + 1
            note_line = modify_some_note_line(cqt,onset_frame,low_start)
            if np.abs(note_line - note_lines_median)>10:
                note_line = int(note_lines_median)
            selected_note_lines.append(note_line)
        else:
            selected_note_lines.append(x)
    return selected_note_lines

def modify_some_note_line(cqt,onset_frame,low_start):
    note_line = 0
    w,h = cqt.shape
    cqt_max = np.max(cqt)
    cqt_min = np.min(cqt)

    x = onset_frame
    sub_cqt = cqt[:,x:x+15]
    # 从下往上逐行判断
    for j in range(low_start,w-10):
        row_cqt = sub_cqt[j]
        #如果存在连续的亮点，长度大于8
        max_acount = np.sum(row_cqt == cqt_max)
        min_acount = np.sum(row_cqt == cqt_min)
        if max_acount > min_acount and np.sum(sub_cqt[j-1] == cqt_min) > min_acount:
            note_line = j
            #print("x,j is {},{}".format(x,j))
            break
    return note_line

def del_end_range(onset_frames,all_note_lines,rms):
    length = len(rms)
    select_onset_frames = []
    select_all_note_lines = []
    for i in range(len(onset_frames)):
        onset_frame = onset_frames[i]
        note_line = all_note_lines[i]
        if onset_frame < length - 40:
            select_onset_frames.append(onset_frame)
            select_all_note_lines.append(note_line)
    return select_onset_frames,select_all_note_lines

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
    #filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律1_40422（95）.wav'
    #filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律七（2）（90）-shift-6.wav'
    filename = 'F:/项目/花城音乐项目/样式数据/4.18MP3/旋律/旋律1.1.wav'
    filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋1罗（96）.wav'
    #filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋10熙(98).wav'
    filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋4.4(0).wav'
    filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋2熙(0).wav'
    filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律一（13）（98）.wav'
    #filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋10罗（92）.wav'
    filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋10文(97).wav'
    filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋1.3(93).wav'
    filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋1熙(90).wav'
    filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋9熙(90).wav'
    filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋6谭（90）.wav'
    filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋1卉(80).wav'
    filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋律二（1）（90）.wav'
    filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋4谭（95）.wav'
    filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋4熙(90).wav'
    filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋8.2(90).wav'
    filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋1.1(96).wav'
    filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋律四（1）（20）.wav'
    filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋律五（4）（60）.wav'



    #y, sr = load_and_trim(filename)

    # plt.colorbar(format='%+2.0f dB')
    # plt.title('Constant-Q power spectrogram (note)')
    plt,total_score,onsets_score,notes_score,trend_number,base_notes_number = get_note_with_cqt_rms_v2(filename)

    plt.show()

    dir_list = ['F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/']
    #dir_list = ['F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/']
    dir_list = ['e:/test_image/m/A/']
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
    file_total = 0
    total_10 = 0
    total_15 = 0
    total_20 = 0
    for dir in dir_list:
        file_list = os.listdir(dir)
        # shuffle(file_list)  # 将语音文件随机排列
        #file_list = ['旋4.1(70).wav']
        file_total = len(file_list)
        for filename in file_list:
            # clear_dir(image_dir)
            # wavname = re.findall(pattern,filename)[0]
            print(dir + filename)
            # plt = draw_start_end_time(dir + filename)
            # plt = draw_baseline_and_note_on_cqt(dir + filename, False)
            plt,total_score,onsets_score,notes_score,trend_number,base_notes_number = get_note_with_cqt_rms_v2(dir + filename)

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
                files_list_a.append([filename + ' - ' + grade, total_score, onsets_score, notes_score,trend_number,base_notes_number])
            elif int(score) >= 75:
                grade = 'B'
                files_list_b.append([filename + ' - ' + grade, total_score, onsets_score, notes_score,trend_number,base_notes_number])
            elif int(score) >= 60:
                grade = 'C'
                files_list_c.append([filename + ' - ' + grade, total_score, onsets_score, notes_score,trend_number,base_notes_number])
            elif int(score) >= 1:
                grade = 'D'
                files_list_d.append([filename + ' - ' + grade, total_score, onsets_score, notes_score,trend_number,base_notes_number])
            else:
                grade = 'E'
            # result_path = result_path + grade + "/"
            # plt.savefig(result_path + filename + '.jpg', bbox_inches='tight', pad_inches=0)
            #grade = 'A'
            if np.abs(total_score - int(score)) > 15:
                plt.savefig(result_path + grade + "/" + filename + "-" + str(total_score) + '.jpg', bbox_inches='tight', pad_inches=0)
            plt.clf()
            if np.abs(total_score - int(score)) <= 10:
                total_10 += 1
            if np.abs(total_score - int(score)) <= 15:
                total_15 += 1
            if np.abs(total_score - int(score)) <= 20:
                total_20 += 1
    t1 = np.append(files_list_a, files_list_b).reshape(len(files_list_a) + len(files_list_b), 6)
    t2 = np.append(files_list_c, files_list_d).reshape(len(files_list_c) + len(files_list_d), 6)
    files_list = np.append(t1, t2).reshape(len(t1) + len(t2), 6)
    #stat_total = [str(file_total) + "-" + str(total_10) + "-" + str(total_15) + "-" + str(total_20)]
    #files_list = np.append(files_list, stat_total).reshape(len(files_list) + 1, 5)

    print("file_total,yes_total is {},{},{},{},{}".format(file_total, total_10, total_15, total_20,
                                                          total_10 / file_total))
    write_txt(files_list, new_old_txt, mode='w')
