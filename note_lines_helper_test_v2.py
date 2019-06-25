import librosa
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
from scipy import interpolate
from create_base import *
from myDtw import *
from note_lines_helper import *
# 1. Get the file path to the included audio example
# Sonify detected beat events
# 定义加载语音文件并去掉两端静音的函数

test_codes = np.array(['[1000,1000;2000;1000,500,500;2000]',
                  '[2000;1000,1000;500,500,1000;2000]',
                  '[1000,1000;500,500,1000;1000,1000;2000]',
                  '[1000,--(1000);1000,--(1000);500,250,250,1000;--(1000),1000]',
                  '[500;1000,500,1000,500;500,500,500,250,250,500,500;250,250,500,500,1000]',
                  '[1000,--(1000);1000,--(1000);1000,-(500),500;1000,1000]',
                  '[750,250,500,500,500,-(500);500,1000,500,500,-(500);750,250,500,500,500,-(500)]',
                  '[500,1000,500,500,250,250;1000,500,750,250,500;3000]',
                  '[500,500,500;1000,500;500,500,500;1500;500,500,500;1000,500;500;1000;1500]',
                  '[500,500,1000;500,500;1000;375,125,250,250,375,125,250,250;500,500,1000]'])
test_note_codes = np.array(['[3,3,3,3,3,3,3,5,1,2,3]',
                       '[5,5,3,2,1,2,5,3,2]',
                       '[5,5,3,2,1,2,2,3,2,6-,5-]',
                       '[5,1+,7,1+,2+,1+,7,6,5,2,4,3,6,5]',
                       '[3,6,7,1+,2+,1+,7,6,3]',
                       '[1+,7,1+,2+,3+,2+,1+,7,6,7,1+,2+,7,1+,7,1+,2+,1+]',
                       '[5,6,1+,6,2,3,1,6-,5-]',
                       '[5,5,6,5,6,5,1,3,0,2,2,5-,2,1]',
                       '[3,2,1,2,1,1,2,3,4,5,3,6,5,5,3]',
                       '[3,4,5,1+,7,6,5]'])
test_rhythm_codes = np.array(['[500,500,1000;500,500,1000;500,500,750,250;2000]',
                        '[1000,1000;500,500,1000;1000,500,500; 2000]',
                        '[1000,1000;500,500,1000;500,250,250,500,500;2000]',
                        '[500,1000,500;250,250,250,250,500,500;500,500,500,500;2000]',
                        '[1000;500,500,1000;500,500,500,500;2000]',
                        '[500;500,500,500,500;500,500,500,500;500,500,500,500;250,250,250,250,500]',
                        '[1000,750,250,2000;500,500,500,500,2000]',
                        '[1000,1000,1000,500,500;1000,1000,1000,--(1000);1000,1000,1000;1000,4000]',
                        '[1500,500,500,500;2500,500;1000,500,500,500,500;2500,500]',
                        '[500,500;1500,500,500,500;2000]'])

def get_code(index,type):

    if type == 1:
        code = test_codes[index]
    if type == 2:
        code = test_rhythm_codes[index]
    if type == 3:
        code = test_note_codes[index]
    # code = code.replace(";", ',')
    # code = code.replace("[", '')
    # code = code.replace("]", '')
    # code = [x for x in code.split(',')]
    return code

def get_onsets_index_by_filename(filename):
    if filename.find("节奏10") >= 0 or filename.find("节奏十") >= 0 or filename.find("节奏题十") >= 0 or filename.find("节奏题10") >= 0 or filename.find("节10") >= 0:
        return 9
    elif filename.find("节奏1") >= 0 or filename.find("节奏一") >= 0 or filename.find("节奏题一") >= 0 or filename.find("节奏题1") >= 0 or filename.find("节1") >= 0:
        return 0
    elif filename.find("节奏2") >= 0 or filename.find("节奏二") >= 0 or filename.find("节奏题二") >= 0 or filename.find("节奏题2") >= 0 or filename.find("节2") >= 0:
        return 1
    elif filename.find("节奏3") >= 0 or filename.find("节奏三") >= 0 or filename.find("节奏题三") >= 0 or filename.find("节奏题3") >= 0 or filename.find("节3") >= 0:
        return 2
    elif filename.find("节奏4") >= 0 or filename.find("节奏四") >= 0 or filename.find("节奏题四") >= 0 or filename.find("节奏题4") >= 0 or filename.find("节4") >= 0:
        return 3
    elif filename.find("节奏5") >= 0 or filename.find("节奏五") >= 0 or filename.find("节奏题五") >= 0 or filename.find("节奏题5") >= 0 or filename.find("节5") >= 0:
        return 4
    elif filename.find("节奏6") >= 0 or filename.find("节奏六") >= 0 or filename.find("节奏题六") >= 0 or filename.find("节奏题6") >= 0 or filename.find("节6") >= 0:
        return 5
    elif filename.find("节奏7") >= 0 or filename.find("节奏七") >= 0 or filename.find("节奏题七") >= 0 or filename.find("节奏题7") >= 0 or filename.find("节7") >= 0:
        return 6
    elif filename.find("节奏8") >= 0 or filename.find("节奏八") >= 0 or filename.find("节奏题八") >= 0 or filename.find("节奏题8") >= 0 or filename.find("节8") >= 0:
        return 7
    elif filename.find("节奏9") >= 0 or filename.find("节奏九") >= 0 or filename.find("节奏题九") >= 0 or filename.find("节奏题9") >= 0 or filename.find("节9") >= 0:
        return 8
    else:
        return -1

def get_onsets_index_by_filename_rhythm(filename):
    if filename.find("旋律10") >= 0 or filename.find("旋律十") >= 0 or filename.find("视唱十") >= 0 or filename.find("视唱10") >= 0 or filename.find("旋10") >= 0:
        return 9
    elif filename.find("旋律1") >= 0 or filename.find("旋律一") >= 0 or filename.find("视唱一") >= 0 or filename.find("视唱1") >= 0 or filename.find("旋1") >= 0:
        return 0
    elif filename.find("旋律2") >= 0 or filename.find("旋律二") >= 0 or filename.find("视唱二") >= 0 or filename.find("视唱2") >= 0 or filename.find("旋2") >= 0:
        return 1
    elif filename.find("旋律3") >= 0 or filename.find("旋律三") >= 0 or filename.find("视唱三") >= 0 or filename.find("视唱3") >= 0 or filename.find("旋3") >= 0:
        return 2
    elif filename.find("旋律4") >= 0 or filename.find("旋律四") >= 0 or filename.find("视唱四") >= 0 or filename.find("视唱4") >= 0 or filename.find("旋4") >= 0:
        return 3
    elif filename.find("旋律5") >= 0 or filename.find("旋律五") >= 0 or filename.find("视唱五") >= 0 or filename.find("视唱5") >= 0 or filename.find("旋5") >= 0:
        return 4
    elif filename.find("旋律6") >= 0 or filename.find("旋律六") >= 0 or filename.find("视唱六") >= 0 or filename.find("视唱6") >= 0 or filename.find("旋6") >= 0:
        return 5
    elif filename.find("旋律7") >= 0 or filename.find("旋律七") >= 0 or filename.find("视唱七") >= 0 or filename.find("视唱7") >= 0 or filename.find("旋7") >= 0:
        return 6
    elif filename.find("旋律8") >= 0 or filename.find("旋律八") >= 0 or filename.find("视唱八") >= 0 or filename.find("视唱8") >= 0 or filename.find("旋8") >= 0:
        return 7
    elif filename.find("旋律9") >= 0 or filename.find("旋律九") >= 0 or filename.find("视唱九") >= 0 or filename.find("视唱9") >= 0 or filename.find("旋9") >= 0:
        return 8
    else:
        return -1

def load_and_trim(path):
    audio, sr = librosa.load(path)
    energy = librosa.feature.rmse(audio)
    frames = np.nonzero(energy >= np.max(energy) / 5)
    indices = librosa.core.frames_to_samples(frames)[1]
    audio = audio[indices[0]:indices[-1]] if indices.size else audio[0:0]

    return audio, sr

def get_longest_note_line(sub_cqt):
    w,h = sub_cqt.shape
    print("w,h is {},{}".format(w,h))

    longest_num = 0
    note_line = 0
    if h > 0:
        min_cqt = np.min(sub_cqt)
        for row in range(10,w-10):
            row_cqt = sub_cqt[row]
            row_cqt = [1 if row_cqt[i] > min_cqt else 0 for i in range(len(row_cqt))]
            total_continue = continueOne(row_cqt)
            if total_continue > longest_num:
                longest_num = total_continue
                note_line = row
    return note_line,longest_num


def continueOne(nums):
    sum1, res = 0, 0
    for i in nums:
        #遇1加1，遇0置0
        sum1 = sum1*i + i
        if sum1 > res:
            #记录连续1的长度
            res = sum1
    return res

def getLongestLine(nums):
    sum1, res,begin,index = 0, 0,0,1
    for i in nums:
        #遇1加1，遇0置0
        sum1 = sum1*i + i
        if sum1 > res:
            #记录连续1的长度
            res = sum1
            begin = index - sum1
        index += 1
    return res,begin



def get_frame_length(cqt_col_diff):
    end = len(cqt_col_diff)
    for i in range(2,len(cqt_col_diff)):
        if np.max(cqt_col_diff[:i-1]) == 0 and cqt_col_diff[i] >0.1:
            start = i

    for i in range(len(cqt_col_diff)-2,0,-1):
        if np.max(cqt_col_diff[i+1:]) == 0 and cqt_col_diff[i] >0.1:
            end = i
    return start,end,end-start

def get_the_nearly_base_frame(frame,base_frames,last_frame,last_base_frame):
    offset = [np.abs(frame - x) for x in base_frames]
    nearly_index = offset.index(np.min(offset))
    if base_frames[nearly_index] > frame: #位于右侧
        if nearly_index -1 <= 0:
            return base_frames[nearly_index], nearly_index
        else:
            left_base_frame = base_frames[nearly_index-1]
            right_base_frame = base_frames[nearly_index]
            left_gap = left_base_frame - last_base_frame - (frame - last_frame)
            right_gap = right_base_frame - last_base_frame - (frame - last_frame)
            if left_gap < right_gap:
                return left_base_frame, nearly_index-1
            else:
                return right_base_frame, nearly_index

    elif base_frames[nearly_index] < frame: #位于左侧
        if nearly_index +1 >= len(base_frames):
            return base_frames[nearly_index], nearly_index
        else:
            left_base_frame = base_frames[nearly_index]
            right_base_frame = base_frames[nearly_index +1]
            left_gap = left_base_frame - last_base_frame - (frame - last_frame)
            right_gap = right_base_frame - last_base_frame - (frame - last_frame)
            if left_gap < right_gap:
                return left_base_frame,nearly_index
            else:
                return right_base_frame, nearly_index+1
    else:
        return base_frames[nearly_index],nearly_index

def get_all_the_nearly_base_frame(onset_frames,base_frames):
    all_nearly_base_frames = []
    all_nearly_indexs = []
    for x in onset_frames:
        if x == onset_frames[0]:
            nearly_base_frames,nearly_index =  base_frames[0],0
            last_nearly_base_frames, last_nearly_index = nearly_base_frames,nearly_index
            last_frame = x
            if nearly_base_frames not in all_nearly_base_frames:
                all_nearly_base_frames.append(nearly_base_frames)
                all_nearly_indexs.append(nearly_index)
        else:
            last_base_frame = last_nearly_base_frames
            nearly_base_frames, nearly_index = get_the_nearly_base_frame(x,base_frames,last_frame,last_base_frame)
            last_nearly_base_frames, last_nearly_index = nearly_base_frames, nearly_index
            last_frame = x
            if nearly_base_frames not in all_nearly_base_frames:
                all_nearly_base_frames.append(nearly_base_frames)
                all_nearly_indexs.append(nearly_index)
    return all_nearly_base_frames,all_nearly_indexs

def add_loss_by_sub_dtw(rms_max_indexs,starts_index, base_frames):
    select_starts_index = starts_index
    all_nearly_base_frames, all_nearly_indexs = get_all_the_nearly_base_frame(starts_index, base_frames)
    for i in range(1,len(all_nearly_indexs)):
        if all_nearly_indexs[i] - all_nearly_indexs[i-1]>1:
            start = starts_index[i-1]
            end = starts_index[i]
            start_base = all_nearly_base_frames[i-1]
            end_base = all_nearly_base_frames[i]
            sub_rms_max_indexs = [x for x in rms_max_indexs if x > start and x < end]
            sub_rms_max_indexs.insert(0,start)
            sub_rms_max_indexs.append(end)
            sub_rms_max_indexs = [x-sub_rms_max_indexs[0] for x in sub_rms_max_indexs if x > sub_rms_max_indexs[0]]
            sub_base_frames = [x for x in base_frames if x >=start_base and x <= end_base]
            sub_base_frames = [x - sub_base_frames[0] for x in sub_base_frames if x > sub_base_frames[0] and x < sub_base_frames[-1]]
            #系数
            rate = (end - start)/(end_base - start_base)
            sub_base_frames = [int(x*rate) for x in sub_base_frames]

            for x in sub_base_frames:
                offset = [np.abs(s - x) for s in sub_rms_max_indexs]
                min_index = offset.index(np.min(offset))
                selected = x + start_base
                select_starts_index.append(selected)
    if all_nearly_indexs[-1] < len(base_frames):
        start = starts_index[-1]
        start_base = all_nearly_base_frames[-1]
        end_base = base_frames[-1]
        sub_rms_max_indexs = [x for x in rms_max_indexs if x > start]
        sub_rms_max_indexs.insert(0, start)
        sub_rms_max_indexs = [x - sub_rms_max_indexs[0] for x in sub_rms_max_indexs if x > sub_rms_max_indexs[0]]
        sub_base_frames = [x for x in base_frames if x >= start_base and x <= end_base]
        sub_base_frames = [x - sub_base_frames[0] for x in sub_base_frames if
                           x > sub_base_frames[0]]
        # 系数
        if end_base != start and end_base != start_base:
            rate = (end_base - start) / (end_base - start_base)
            sub_base_frames = [int(x * rate) for x in sub_base_frames]

        for x in sub_base_frames:
            offset = [np.abs(s - x) for s in sub_rms_max_indexs]
            min_index = offset.index(np.min(offset))
            selected = x + start_base
            select_starts_index.append(selected)

    select_starts_index.sort()
    return select_starts_index

# type_index = get_onsets_index_by_filename_rhythm(filename)
# rhythm_code = get_code(type_index, 2)
# pitch_code = get_code(type_index, 3)
# def get_melody_score_v2(filename,rhythm_code,pitch_code):
#     y, sr = load_and_trim(filename)
#     y,sr = librosa.load(filename)
#     rms = librosa.feature.rmse(y=y)[0]
#     rms = [x / np.std(rms) for x in rms]
#     time = librosa.get_duration(filename=filename)
#     print("time is {}".format(time))
#     CQT = librosa.amplitude_to_db(librosa.cqt(y, sr=16000), ref = np.max)
#     w,h = CQT.shape
#     print("w.h is {},{}".format(w,h))
#     #onsets_frames = get_real_onsets_frames_rhythm(y)
#     CQT = np.where(CQT > -22, np.max(CQT), np.min(CQT))
#     max_cqt = np.max(CQT)
#     min_cqt = np.min(CQT)
#     result = []
#     starts = []
#     last_i = 1
#     for i in range(1,h-4):
#         col_cqt = CQT[10:,i]
#         before_col_cqt = CQT[10:,i-1]
#         after_col_cqt = CQT[10:, i+3]
#         max_sum = np.sum([1 if x > np.min(col_cqt) else 0 for x in col_cqt])
#         before_max_sum = np.sum([1 if x > np.min(before_col_cqt) else 0 for x in before_col_cqt])
#         after_before_max_sum = np.sum([1 if x > np.min(after_col_cqt) else 0 for x in after_col_cqt])
#         #sum = np.sum(np.array(col_cqt) - np.array(before_col_cqt))
#         sum = np.sum([1 if (before_col_cqt[i] == min_cqt and col_cqt[i] == max_cqt) and max_sum > 0.7*before_max_sum else 0 for i in range(len(col_cqt))])
#         start = np.sum([1 if max_sum > 1.3*before_max_sum and before_max_sum <= 1 and after_before_max_sum > 0else 0 for i in range(len(col_cqt))])
#         result.append(sum)
#         if start == 0:
#             starts.append(start)
#         else:
#             if i - last_i > 8:
#                 starts.append(start)
#                 last_i = i
#
#         # for n in range(w-1,10,-1):
#         #     if i >10:
#         #         before_row = CQT[n,i-4:i]
#         #         after_row = CQT[n, i:i+8]
#         #         if np.max(before_row) == min_cqt and np.min(after_row) == max_cqt:
#         #             starts.append(start)
#         #             last_i = i
#
#
#     result = [x if x>0 else 0 for x in result]
#     result = [x/np.max(result) for x in result]
#     result = [x if x>0.1 else 0 for x in result]
#
#
#     starts = [x/np.max(starts) for x in starts]
#     starts = [starts[i] if starts[i] ==1 and starts[i-1] == 0 else 0 for i in range(1,len(starts))]
#     starts.insert(0,0)
#     starts_index = [i for i in range(len(starts)) if starts[i]>0]
#     for i in range(1,len(result)):
#         offset = [np.abs(i-x) for x in starts_index]
#         if (result[i] > 0.46 or (i > 10 and result[i] > 0.2 and np.max(result[i-8:i-2]) == 0))and np.min(offset) > 8:
#             starts_index.append(i)
#     starts_index.sort()
#     rms = np.diff(rms, 1)
#     rms = [x if x > 0 else 0 for x in rms]
#     rms = [x / np.max(rms) for x in rms]
#     rms_max_indexs = [i for i in range(1, len(rms) - 1) if
#                       rms[i] > rms[i - 1] and rms[i] > rms[i + 1] and rms[i] > 0.1 or rms[i] > 0.6]
#
#     print("rms_max_index is {}".format(rms_max_indexs))
#     start, end, length = get_frame_length(result)
#     base_frames = onsets_base_frames_rhythm(rhythm_code, length)
#     base_frames = [x - (base_frames[0] - start) for x in base_frames]
#     select_starts_index = add_loss_by_sub_dtw(rms_max_indexs, starts_index.copy(), base_frames)
#     print("select_starts_index is {}".format(select_starts_index))
#     # plt.vlines(select_starts_index, 0, np.max(result), color='r', linestyle='dashed')
#     onsets_frames = select_starts_index
#     onsets_frames, note_lines, note_times = get_note_lines_v2(CQT, onsets_frames)
#     onsets_frames = [onsets_frames[i] for i in range(len(note_times)) if note_times[i] > 5]
#     note_lines = [note_lines[i] for i in range(len(note_times)) if note_times[i] > 5]
#     note_times = [note_times[i] for i in range(len(note_times)) if note_times[i] > 5]
#     base_notes = base_note(filename, pitch_code)
#     score, onset_score, note_score, detail_content = cal_score_v1(filename, onsets_frames, note_lines, base_frames,
#                                                                      base_notes, note_times, rhythm_code)
#     if note_score > 55 and onset_score < 30:
#         detail_content = detail_content + ',音高得分较好，节奏方面需要进一步加强'
#         onset_score = int(note_score / 60 * 40)
#         score = onset_score + note_score
#     print('total_score,onset_score, note_scroe is ' + str(score) + ' ' + str(onset_score) + ' ' + str(note_score))
#
#     return score, onset_score, note_score, detail_content,CQT,rms,result,onsets_frames,starts_index

if __name__ == "__main__":
    # y, sr = load_and_trim('F:/项目/花城音乐项目/样式数据/ALL/旋律/1.31MP3/旋律1.100分.wav')
    filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律2.1(80).wav'
    filename = 'F:/项目/花城音乐项目/样式数据/ALL/旋律/1.31MP3/旋律3.100分.wav'
    filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律一（9）（100）.wav'
    # filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律一（14）（95）.wav'
    # filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/节奏/节2罗（75）.wav'
    # filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏一（4）（96）.wav'
    # filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋2熙(0).wav'
    filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋1罗（96）.wav'
    #filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋10罗（92）.wav'
    #filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋2录音4(72).wav'
    #filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋10罗（92）.wav'
    # filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律一（13）（98）.wav'
    #filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋1熙(90).wav'
    # filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋1谭（98）.wav'
    #filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋3.3(96).wav'
    filename = 'F:/项目/花城音乐项目/样式数据/6.24MP3/旋律/两只老虎20190624-2939.wav'

    #filename = 'F:/项目/花城音乐项目/样式数据/6.21MP3/旋律/两只老虎20190621-7278.wav'  # 只是节奏基本对，旋律全错 score, onset_score, note_scroe is 57,37,20
    #filename = 'F:/项目/花城音乐项目/样式数据/6.21MP3/旋律/两只老虎20190621-6314.wav' #唱的不是同一首歌，旋律也相差较大 score, onset_score, note_scroe is 50,32,18
    #filename = 'F:/项目/花城音乐项目/样式数据/6.21MP3/旋律/20190621-3533.wav' #节奏对，旋律全错  score, onset_score, note_scroe is 45,32,13
    #filename = 'F:/项目/花城音乐项目/样式数据/6.21MP3/旋律/两只老虎20190621-9983.wav' #节奏对，旋律错 score, onset_score, note_scroe is 60,37,23
    #filename = 'F:/项目/花城音乐项目/样式数据/6.21MP3/旋律/20190621-3858.wav'# 速度偏快，节奏旋律基本正确 score, onset_score, note_scroe is 61,36,25
    #filename = 'F:/项目/花城音乐项目/样式数据/6.21MP3/旋律/两只老虎20190621-6264.wav' #唱速不稳定，旋律正确，节奏错误 score, onset_score, note_scroe is 44,33,11
    #filename = 'F:/项目/花城音乐项目/样式数据/6.21MP3/旋律/20190621-3192.wav' #只是节奏基本对，旋律全错 score, onset_score, note_scroe is 50,32,18
    #filename = 'F:/项目/花城音乐项目/样式数据/6.21MP3/旋律/两只老虎20190621-6143.wav' #速度偏快，节奏旋律基本正确 score, onset_score, note_scroe is 10,4,6
    #filename = 'F:/项目/花城音乐项目/样式数据/6.21MP3/旋律/两只老虎20190621-9805.wav' #节奏对，旋律错，score, onset_score, note_scroe is 52,32,20

    filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋1录音3(90).wav'
    filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋9.1(73).wav'

    rhythm_code = '[500,500,500,500;500,500,500,500;500,500,1000;500,500;1000]'
    pitch_code = '[1,2,3,1,1,2,3,1,3,4,5,3,4,5]'
    type_index = get_onsets_index_by_filename_rhythm(filename)
    rhythm_code = get_code(type_index, 2)
    pitch_code = get_code(type_index, 3)

    score1, onset_score1, note_score1, detail_content, CQT, rms, result, onsets_frames,starts_index = get_melody_score_v2(filename,rhythm_code,pitch_code)

    plt.subplot(4,1,1)
    librosa.display.specshow(CQT ,x_axis='time')
    plt.vlines(onsets_frames, 0, CQT.shape[0], color='r', linestyle='dashed')


    plt.subplot(4,1,2)
    start,end,length = get_frame_length(result)
    print("==============start,end is {},{}".format(start,end))
    base_frames = onsets_base_frames_rhythm(rhythm_code, length)
    base_frames = [x - (base_frames[0] - start-1) for x in base_frames]
    #result, max_indexs = filter_hold_local_max(result, base_frames)
    max_indexs = [i for i in range(len(result)) if i>start and i < end -3 and result[i] > 0 ]
    onsets_frames, note_lines, note_times = get_note_lines_v2(CQT, max_indexs)

    print("==============note_lines is {}".format(note_lines))
    print("==============note_times is {}".format(note_times))
    note_lines_diff = np.diff(note_lines)
    max_indexs = [max_indexs[i+1] for i in range(len(note_lines_diff)) if np.abs(note_lines_diff[i]) >1]
    times = range(len(result))
    plt.plot(times, result)
    plt.xlim(0, np.max(times))
    plt.vlines(start, 0, np.max(result), color='r', linestyle='dashed')
    plt.vlines(end, 0, np.max(result), color='r', linestyle='dashed')
    plt.vlines(base_frames, 0, np.max(result), color='b', linestyle='dashed')
    #plt.vlines(max_indexs, 0, np.max(result), color='r', linestyle='solid')
    #plt.plot(times, starts, color='r', linestyle='solid')
    plt.vlines(starts_index, 0, np.max(result), color='y', linestyle='dashed')
    all_nearly_base_frames,all_nearly_indexs = get_all_the_nearly_base_frame(onsets_frames,base_frames)
    print("starts_index is {}".format(starts_index))
    print("all_nearly_base_frames is {}".format(all_nearly_base_frames))
    print("all_nearly_indexs is {}".format(all_nearly_indexs))

    plt.subplot(4,1,3)
    plt.xlim(0, np.max(times))
    plt.vlines(onsets_frames, 0, np.max(result), color='r', linestyle='dashed')

    plt.subplot(4,1,4)
    together = [result[i] if result[i] > rms[i] else rms[i] for i in range(len(result))]
    plt.plot(times, together)
    plt.xlim(0, np.max(times))
    plt.show()