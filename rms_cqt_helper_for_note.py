# -*- coding: UTF-8 -*-
from rms_helper_for_note import *
from cqt_helper_for_note import *
from create_base import *

def check_starts_with_max_index(filename):

    #获取所有的开始点
    start_indexs = get_cqt_start_indexs(filename)

    #获取所有的极大点
    rms,rms_diff, sig_ff, max_indexs = get_rms_max_indexs_for_onset(filename)

    selected_starts = []
    for i in range(len(start_indexs)):
        if i< len(start_indexs)-1:
            first = start_indexs[i]
            second = start_indexs[i+1]
            find_max = [x for x in max_indexs if x > first and x < second]
        else: # 最后一个
            first = start_indexs[i]
            find_max = [x for x in max_indexs if x > first]
        if len(find_max) > 0:
            selected_starts.append(first)
            # for x in find_max:
            #     selected_starts.append(x - 3)
    return selected_starts

def check_250(start_indexs,max_indaxs,rhythm_code):
    code = parse_rhythm_code(rhythm_code)
    index_250 = [i for i in range(1,len(code)) if code[i] == 250 and code[i-1] != 250]



def parse_rhythm_code(rhythm_code):
    code = rhythm_code
    index = 0
    code = code.replace(";", ',')
    code = code.replace("[", '')
    code = code.replace("]", '')
    if code.find("(") >= 0:
        tmp = [x for x in code.split(',')]
        for i in range(len(tmp)):
            if tmp[i].find("(") >= 0:
                index = i
                break
        code = code.replace("(", '')
        code = code.replace(")", '')
        code = code.replace("-", '')
        code = code.replace("--", '')
    code = [x for x in code.split(',')]
    # code = [int(x) for x in code]
    if index > 0:
        code[index - 1] += code[index]
        del code[index]
    return code

def cal_score_onset_and_note(filename,rhythm_code,pitch_code):
    # 标准节拍时间点
    start, end, total_frames_number = get_onset_frame_length(filename)
    base_frames = onsets_base_frames(rhythm_code, total_frames_number)

    base_frames_diff = np.diff(base_frames)

    start_indexs, maybe_start_indexs, starts_width = get_cqt_start_indexs(filename)
    #print("0 start_indexs is {},size {}".format(start_indexs, len(start_indexs)))
    #print("0 starts_width is {},size {}".format(starts_width, len(starts_width)))
    start_indexs = [x for x in start_indexs if x > start - 5 and x < end]
    raw_start_indexs = start_indexs.copy()
    start_indexs_diff = np.diff(start_indexs)

    rms, rms_diff, sig_ff, max_indexs = get_rms_max_indexs_for_onset(filename)
    max_indexs = [x for x in max_indexs if x > start - 5 and x < end]

    raw_start_indexs = start_indexs.copy()

    if len(start_indexs) > 2:
        dis_with_starts = get_dtw(start_indexs_diff, base_frames_diff)
        #print("dis_with_starts is {}".format(dis_with_starts))
        dis_with_starts_no_first = get_dtw(start_indexs_diff[1:], base_frames_diff)
        #print("dis_with_starts_no_first is {}".format(dis_with_starts_no_first))

        all_dis = [dis_with_starts,dis_with_starts_no_first]
        dis_min = np.min(all_dis)
        min_index = all_dis.index(dis_min)
        if 0 == min_index:
            onsets_frames = start_indexs
        elif 1 == min_index:
            sum_cols, sig_ff = get_sum_max_for_cols(filename)
            first_range = np.sum([1 if i > start and i < start + start_indexs_diff[0] and sum_cols[i] > sum_cols[start+3]*0.2 else 0 for i in range(start,start + start_indexs_diff[0])])  #根据节拍长度判断是否为真实节拍
            if len(start_indexs) == len(base_frames) + 1:
                onsets_frames = start_indexs[1:]
            elif first_range > base_frames_diff[0]*0.3:
                onsets_frames = start_indexs
            else:
                onsets_frames = start_indexs[1:]

    else:
        onsets_frames = max_indexs

    #print("3 onsets_frames is {},size is {}".format(onsets_frames, len(onsets_frames)))
    #print("3 starts_width is {},size is {}".format(starts_width, len(starts_width)))
    #print("3 rhythm_code is {},size is {}".format(rhythm_code, len(rhythm_code)))
    onsets_frames = get_losses_from_maybe_onset(raw_start_indexs, starts_width, maybe_start_indexs, rhythm_code,end)
    #print("4 onsets_frames is {},size is {}".format(onsets_frames, len(onsets_frames)))

    if len(onsets_frames) == 0:
        return 0,0,0,0,[],[],[]
    # if len(base_frames) > len(onsets_frames):
    #     onsets_frames = add_loss_small_onset(onsets_frames, rhythm_code)
    # if len(base_frames) > len(onsets_frames):
    #     rms,rms_diff, sig_ff, max_indexs = get_rms_max_indexs_for_onset(filename)
    #     max_indexs_on_middle = [x for x in max_indexs if x > start - 10 and x <end]
    #     if len(base_frames) == len(max_indexs_on_middle):
    #         onsets_frames = [x - 3 for x in max_indexs_on_middle]
    # elif len(onsets_frames) > len(base_frames):
    #     max_indexs = get_topN_rms_max_indexs_for_onset(filename, len(base_frames))
    #     onsets_frames = [x - 3 for x in max_indexs]

    base_frames = [x - (base_frames[0] - onsets_frames[0]) for x in base_frames]
    #print("base_frames is {},size is {}".format(base_frames, len(base_frames)))
    recognize_y = onsets_frames

    standard_y = base_frames.copy()

    y, sr = librosa.load(filename)
    CQT = librosa.amplitude_to_db(librosa.cqt(y, sr=16000), ref=np.max)
    CQT = np.where(CQT > -22, np.max(CQT), np.min(CQT))

    CQT = signal.medfilt(CQT, (3, 3))  # 二维中值滤波
    all_notes = get_all_notes(onsets_frames, CQT, end)
    base_notes = base_note(filename, pitch_code)
    notes_score, notes_detail_content = cal_note_score_by_diff(all_notes, base_notes)

    each_onset_score = 100 / len(standard_y)
    try:
        if len(standard_y) != len(recognize_y):
            xc,yc = get_match_lines(standard_y,recognize_y)
            #xc, yc = get_matched_onset_frames_compared(standard_y,recognize_y)
        else:
            xc, yc = standard_y, recognize_y
    except AssertionError as e:
        lenght = len(standard_y) if len(standard_y) <= len(recognize_y) else len(recognize_y)
        xc, yc = standard_y[:lenght], recognize_y[:lenght]
    std_number = len(standard_y) - len(xc) + len(recognize_y) - len(yc)
    # 未匹配节拍的序号
    loss_indexs = [i for i in range(len(standard_y)) if standard_y[i] not in xc]
    #多出节拍的序号
    ex_indexs = [i for i in range(len(recognize_y)) if recognize_y[i] not in yc]
    added_sum = 0
    added_indexs = []
    if len(loss_indexs) > 0:
        rms,rms_diff, sig_ff, max_indexs = get_rms_max_indexs_for_onset(filename)
        #print("0 max_indexs is {},size is {}".format(max_indexs, len(max_indexs)))
        for i in loss_indexs:
            if i < len(onsets_frames):
                maybe_indexs = [x for x in max_indexs if x > onsets_frames[i-1] and x < onsets_frames[i]]
                xc.append(standard_y[i])  # 补齐便为比较
                # yc.append(yc[i-1]+(standard_y[i] - standard_y[i-1]))
                if len(maybe_indexs) == 1:
                    yc.append(maybe_indexs[0]-3)
                    added_sum += 1
                    added_indexs.append(i)
                elif len(maybe_indexs) == 2:
                    yc.append(maybe_indexs[1]-3)
                    added_sum += 1
                    added_indexs.append(i)
                else:
                    yc.append(onsets_frames[i] + (standard_y[i] - standard_y[i-1]))
                yc.sort()

    xc.sort()
    yc.sort()

    code = parse_rhythm_code(rhythm_code)
    # if len(loss_indexs) == added_sum:
    #     loss_indexs = []
    #     added_sum = 0
    # code = [code[i] for i in range(len(code)) if i not in loss_indexs]
    min_d, onset_detail_content,a = get_detail(xc, yc, code, each_onset_score,total_frames_number,loss_indexs,added_indexs)

    if len(onsets_frames) == len(base_frames):
        lost_score = 0
    else:
        lost_score = int(each_onset_score * (len(loss_indexs) -added_sum))
    ex_score = int(each_onset_score * (len(onsets_frames) - len(code)))
    if ex_score < 0:
        print("--------0ipi9-99- is {}".format(ex_score))
        lost_score,ex_score = 0,0
        onset_score = 100 - lost_score - ex_score - int(min_d)
    else:
        onset_score = 100 - lost_score - ex_score - int(min_d)
    # score, lost_score, ex_score, min_d = get_score1(standard_y.copy(), recognize_y.copy(), len(base_frames),
    #                                                 onsets_frames_strength, min_d)
    score = int(onset_score*0.40) + int(notes_score*0.6)
    detail_content = onset_detail_content + " " + notes_detail_content
    print('最终得分为：{}'.format(score))
    return int(score), int(onset_score*0.40), int(notes_score*0.6),  xc, yc, detail_content

def get_detail(standard_y,recognize_y,codes,each_onset_score,total_frames_number,loss_indexs,added_indexs):
    #each_onset_score = 100/len(standard_y)
    score = 0
    total = 0
    a = 0
    b = 0
    c = 0
    detail_list = []
    continue_right = []
    for i in range(len(standard_y)):
        if i < len(standard_y)-1:
            offset =np.abs((recognize_y[i+1]-recognize_y[i]) /(standard_y[i+1] - standard_y[i]) -1)
        else:
            if total_frames_number == standard_y[i]: #分母为0的情况
                offset = 0
            else:
                offset = np.abs((total_frames_number - recognize_y[i]) / (total_frames_number - standard_y[i]) - 1)
                if offset > 0.2:
                    offset = 0
        standard_offset = get_code_offset(codes[i])
        if offset <= standard_offset:
            score = 0
            if i in loss_indexs and i not in added_indexs:
                detail_list.append("?")
            else:
                a += 1
                detail_list.append("1")
                continue_right.append(1)
        elif offset >= 1:
            score = each_onset_score
            if i in loss_indexs and i not in added_indexs:
                detail_list.append("?")
            else:
                b += 1
                detail_list.append("0")
                continue_right.append(0)
        else:
            score = each_onset_score * offset
            if i in loss_indexs and i not in added_indexs:
                detail_list.append("?")
            else:
                c += 1
                detail_list.append("-")
                continue_right.append(0)
        total +=score
    # if b == 1:
    #     total -= int(each_onset_score*0.5)
    str_detail_list = '识别的结果是：' + str(detail_list)
    str_detail_list = str_detail_list.replace("1","√")
    total_continue = continueOne(continue_right)
    if total_continue >= 4 and total > 20:
        total -= 15
        str_continue = '连续唱对的节拍数为' + str(total_continue) + '个。'
        str_detail_list = str_continue + str_detail_list

    #print(total_continue)
    detail_content = '未能匹配的节奏有'+ str(len(loss_indexs)-len(added_indexs)) + '，节奏时长偏差较大的有' + str(b) + '个，偏差较小的有' + str(c) + '个，偏差在合理区间的有' + str(a) + '个，' + str_detail_list
    return total,detail_content,a

def cal_note_score_by_diff(longest_note,base_notes):
    detail_content = ''
    a = 1
    b = 0
    c = 0
    longest_note = np.diff(longest_note)
    # off = int((base_notes[0] - longest_note[0]))
    each_note_score = 100 / len(base_notes)
    base_notes = np.diff(base_notes)
    #print("base_notes is {}".format(base_notes))
    euclidean_norm = lambda x, y: np.abs(x - y)
    detail_list = []
    detail_list.append("1")
    if (len(longest_note) != len(base_notes)):
        num_gap = len(base_notes)
        longest_note, base_notes = get_matched_note_lines_compared(longest_note, base_notes)
        notes_score = each_note_score
        for i in range(len(longest_note)):
            if np.abs(longest_note[i] - base_notes[i]) <=1:
                notes_score += each_note_score
                a += 1
                detail_list.append("1")
            elif np.abs(longest_note[i] - base_notes[i]) == 2:
                notes_score += each_note_score * 0.8
                b += 1
                detail_list.append("-")
            elif longest_note[i] * base_notes[i] > 0: # 相同趋势，即都为正，或都为负
                notes_score += each_note_score * 0.8
                b += 1
                detail_list.append("-")
            else:
                c += 1
                detail_list.append("0")
        notes_score = int(notes_score)
        #d, cost_matrix, acc_cost_matrix, path = dtw(longest_note, base_notes, dist=euclidean_norm)
        #notes_score = 60 - int(d * np.sum(acc_cost_matrix.shape))
        num_gap = num_gap - len(base_notes)
        detail_content += '识别的音高个数与标准不一致，未匹配的音高个数为' + str(num_gap) + '个。'
    else:
        #each_note_score = 100 / len(longest_note)
        notes_score = each_note_score
        for i in range(len(longest_note)):
            if np.abs(longest_note[i] - base_notes[i]) <= 1:
                notes_score += each_note_score
                a += 1
                detail_list.append("1")
            elif np.abs(longest_note[i] - base_notes[i]) == 2:
                notes_score += each_note_score * 0.8
                b += 1
                detail_list.append("-")
            elif longest_note[i] * base_notes[i] > 0: # 相同趋势，即都为正，或都为负
                notes_score += each_note_score * 0.8
                b += 1
                detail_list.append("-")
            else:
                c += 1
                detail_list.append("0")
        notes_score = int(notes_score)
    str_detail_list = '识别的结果是：' + str(detail_list)
    str_detail_list = str_detail_list.replace("1", "√")
    detail_content += '音高偏差较大的有' + str(c) + '个，偏差较小的有' + str(b) + '个，偏差在合理区间的有' + str(a) +  '个，' + str_detail_list
    return notes_score,detail_content


def get_code_offset(code):
    offset = 0
    code = re.sub("\D", "", code)  # 筛选数字
    code = int(code)
    if code >= 4000:
        offset = 1/32
    elif code >= 2000:
        offset = 1/4
    elif code >= 1000:
        offset = 1/8
    elif code >= 500:
        offset = 1/4
    elif code >= 250:
        offset = 1/2
    return offset

def continueOne(nums):
    sum1, res = 0, 0
    for i in nums:
        #遇1加1，遇0置0
        sum1 = sum1*i + i
        if sum1 > res:
            #记录连续1的长度
            res = sum1
    return res