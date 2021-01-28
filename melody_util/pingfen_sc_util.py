from melody_util.parselmouth_sc_util import get_best_candidate_names_by_moved, get_matched_positions, get_numbered_musical_notation,get_pitch_by_parselmouth,get_all_numbered_notation_and_offset
import numpy as np
from melody_util.cnn_bilstm_attention_model_for_onset_predict import predict_onset_frames_from_single_file
from parselmouth_util import get_all_absolute_pitchs_for_filename,get_encode_pitch_seque,parse_rhythm_code_for_absolute_pitch,get_matched_detail_absolute_pitch,check_from_second_candidate_names
import time


def parse_rhythm_code(rhythm_code):
    code = rhythm_code
    indexs = []
    code = code.replace(";", ',')
    code = code.replace("[", '')
    code = code.replace("]", '')
    tmp = [x for x in code.split(',')]
    for i in range(len(tmp)):
        if tmp[i].find("(") >= 0:
            indexs.append(i)
    while code.find("(") >= 0:
        code = code.replace("(", '')
        code = code.replace(")", '')
        code = code.replace("-", '')
        code = code.replace("--", '')
    code = [x for x in code.split(',')]
    result = []
    for i in range(len(code)):
        if i in indexs:
            continue
        elif i + 1 not in indexs:
            result.append(code[i])
        else:
            t = int(code[i]) + int(code[i + 1])
            result.append(t)
    return result

def parse_pitch_code(pitch_code):
    code = pitch_code
    code = code.replace("[", '')
    code = code.replace("]", '')
    code = [x for x in code.split(',')]
    result = []
    for i in range(len(code)):
        c = code[i]
        result.append(c)
    return result

'''
standard_notations = '3,3,3,3,3,3,3,5,1,2,3'
test_notations = ['C4', 'C3', 'C3', 'C4', 'C4', 'C4', 'C4', 'C4', None, 'C4', 'D4', 'G#3', 'G#3', 'A#3', 'C4', 'C4']
'''
def calculate_note_score(standard_notations, test_notations, threshold_score):
    standard_notations = standard_notations.replace("[", '')
    standard_notations = standard_notations.replace("]", '')
    code = standard_notations.split(',')
    code = [s[0] for s in code]
    # standard_notations = ''.join(standard_notations)

    # 获取平移后最匹配的音高及匹配矩阵
    best_numbered_notations, d = get_best_candidate_names_by_moved(standard_notations, test_notations)

    # 获取匹配位置信息
    standard_positions, test_positions = get_matched_positions(d)

    # 对未匹配的音高进行两次偏差匹配
    offset_matched_positions = get_match_by_offset(standard_notations, test_notations, standard_positions,test_positions)

    each_symbol_score = threshold_score / len(code)
    total_score = int(len(standard_positions) * each_symbol_score) # 完全匹配的可以计满分
    total_score = total_score + int(len(offset_matched_positions) * each_symbol_score * 0.5) # 偏差匹配的可以计一半分

    detail = get_matched_detail(standard_notations,standard_positions,offset_matched_positions)

    test_notations_without_none = [tn for tn in test_notations if tn is not None]
    ex_total = len(test_notations_without_none) - len(code)
    ex_rate = ex_total / len(code)
    if len(test_notations_without_none) > len(standard_notations):
        if ex_rate > 0.4:  # 节奏个数误差超过40%，总分不超过50分
            total_score = total_score if total_score < threshold_score * 0.50 else threshold_score * 0.50
            detail = "多唱节奏个数误差超过40%，总分不得超过50分"
        elif ex_rate > 0.3:  # 节奏个数误差超过30%，总分不超过65分（超过的）（30-40%）
            total_score = total_score if total_score < threshold_score * 0.65 else threshold_score * 0.65
            detail = "多唱节奏个数误差超过30%，总分不得超过65分"
        elif ex_rate > 0.2:  # 节奏个数误差超过20%，总分不超过80分（超过的）（20-30%）
            total_score = total_score if total_score < threshold_score * 0.80 else threshold_score * 0.80
            detail = "多唱节奏个数误差超过20%，总分不得超过80分"
        elif ex_rate > 0:  # 节奏个数误差不超过20%，总分不超过90分（超过的）（0-20%）
            total_score = total_score if total_score < threshold_score * 0.90 else threshold_score * 0.90
            detail = "多唱节奏个数误差在（1-20%），总分不得超过90分"
    return int(total_score), detail


'''
对未匹配的音高进行两次偏差匹配
'''
def get_match_by_offset(standard_notations, test_notations, standard_positions, test_positions):
    standard_notations = standard_notations.split(',')
    standard_notations = [s[0] for s in standard_notations]

    no_match_positions = [i for i in range(1, len(standard_notations)) if i not in standard_positions]

    offset_matched_positions = []
    for nmp in no_match_positions:
        before_position = [i for i, p in enumerate(standard_positions) if p < nmp]  # 该未匹配前面的匹配位置
        after_position = [i for i, p in enumerate(standard_positions) if p > nmp]  # 该未匹配后面的匹配位置

        if len(before_position) == 0:  # 该未匹配前面没有匹配位置
            start = 0
        else:
            start = before_position[-1]
            start = test_positions[start] + 1

        if len(after_position) == 0:  # 该未匹配后面没有匹配位置
            end = len(test_notations) + 1
        else:
            end = after_position[0]
            end = test_positions[end]
        tmp = test_notations[start:end]

        tmp = [t for t in tmp if t is not None]
        numbered_tmp = [get_numbered_musical_notation(n)[0] for n in tmp]
        if len(numbered_tmp) == 0:
            continue
        notation_no_matched = standard_notations[nmp]
        offset = [np.abs(int(notation_no_matched) - int(nt)) for nt in numbered_tmp]
        if np.min(offset) <= 1:
            offset_matched_positions.append(nmp)
    return offset_matched_positions


def get_matched_detail(standard_notations,standard_positions,offset_matched_positions):
    standard_notations = standard_notations.split(',')
    standard_notations = [s[0] for s in standard_notations]
    detail_list = np.zeros(len(standard_notations))
    # start_index = 0
    # positions = standard_positions + offset_matched_positions
    standard_positions = [p-1 for p in standard_positions]
    offset_matched_positions = [p - 1 for p in offset_matched_positions]
    for index in standard_positions:
        # index = base_symbols[start_index:].index(l)
        detail_list[index] = 1
    for index in offset_matched_positions:
        # index = base_symbols[start_index:].index(l)
        detail_list[index] = 2

    str_detail_list = '识别的结果是：' + str(detail_list)
    str_detail_list = str_detail_list.replace("1", "√")
    str_detail_list = str_detail_list.replace("0", "×")
    str_detail_list = str_detail_list.replace("2", "!")

    return str_detail_list

def get_earch_rhythm_duration(rhythm_code,start,end):
    rhythm_sum = np.sum(rhythm_code)
    durations = []
    rate = (end - start)/rhythm_sum
    for rc in rhythm_code:
        duration = int(rc * rate)
        durations.append(duration)
    return durations

def get_standard_frames(rhythm_code,start,end):
    standard_frames = [start]
    rhythm_sum = np.sum(rhythm_code)
    rate = (end - start)/rhythm_sum
    for code in rhythm_code:
        duration = int(code * rate)
        frame = standard_frames[-1] + duration
        standard_frames.append(frame)
    return standard_frames


def check_onset_matched_result(standard_duration, onset_frames,current_check_point, offset_threshold):
    result = False
    range_begin = current_check_point
    range_end = current_check_point + 4 if current_check_point + 4 < len(onset_frames) else len(onset_frames)
    for start in range(range_begin,range_end):
        for end in range(start,range_end):
            duration = onset_frames[end] - onset_frames[start]
            if np.abs((standard_duration - duration) / standard_duration) <= offset_threshold:
                current_check_point = end
                result = True

'''
获取匹配区间内的可对比帧序号
'''
def get_frames_for_check(onset_frames,match_begin,match_end,pitch,best_test_notations,note_match_result):
    result = [match_begin]
    for i in range(0,len(onset_frames)-1):
        if note_match_result == 1:
            if match_begin <= onset_frames[i+1] <= match_end and onset_frames[i+1] not in result and np.abs(int(pitch) - int(best_test_notations[i]) <=1):   # 只取音高偏差不大的
                result.append(onset_frames[i+1])
        else:
            if match_begin <= onset_frames[i + 1] <= match_end and onset_frames[i + 1] not in result:   # 不考虑音高
                result.append(onset_frames[i + 1])
    return result

def check_duration(standard_duration,frames_for_check,offset_threshold):
    if len(frames_for_check) <= 1:
        return False,-1,100
    offset_rate = 100
    for i in range(len(frames_for_check)-1):
        first_frame = frames_for_check[i]
        durations = [frame - first_frame for frame in frames_for_check]
        offset = [np.abs((d - standard_duration)/standard_duration) for d in durations]
        match_index = -1
        flag = False
        if np.min(offset) <= offset_threshold:
            match_index = offset.index(np.min(offset))
            flag = True
            offset_rate = np.min(offset)
            return flag,match_index,offset_rate
    return flag,match_index,offset_rate


def calculate_onset_score(rhythm_code,onset_frames,standard_notations,best_test_notations,start,end,note_match_positions,threshold_score):
    standard_notations_bak = standard_notations
    standard_notations = standard_notations.replace("[", '')
    standard_notations = standard_notations.replace("]", '')
    standard_notations = standard_notations.split(',')
    standard_notations = [s[0] for s in standard_notations]
    each_symbol_score = threshold_score / len(standard_notations)

    best_test_notations = [get_numbered_musical_notation(n)[0] for n in best_test_notations]

    offset_threshold = 0.60
    # 根据开始时间和结束时间，计算每种节奏类型的参考时长
    standard_durations = get_earch_rhythm_duration(rhythm_code,start,end)

    # 对参考节奏逐一比对
    total_score = 0
    standard_positions = []
    duration_threshold = 100 # 对比区间偏差为100，即为1秒
    current_index = 0
    for i,sd in enumerate(standard_durations):
        match_begin = onset_frames[current_index]
        match_end = match_begin + sd + duration_threshold

        # 异常处理
        if i >= len(standard_durations) or i >= len(note_match_positions):
            continue

        # 获取对比区间内的可对比帧序号
        notation = standard_notations[i]
        note_match_result = note_match_positions[i]
        frames_for_check = get_frames_for_check(onset_frames,match_begin,match_end,notation,best_test_notations,note_match_result)
        # print("i,sd,notation is {},{},{}".format(i,sd,notation))
        # print("match_begin,match_end is {},{}".format(match_begin,match_end))
        # print("frames_for_check is {}".format(frames_for_check))

        # 计算节奏偏差
        # 如果是最后一个，放宽对偏差的限制
        if i == len(standard_durations)-1:
            offset_threshold = 1
        match_result,match_index,offset_rate = check_duration(sd,frames_for_check,offset_threshold)
        if match_result is True:
            total_score += each_symbol_score
            standard_positions.append(i+1)
            current_index += match_index
            # print('true')
        else:
            current_index += 1
            # print('false')
        if current_index >= len(onset_frames):
            # 判断最后一个
            if np.abs((end - onset_frames[-1]) - standard_durations[-1]) <= 1:
                total_score += each_symbol_score
                standard_positions.append(len(standard_durations) - 1)
            break
    detail = get_matched_detail(standard_notations_bak, standard_positions, [])


    # ex_total = len(all_symbols) - len(base_symbols)
    # ex_rate = ex_total / len(base_symbols)
    # if len(all_symbols) > len(base_symbols):
    #     if ex_rate > 0.4:                                # 节奏个数误差超过40%，总分不超过50分
    #         total_score = total_score if total_score < threshold_score*0.50 else threshold_score*0.50
    #         detail = detail + "，多唱节奏个数误差超过40%，总分不得超过总分的0.50"
    #     elif ex_rate > 0.3:                             # 节奏个数误差超过30%，总分不超过65分（超过的）（30-40%）
    #         total_score = total_score if total_score < threshold_score*0.65 else threshold_score*0.65
    #         detail = detail + "，多唱节奏个数误差超过30%，总分不得超过总分的0.65"
    #     elif ex_rate > 0.2:                             # 节奏个数误差超过20%，总分不超过80分（超过的）（20-30%）
    #         total_score = total_score if total_score < threshold_score*0.80 else threshold_score*0.80
    #         detail = detail + "，多唱节奏个数误差超过20%，总分不得超过总分的0.80"
    #     elif ex_rate > 0:                                           # 节奏个数误差不超过20%，总分不超过90分（超过的）（0-20%）
    #         total_score = total_score if total_score < threshold_score*0.90 else threshold_score*0.90
    #         detail = detail + "，多唱节奏个数误差在（1-20%），总分不得超过总分的0.90"
    return int(total_score),detail

def get_match_positions_from_detail_str(detail):
    tmp = detail.split("[")[-1]
    match_reult = tmp.replace("]", '').split(".")
    positions = []
    for m in match_reult:
        if m.strip() == "√":
            positions.append(1)
        else:
            positions.append(0)
    return positions
def get_score(filename,rhythm_code,pitch_code):
    s = time.time()
    onset_frames = predict_onset_frames_from_single_file(filename)

    if len(onset_frames) == 0:
        note_total_score, note_detail,onset_total_score, onset_detail = 0,'该音频未能识别出音符起始点，请核对音频是否正确或有噪声干扰',0,'该音频未能识别出音符起始点，请核对音频是否正确或有噪声干扰'
        total_score = note_total_score + onset_total_score
        print("total_score is {}".format(total_score))
        print("note_total_score is {}".format(note_total_score))
        print("detail is {}".format(note_detail))
        print("onset_total_score is {}".format(onset_total_score))
        print("detail is {}".format(onset_detail))
        return total_score, note_total_score, note_detail,onset_total_score, onset_detail

    pitch = get_pitch_by_parselmouth(filename)
    all_first_candidate_names, _,_ = get_all_numbered_notation_and_offset(pitch,onset_frames.copy())
    threshold_score = 60
    test_notations = all_first_candidate_names
    standard_notations = pitch_code
    note_total_score, note_detail = calculate_note_score(standard_notations, test_notations, threshold_score)
    note_match_positions = get_match_positions_from_detail_str(note_detail)
    # print("note_match_positions is {}".format(note_match_positions))

    start = onset_frames[0]
    pitch_values = pitch.selected_array['frequency']
    start_end = [i for i, p in enumerate(pitch_values) if p != 0.0]
    end = start_end[-1]
    onset_frames = [f for i, f in enumerate(onset_frames) if i < len(test_notations) and test_notations[i] is not None]
    best_numbered_notations, d = get_best_candidate_names_by_moved(standard_notations, test_notations)
    threshold_score = 40
    rhythm_code = parse_rhythm_code(rhythm_code)
    rhythm_code = [int(x) for x in rhythm_code]
    onset_total_score, onset_detail = calculate_onset_score(rhythm_code, onset_frames, standard_notations, best_numbered_notations,start, end, note_match_positions,threshold_score)

    #最后一个节奏：如果音高是对的而节奏是错的，那么节奏也纠正为对的
    if len(note_detail.split('.')) >= 2 and note_detail.split('.')[-2].strip() == "√" and len(onset_detail.split('.')) >= 2 and onset_detail.split('.')[-2].strip() == "×":
        onset_total_score += int(threshold_score/len(rhythm_code))
        onset_detail = onset_detail[:-3] + '√.]'

    total_score = note_total_score + onset_total_score
    e = time.time()
    print("total_score is {}".format(total_score))
    print("note_total_score is {}".format(note_total_score))
    print("detail is {}".format(note_detail))
    print("onset_total_score is {}".format(onset_total_score))
    print("detail is {}".format(onset_detail))
    print("time speed is {}".format(e-s))
    return total_score, note_total_score, note_detail, onset_total_score, onset_detail


def get_score_with_absolute_pitch(filename,rhythm_code,pitch_code):
    total_score, note_total_score, note_detail, onset_total_score, onset_detail = get_score(filename,rhythm_code,pitch_code)
    detail = "旋律" + note_detail + "。节奏" + onset_detail
    #######################################################################
    # 追加绝对音高的评分流程
    #######################################################################
    # 获取绝对音高
    all_starts = predict_onset_frames_from_single_file(filename)
    sr = 44100
    all_first_candidate_names, result, all_second_candidate_names = get_all_absolute_pitchs_for_filename(filename, all_starts, sr)
    # print("all_first_candidate_names is {},size is {}".format(all_first_candidate_names,len(all_first_candidate_names)))
    # print("result is {},size is {}".format(result,len(result)))
    encode_absolute_pitchs = get_encode_pitch_seque(result)
    # print("encode_absolute_pitchs is {},size is {}".format(encode_absolute_pitchs, len(encode_absolute_pitchs)))
    # print("onset_frames is {},size is {}".format(all_starts,len(all_starts)))
    pitch_code_for_absolute_pitch = parse_rhythm_code_for_absolute_pitch(pitch_code)
    # print("pitch_code_for_absolute_pitch is {},size is {}".format(pitch_code_for_absolute_pitch, len(pitch_code_for_absolute_pitch)))
    encode_pitch_code = get_encode_pitch_seque(pitch_code_for_absolute_pitch)
    # print("encode_pitch_code is {},size is {}".format(encode_pitch_code, len(encode_pitch_code)))
    note_score_absolute_pitch, lcseque, str_detail_list, detail_list, raw_positions = get_matched_detail_absolute_pitch(pitch_code_for_absolute_pitch, result)
    note_score_absolute_pitch, str_detail_list = check_from_second_candidate_names(note_score_absolute_pitch,
                                                                                   str_detail_list, detail_list,
                                                                                   all_second_candidate_names,
                                                                                   pitch_code_for_absolute_pitch)
    # print("lcseque is {},size is {}".format(lcseque, len(lcseque)))
    # print("str_detail_list is {}".format(str_detail_list))
    total_score_absolute_pitch = onset_total_score + note_score_absolute_pitch
    # print("绝对音高评分总分 is {}".format(total_score_absolute_pitch))
    str_detail_list = str_detail_list + "。节奏" + onset_detail

    # 如果绝对音高评测得分高于相对评测的结晶，则需要替换相对评测的结果，因为以更严格的评测标准为准
    # if total_score_absolute_pitch > total_score:
    #     total_score, detail = total_score_absolute_pitch, str_detail_list
    return total_score, all_starts, detail, total_score_absolute_pitch, str_detail_list, all_starts