# -*- coding: UTF-8 -*-
import numpy as np

type_dict = {}
type_dict[4000] = 'A'
type_dict[3500] = 'N'
type_dict[3000] = 'B'
type_dict[2500] = 'M'
type_dict[2000] = 'C'
type_dict[1500] = 'D'
type_dict[1000] = 'E'
type_dict[750] = 'F'
type_dict[500] = 'G'
type_dict[375] = 'H'
type_dict[250] = 'I'
type_dict[125] = 'J'

note_dict = {}
note_dict[7] = 1
note_dict[6] = 2
note_dict[5] = 2
note_dict[4] = 2
note_dict[3] = 1
note_dict[2] = 2
note_dict[1] = 2
# 1-,2-,3-,4-,5-,6-,7-,1,2,3,4,5,6,7,1+,2+,3+,4+,5+,6+,7+
# 2, 2, 1, 2, 2, 2, 1, 2,2,1,2,2,2,1,2, 2, 1, 2, 2, 2, 1
note_gaps = [2,2,1,2,2,2,1,2,2,1,2,2,2,1,2,2,1,2,2,2,1]
note_types = ['c','d','e','f','g','a','b','1','2','3','4','5','6','7','C','D','E','F','G','A','B']

def get_type_symbol(type):
    return type_dict.get(type)

def get_note_symbol(type):
    if len(type) == 2 and type[1] == '+':
        t = type.replace("+", "")
        t = int(t) + 13
    elif len(type) == 2 and type[1] == '-':
        t = type.replace("-", "")
        t = int(t)
    else:
        t = int(type) + 6
    return note_types[t]

def get_symbol_index(symbol):
    for i in range(len(note_types)):
        n = note_types[i]
        if n == symbol:
            return i
    return -1

def get_note_type(pitch,first_pitch,first_type):
    if len(first_type) == 2:
        if first_type[1] == '+':
            start_point = 13 + int(first_type[0])
        elif first_type[1] == '-':
            start_point =  int(first_type[0]) -1
    elif len(first_type) == 1:
        start_point = 6 + int(first_type[0])

    if pitch > first_pitch:
        tmp = note_gaps[start_point:]
        tmp_sum = [np.sum(tmp[:x]) for x in range(1,len(tmp))]
        real_gap = pitch - first_pitch
        offset = [np.abs(x - real_gap) for x in tmp_sum]
        min_index = offset.index(np.min(offset))
        result = start_point + min_index + 1

    elif pitch < first_pitch:
        tmp = note_gaps[:start_point]
        tmp_sum = [np.sum(tmp[x:]) for x in range(0, len(tmp))]
        real_gap = first_pitch - pitch
        offset = [np.abs(x - real_gap) for x in tmp_sum]
        min_index = offset.index(np.min(offset))
        result = start_point - len(tmp) + min_index
    else:
        result = start_point
    return note_types[result]
def get_all_note_type(base_pitch,first_type):
    if len(base_pitch) < 1:
        return []
    all_note_types = []
    all_note_type_position = []
    more_than_zero = [x for x in base_pitch if x > 0]
    if len(more_than_zero) < 1:
        return []
    first_pitch = more_than_zero[0]-1
    pass_zero = [i for i in range(1,len(base_pitch)-1) if base_pitch[i] > 0 and base_pitch[i-1] == 0]
    all_note_type_position = pass_zero.copy()
    all_note_types.append(first_type)
    all_note_type_position.append(list(base_pitch).index(more_than_zero[0]))
    for i in range(1,len(base_pitch)):
        c_pitch = base_pitch[i]
        b_pitch = base_pitch[i-1]
        if c_pitch != b_pitch and c_pitch > 0:
            c_note = get_note_type(c_pitch,first_pitch,first_type)
            if c_note is not None and i not in all_note_type_position:
                all_note_types.append(c_note)
                all_note_type_position.append(i)
            elif i not in all_note_type_position:
                all_note_types.append(first_type)
                all_note_type_position.append(i)
    all_note_type_position.sort()
    return all_note_types,all_note_type_position
if __name__ == '__main__':
    # for key in type_dict:
    #     print(str(key) + ":" + type_dict.get(key))

    pitch, first_pitch, first_type = 12,11,'1'
    note_tpye = get_note_type(pitch,first_pitch,first_type)
    print(note_tpye)

    base_pitch = [14.0, 16.0, 18.0, 14.0, 14.0, 16.0, 19.0, 15.0, 18.0, 19.0, 21.0, 18.0, 19.0, 21.0]
    all_types,all_note_type_position =  get_all_note_type(base_pitch,first_type)
    print("all_types is {}, size {}".format(all_types,len(all_types)))