# -*- coding:utf-8 -*-
import numpy as np

class param_handle(object):
    # 初始化
    def __init__(self, kc, kc_time, pitch, pitch_time):
        self.kc = list(kc)
        self.kc_time = kc_time
        self.kc_time_diff = np.diff(kc_time)
        self.pitch = pitch.split(',')
        self.pitch_time = pitch_time
        self.pitch_time_diff = np.diff(pitch_time)

    def get_detail_by_index(self,index):
        if index >= len(self.kc):
            return None
        else:
            kc_on_index = kc[index]     # 该位置上的歌词
            kc_duration = self.kc_time_diff[index]  # 歌词的时长
            kc_detail = {kc_on_index: kc_duration}

            kc_start = self.kc_time[index]
            kc_end = self.kc_time[index + 1]

            less_tmp = [t for t in self.pitch_time if t <= kc_start]
            more_tmp = [t for t in self.pitch_time if t >= kc_end]

            selected_pitchs = self.pitch[len(less_tmp)-1:len(self.pitch) + 1 - len(more_tmp)]
            selected_pitch_time = [t for t in self.pitch_time if t >= less_tmp[-1] and t <= more_tmp[0]]
            selected_pitch_time[0] = kc_start   #开始时间必须是歌词起点
            selected_pitch_time[-1] = kc_end     #结束时间必须是歌词终点
            selected_pitch_time_diff = np.diff(selected_pitch_time)
            pitch_detail = {}
            for i,p in enumerate(selected_pitchs):
                pitch_detail[p] = selected_pitch_time_diff[i]

            result = {"kc_detail": kc_detail,"pitch_detail":pitch_detail}
            return result

    def get_index_in_standard_kc(self,test_char,test_index):
        select_kc_index = [i for i,k in enumerate(self.kc) if k == test_char]
        offset = [np.abs(test_index - s) for s in select_kc_index]
        min_offset = np.min(offset)
        index = offset.index(min_offset)
        index = select_kc_index[index]
        return index

if __name__ == "__main__":
    kc = '喜爱春天的人儿是心地纯洁的人像紫罗兰花儿一样是我知心朋友加'
    # kc = list(kc)
    # kc = ['喜', '爱', '春', '天', '的', '人', '儿', '是', '心', '地', '纯', '洁', '的', '人', '像', '紫', '罗', '兰', '花', '儿', '一', '样', '是', '我', '知', '心', '朋', '友']
    kc_time = [0,1, 2, 3, 3.5, 4, 5, 6, 8, 9, 10, 11, 11.5, 12, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 26.5, 27, 28,30,32]
    pitch = '3,3,2,1,1,7-,6-,6-,6-,4,4,3,2,1,2,4,3,4,4,3,2,2,4,3,3,1,6-,6-,7-,3,2,1,7-,1,6-'
    # pitch = pitch.split(',')
    pitch_time = [0,1, 1.5, 2, 3, 3.5, 4, 5, 6, 8, 9, 9.5, 10, 10.5, 11, 11.5, 12, 16, 17, 17.5, 18, 19, 19.5, 20, 21,
               21.5, 22, 23, 24, 25, 26, 26.5, 27,27.5, 28,32]
    print(len(kc))
    print(len(kc_time))
    print(len(pitch))
    print(len(pitch_time))

    ph = param_handle(kc, kc_time, pitch, pitch_time)
    index = 18
    detail = ph.get_detail_by_index(index)
    print(detail)

    test_kc = '喜春天的人儿是心地纯洁的像紫罗兰花儿一样是我知心朋友加'
    test_char = '是'
    test_index = 20
    index = ph.get_index_in_standard_kc(test_char,test_index)
    print(index)