import librosa.display

from base_helper import *
from create_base import *
from pitch_helper import *


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





#y, sr = load_and_trim('F:/项目/花城音乐项目/样式数据/ALL/旋律/1.31MP3/旋律1.100分.wav')
filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律2.1(80).wav'
filename = 'F:/项目/花城音乐项目/样式数据/ALL/旋律/1.31MP3/旋律3.100分.wav'
filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律一（9）（100）.wav'
#filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律一（14）（95）.wav'
#filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/节奏/节2罗（75）.wav'
#filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏一（4）（96）.wav'
filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋2熙(0).wav'
filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋1罗（96）.wav'
#filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋10罗（92）.wav'
#filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋2录音4(72).wav'
#filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋10罗（92）.wav'
#filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律一（13）（98）.wav'
#filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋1熙(90).wav'
#filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋1谭（98）.wav'
#filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋3.3(96).wav'
filename = 'F:/项目/花城音乐项目/样式数据/6.24MP3/旋律/两只老虎20190624-2939.wav'
filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋1王（98）.wav'
filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋9.1(73).wav'
# filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/6.24MP3/旋律/两只老虎20190624-1089.wav', '[500,500,500,500;500,500,500,500;500,500,1000;500,500;1000]', '[1,2,3,1,1,2,3,1,3,4,5,3,4,5]'  # 音准节奏均正确，给分偏低  94      =======================
filename,rhythm_code,pitch_code = 'F:/项目/花城音乐项目/样式数据/6.24MP3/旋律/两只老虎20190624-1328.wav','[500,500,500,500;500,500,500,500;500,500,1000;500,500;1000]','[1,2,3,1,1,2,3,1,3,4,5,3,4,5]'       #音准节奏均正确，给分偏低  95
# filename,rhythm_code,pitch_code = 'F:/项目/花城音乐项目/样式数据/6.24MP3/旋律/两只老虎20190624-1586.wav','[500,500,500,500;500,500,500,500;500,500,1000;500,500;1000]','[1,2,3,1,1,2,3,1,3,4,5,3,4,5]'      #100  95
# filename,rhythm_code,pitch_code = 'F:/项目/花城音乐项目/样式数据/6.24MP3/旋律/两只老虎20190624-2939.wav','[500,500,500,500;500,500,500,500;500,500,1000;500,500;1000]','[1,2,3,1,1,2,3,1,3,4,5,3,4,5]'      #音准节奏均正确，给分偏低 90
# filename,rhythm_code,pitch_code = 'F:/项目/花城音乐项目/样式数据/6.24MP3/旋律/两只老虎20190624-7881.wav','[500,500,500,500;500,500,500,500;500,500,1000;500,500;1000]','[1,2,3,1,1,2,3,1,3,4,5,3,4,5]'          # 音准节奏均正确，给分偏低 90
# filename,rhythm_code,pitch_code = 'F:/项目/花城音乐项目/样式数据/6.24MP3/旋律/两只老虎20190624-8973.wav','[500,500,500,500;500,500,500,500;500,500,1000;500,500;1000]','[1,2,3,1,1,2,3,1,3,4,5,3,4,5]'       #100  ========


# filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/6.24MP3/旋律/小学8题20190624-3898-6.wav', '[1000,500,500;2000;250,250,500,500,500;2000]', '[6,5,3,6,3,5,3,2,1,6-]'  #故意把最后一个音唱错了，节奏全对,扣0.5左右即可 85
# filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/6.24MP3/旋律/小学8题20190624-3898-7.wav', '[2000;250,250,250,250,1000;2000;500,500,1000]', '[6,5,6,3,5,6,3,2,1,6-]'  # 这一条故意唱错了两个音，节奏是对的，这个扣一分即可 72
# filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/6.24MP3/旋律/小学8题20190624-3898-8.wav', '[1000,250,250,250,250;2000;1000,500,500;2000]', '[1,3,5,1+,6,5,1,3,2,1]'  # 这一条节奏不是太稳，但音高基本正确,9.5分是没问题的 93

# filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/7.12MP3/旋律/小学8题20190702-2647-5.wav', '[1000,1000;500,250,250,1000;1000,500,500;2000]', '[3,1,5,5,6,5,1+,6,3,5]'  #  准确 96
# filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/7.12MP3/旋律/小学8题20190702-2647-6.wav', '[1000,500,500;2000;250,250,500,500,500;2000]', '[6,5,3,6,3,5,3,2,1,6]'  #基本可给满分 59
# filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/7.12MP3/旋律/小学8题20190702-2647-7.wav', '[2000;250,250,250,250,1000;2000;500,500,1000]', '[6,5,6,3,5,6,3,2,1,6]'  #第一个节奏应该扣分，最后一个音没唱，应该没分  78
# filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/7.12MP3/旋律/小学8题20190702-2647-8.wav', '[1000,250,250,250,250;2000;1000,500,500;2000]', '[1,3,5,1+,6,5,1,3,2,1]'  #100       =======================


# filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/7.17MP3/旋律/小学8题20190717-2776-5.wav', '[1000,1000;500,250,250,1000;1000,500,500;2000]', '[3,1,5,5,6,5,1+,6,3,5]'  #100  ===========
# filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/7.17MP3/旋律/小学8题20190717-2776-6.wav', '[1000,500,500;2000;250,250,500,500,500;2000]', '[6,5,3,6,3,5,3,2,1,6-]'  #节奏全对，旋律错最后一个  90
# filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/7.17MP3/旋律/小学8题20190717-2776-7.wav', '[2000;250,250,250,250,1000;2000;500,500,1000]', '[6,5,6,3,5,6,3,2,1,6-]'  #节奏全对，旋律错最后一个  90
# filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/7.17MP3/旋律/小学8题20190717-2776-8.wav', '[1000,250,250,250,250;2000;1000,500,500;2000]', '[1,3,5,1+,6,5,1,3,2,1]'  #节奏全对，旋律错最后一个  90

# filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/7.17MP3/旋律/小学8题20190717-5668-5.wav', '[1000,1000;500,250,250,1000;1000,500,500;2000]', '[3,1,5,5,6,5,1+,6,3,5]'  #节奏全对，旋律错最后一个  90
# filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/7.17MP3/旋律/小学8题20190717-5668-6.wav', '[1000,500,500;2000;250,250,500,500,500;2000]', '[6,5,3,6,3,5,3,2,1,6-]'  #节奏全对，旋律错最后一个  90
filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/7.17MP3/旋律/小学8题20190717-5668-7.wav', '[2000;250,250,250,250,1000;2000;500,500,1000]', '[6,5,6,3,5,6,3,2,1,6-]'  #节奏全对，旋律错最后一个  90
# filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/7.17MP3/旋律/小学8题20190717-5668-8.wav', '[1000,250,250,250,250;2000;1000,500,500;2000]', '[1,3,5,1+,6,5,1,3,2,1]'  #100      =======================

# filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/7.17MP3/旋律/小学8题20190717-6249-5.wav', '[1000,1000;500,250,250,1000;1000,500,500;2000]', '[3,1,5,5,6,5,1+,6,3,5]'  #节奏全对，旋律错最后一个  90
# filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/7.17MP3/旋律/小学8题20190717-6249-6.wav', '[1000,500,500;2000;250,250,500,500,500;2000]', '[6,5,3,6,3,5,3,2,1,6-]'  #节奏全对，旋律错最后一个  90
filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/7.17MP3/旋律/小学8题20190717-6249-7.wav', '[2000;250,250,250,250,1000;2000;500,500,1000]', '[6,5,6,3,5,6,3,2,1,6-]'  #节奏全对，旋律错最后一个  90  ================XXXXXXXXXXXXXXXXXXXXXXXXXXXX=========
# filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/7.17MP3/旋律/小学8题20190717-6249-8.wav', '[1000,250,250,250,250;2000;1000,500,500;2000]', '[1,3,5,1+,6,5,1,3,2,1]'  #节奏全对，旋律错最后一个  90

# filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/7.17MP3/旋律/小学8题20190717-6285-5.wav', '[1000,1000;500,250,250,1000;1000,500,500;2000]', '[3,1,5,5,6,5,1+,6,3,5]'  #节奏全对，旋律错最后一个  90
# filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/7.17MP3/旋律/小学8题20190717-6285-6.wav', '[1000,500,500;2000;250,250,500,500,500;2000]', '[6,5,3,6,3,5,3,2,1,6-]'  #62  =======*******============?????????
# filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/7.17MP3/旋律/小学8题20190717-6285-7.wav', '[2000;250,250,250,250,1000;2000;500,500,1000]', '[6,5,6,3,5,6,3,2,1,6]'  #节奏全对，旋律错最后一个  90
# filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/7.17MP3/旋律/小学8题20190717-6285-8.wav', '[1000,250,250,250,250;2000;1000,500,500;2000]', '[1,3,5,1+,6,5,1,3,2,1]'  #节奏全对，旋律错最后一个  90

# filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/7.12MP3/旋律/小学8题20190712-4290-5.wav', '[1000,1000;500,250,250,1000;1000,500,500;2000]', '[3,1,5,5,6,5,1+,6,3,5]'  #节奏全对，旋律错最后一个  90
# filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/7.12MP3/旋律/小学8题20190712-4290-6.wav', '[1000,500,500;2000;250,250,500,500,500;2000]', '[6,5,3,6,3,5,3,2,1,6]'  #节奏全对，旋律错最后一个  90
# filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/7.12MP3/旋律/小学8题20190712-4290-7.wav','[2000;250,250,250,250,1000;2000;500,500,1000]', '[6,5,6,3,5,6,3,2,1,6]'  #节奏全对，旋律错最后一个  90
# filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/7.12MP3/旋律/小学8题20190712-4290-8.wav', '[1000,250,250,250,250;2000;1000,500,500;2000]', '[1,3,5,1+,6,5,1,3,2,1]'  #100   ???????????==================

# filename, rhythm_code, pitch_code  = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋3罗（80）.wav', '[1000,1000;500,500,1000;500,250,250,500,500;2000]', '[5,5,3,2,1,2,2,3,2,6-,5-]'
# filename, rhythm_code, pitch_code  = 'F:/项目/花城音乐项目/样式数据/6.18MP3/旋律/01，98.wav', '[500,250,250,500,500;250,250,250,250,500,500;500,250,250,500,500;500,250,250,1000]', '[5,5,6,5,3,4,5,4,5,4,2,3,3,4,3,1,2,3,5,1]'

# filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/7.17MP3/旋律/test.wav','[1000,250,250,250,250;2000;1000,500,500;2000]', '[1,3,5,1+,6,5,1,3,2,1]'  #节奏全对，旋律错最后一个  90
# filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋4.4(0).wav'
# filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋2.3(95).wav'
# filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋2.4(50).wav'
# filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋2录音1(90).wav'
# type_index = get_onsets_index_by_filename_rhythm(filename)
# rhythm_code = get_code(type_index, 2)
# pitch_code = get_code(type_index, 3)

# filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/8.05MP3/旋律/小学8题20190805-9112-5.wav', '[1000,1000;500,250,250,1000;1000,500,500;2000]', '[3,1,5,5,6,5,1+,6,3,5]'  #100
# filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/8.05MP3/旋律/小学8题20190805-9112-6.wav', '[1000,500,500;2000;250,250,500,500,500;2000]', '[6,5,3,6,3,5,3,2,1,6]'  #84 =============明天测试一下===========================
# filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/8.05MP3/旋律/小学8题20190805-9112-7.wav','[2000;250,250,250,250,1000;2000;500,500,1000]', '[6,5,6,3,5,6,3,2,1,6]'  #90
# filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/8.05MP3/旋律/小学8题20190805-9112-8.wav', '[1000,250,250,250,250;2000;1000,500,500;2000]', '[1,3,5,1+,6,5,1,3,2,1]'  #100

filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/8.05MP3/旋律/中学8题20190805-6858-5.wav', '[500,250,250,500,500;1500,500;1000,1000;2000]', '[3,3,1,3,4,5,5,6,7,1+]'  #92
# filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/8.05MP3/旋律/中学8题20190805-6858-6.wav', '[1000,1000;1500,500;500,250,250,500,500;2000]', '[1+,7,6,5,4,5,4,3,2,1]'  #100
# filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/8.05MP3/旋律/中学8题20190805-6858-7.wav','[500,1000,500;2000;500,250,250,500,500;2000]', '[1,3,4,5,6,6,1+,7,6,1+]'  #88
# filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/8.05MP3/旋律/中学8题20190805-6858-8.wav', '[500,1000,500;2000;500,500,500,250,250;2000]', '[1+,7,6,5,6,5,4,3,2,1]'  #94


# filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/8.12MP3/旋律/小学8题20190809-3492-5.wav', '[1000,1000;500,250,250,1000;1000,500,500;2000]', '[3,1,5,5,6,5,1+,6,3,5]'  #======================================================
# filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/8.12MP3/旋律/小学8题20190809-3492-6.wav', '[1000,500,500;2000;250,250,500,500,500;2000]', '[6,5,3,6,3,5,3,2,1,6]'  #
# filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/8.12MP3/旋律/小学8题20190809-3492-7.wav','[2000;250,250,250,250,1000;2000;500,500,1000]', '[6,5,6,3,5,6,3,2,1,6]'  #=====================================????????????????????????????
# filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/8.12MP3/旋律/小学8题20190809-3492-8.wav', '[1000,250,250,250,250;2000;1000,500,500;2000]', '[1,3,5,1+,6,5,1,3,2,1]'  #

# filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/8.12MP3/旋律/小学8题20190812-6117-5.wav', '[1000,1000;500,250,250,1000;1000,500,500;2000]', '[3,1,5,5,6,5,1+,6,3,5]'  #
# filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/8.12MP3/旋律/小学8题20190812-6117-6.wav', '[1000,500,500;2000;250,250,500,500,500;2000]', '[6,5,3,6,3,5,3,2,1,6]'  #===================================================12341234134
# filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/8.12MP3/旋律/小学8题20190812-6117-7.wav','[2000;250,250,250,250,1000;2000;500,500,1000]', '[6,5,6,3,5,6,3,2,1,6]'  #
# filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/8.12MP3/旋律/小学8题20190812-6117-8.wav', '[1000,250,250,250,250;2000;1000,500,500;2000]', '[1,3,5,1+,6,5,1,3,2,1]'  #

# y, sr = load_and_trim(filename)
# y,sr = librosa.load(filename)
# rms = librosa.feature.rmse(y=y)[0]
# rms = [x / np.std(rms) for x in rms]
# time = librosa.get_duration(filename=filename)
# # print("time is {}".format(time))
# CQT = librosa.amplitude_to_db(librosa.cqt(y, sr=16000), ref = np.max)
# cqt_bak = CQT.copy()
# w,h = CQT.shape
# # print("w.h is {},{}".format(w,h))
# #onsets_frames = get_real_onsets_frames_rhythm(y)
#
# CQT = np.where(CQT > -35, np.max(CQT), np.min(CQT))
#
# CQT = signal.medfilt(CQT, (5, 5))  # 二维中值滤波
# plt.subplot(5,1,1)
# librosa.display.specshow(cqt_bak ,x_axis='time')
#
#
# plt.subplot(5,1,2)
# base_pitch = get_base_pitch_from_cqt(CQT)
# # base_pitch = filter_cqt(cqt_bak)
# base_pitch_bak = base_pitch.copy()
#
# # base_pitch = signal.medfilt(base_pitch, 11)  # 二维中值滤波
# t = librosa.frames_to_time(np.arange(len(base_pitch)))
# plt.plot(t, base_pitch)
# plt.xlim(0, np.max(t))
# plt.ylim(0, 84)
# first_type = pitch_code[1]
# all_note_types,all_note_type_position = get_all_note_type(base_pitch,first_type)
# all_note_type_position = check_note_type_position(CQT,base_pitch,all_note_type_position)
# print("all_note_types is {}".format(all_note_types))
# print("1 all_note_type_position is {} ,size {}".format(all_note_type_position,len(all_note_type_position)))
# all_note_type_position_time = librosa.frames_to_time(all_note_type_position)
# plt.vlines(all_note_type_position_time, 0, 84, color='r', linestyle='dashed')
# change_points = get_change_point_on_pitch(CQT,first_type)
# print("change_points is {}, size {}".format(change_points,len(change_points)))
# change_points_time = librosa.frames_to_time(change_points)
# plt.vlines(change_points_time, 0, 40, color='b', linestyle='dashed')
# start,end,length = get_start_and_end(CQT)
# start_time = librosa.frames_to_time(start)
# end_time = librosa.frames_to_time(end)
# plt.vlines(start_time, 0, 40, color='black', linestyle='solid')
# plt.vlines(end_time, 0, 40, color='black', linestyle='solid')
#
#
#
#
# plt.subplot(5, 1, 3)
# y, sr = librosa.load(filename)
# rms = librosa.feature.rmse(y=y)[0]
# rms_bak = rms.copy();
# rms = [x / np.std(rms) for x in rms]
# rms = list(np.diff(rms))
# rms.insert(0, 0)
#
# b, a = signal.butter(8, 0.5, analog=False)
# sig_ff = signal.filtfilt(b, a, rms)
#
# # Savitzky-Golay filter 平滑
# # from scipy.signal import savgol_filter
# # sig_ff = savgol_filter(rms, 5, 1)  # window size 51, polynomial order 3
# # sig_ff = signal.medfilt(rms, 5)  # 二维中值滤波
# sig_ff = [x / np.std(sig_ff) for x in sig_ff]
# # sig_ff = [x if x > 0 else x - np.min(sig_ff) for x in sig_ff]
# # rms = signal.medfilt(rms,3)
# times = librosa.frames_to_time(np.arange(len(rms)))
# plt.plot(times, sig_ff)
# plt.plot(times, rms)
# # threshold = 0.4
# # starts_from_rms_must = get_starts_from_rms_by_threshold(filename,threshold)
# # starts_on_highest_point = get_starts_on_highest_point_of_cqt(CQT)  #最高点变化曲线的波谷点
# # select_starts_from_rms_must = []
# # for x in starts_from_rms_must: # 去掉没有音高跳跃点的伪节拍
# #     offset = [np.abs(x - a) for a in all_note_type_position if a < x + 6]
# #     offset2 = [np.abs(x - a) for a in starts_on_highest_point if a < x + 2]  # 与最高点变化曲线波谷点的距离
# #     if ((len(offset) > 0 and np.min(offset) < 12) or (len(offset2) > 0 and np.min(offset2) < 16)) and np.max(base_pitch[x+2:x+3]) != 0 and (np.max(base_pitch[x:x + 5]) != 0 or np.min(base_pitch[x-5:x]) == 0):
# #         select_starts_from_rms_must.append(x)
# # starts_from_rms_must = select_starts_from_rms_must
# # if starts_from_rms_must[0] - start > 15: #如果没包括开始点，则需要添加开始点
# #     starts_from_rms_must.append(start)
# # starts_from_rms_must.sort()
# # starts_from_rms_must = get_must_starts_on_rms(filename, 1.5)
# starts_from_rms_must = get_must_starts(filename,1.8)
# starts_from_rms_must_time = librosa.frames_to_time(starts_from_rms_must)
# plt.vlines(starts_from_rms_must_time, 0, np.max(sig_ff), color='b', linestyle='dashed')
# threshold = 0.15
# starts_from_rms_maybe = get_starts_from_rms_by_threshold(filename,threshold)
# select_starts_from_rms_maybe = []
# for x in starts_from_rms_maybe: # 去掉音高为0的伪节拍
#     offset = [np.abs(x - a) for a in all_note_type_position if a < x + 2]
#     if np.max(base_pitch[x:x + 5]) != 0:
#         select_starts_from_rms_maybe.append(x)
# starts_from_rms_maybe = select_starts_from_rms_maybe
# starts_from_rms_maybe = [x for x in starts_from_rms_maybe if x not in starts_from_rms_must]
# starts_from_rms_maybe_time = librosa.frames_to_time(starts_from_rms_maybe)
# plt.vlines(starts_from_rms_maybe_time, 0, np.max(sig_ff)/2, color='r', linestyle='dashed')
# plt.xlim(0, np.max(t))
# start_250, end_250 = get_range_of_250(filename,rhythm_code)
# start_250_time = librosa.frames_to_time(start_250)
# end_250_time = librosa.frames_to_time(end_250)
# plt.vlines(start_250_time, 0, np.max(sig_ff), color='black', linestyle='solid')
# plt.vlines(end_250_time, 0, np.max(sig_ff), color='black', linestyle='solid')
#
# plt.subplot(5,1,4)
# # c = np.where(cqt_bak > -15, np.max(CQT), np.min(CQT))
# # librosa.display.specshow(c, x_axis='time')
# librosa.display.specshow(CQT, x_axis='time')
# plt.ylim(0, 84)
#
# plt.subplot(5,1,5)
# # b, a = signal.butter(8, 0.6, analog=False)
# # gaps = signal.filtfilt(b, a, gaps)
# # gaps = get_gap_on_cqt(CQT)
# gaps = get_highest_point_on_cqt(CQT)
# b, a = signal.butter(8, 0.25, analog=False)
# gaps = signal.filtfilt(b, a, gaps)
# # from scipy.signal import savgol_filter
# # gaps = savgol_filter(gaps, 53, 1)  # window size 51, polynomial order 3
# # gaps = [x/np.max(gaps) for x in gaps]
# t = librosa.frames_to_time(np.arange(len(gaps)))
# plt.plot(t, gaps)
# select_starts = get_all_starts_by_rms_and_note_change_position(filename,rhythm_code,pitch_code)
# # 最后一个为2000,判断结尾是否需要去掉噪声
# code = parse_rhythm_code(rhythm_code)
# code = [int(x) for x in code]
# if code[-1] == 2000 and end - select_starts[-1] < 10:
#     select_starts = select_starts[:-1]
# select_starts_time = librosa.frames_to_time(select_starts)
# plt.vlines(select_starts_time, 0, np.max(gaps)/2, color='r', linestyle='dashed')
# # starts_on_highest_point = get_starts_on_highest_point_of_cqt(CQT)
# starts_on_highest_point = get_must_starts_on_highest_point_of_cqt(CQT)
# starts_on_highest_point_time = librosa.frames_to_time(starts_on_highest_point)
# plt.vlines(starts_on_highest_point_time, 0, np.max(gaps)/4, color='b', linestyle='dashed')
# plt.xlim(0, np.max(t))
# # plt.ylim(0, 2)
# plt.show()

if __name__ == "__main__":
    get_all_starts_by_steps(filename, rhythm_code)
    detail = draw_detail(filename,rhythm_code, pitch_code)
    detail.show()
