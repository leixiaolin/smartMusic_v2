import librosa
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
from create_base import *
from myDtw import *
from find_mismatch import *
from filters import *
from vocal_separation import *

filepath = 'F:\项目\花城音乐项目\样式数据\音乐样本2019-01-29\节奏九\\'
# filename = 'F:/项目/花城音乐项目/样式数据/ALL/节奏/节奏八/节奏八（标准音频）.wav'
#filename = 'F:/项目/花城音乐项目/样式数据/ALL/节奏/节奏八/节奏8.100分.wav'
#filename = 'F:/项目/花城音乐项目/样式数据/ALL/旋律/1.31MP3/旋律1.100分.wav'
#filename = 'F:/项目/花城音乐项目/样式数据/ALL/旋律/1.31MP3/旋律2.100分.wav'
#filename = 'F:/项目/花城音乐项目/样式数据/ALL/节奏/节奏八/节奏八（1）(90).wav'
#filename = 'F:/项目/花城音乐项目/样式数据/ALL/节奏/节奏八/节奏八（2）（90分）.wav'
filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏4卢(65).wav'
filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏2-01（80）.wav'
filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏4-02（68）.wav'
filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/节奏/节奏二（4）（100）.wav'
filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏7-02（30）.wav'
filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏1（三）(95).wav'
#filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/视唱1-02（90）.wav'
#filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律2（四）(96).wav'
#filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律1.1(95).wav'
#filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律2.1(80).wav'
#filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律2.3(55).wav'
#filename = 'F:/项目/花城音乐项目/样式数据/3.19MP3/节奏/节奏六1(10).wav'

#filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律二（10）（75）.wav'
# filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律二（8）（100）.wav'
# filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律7_40218（20）.wav'
# filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律一（9）（100）.wav'
# filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律一（14）（95）.wav'
#filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/节奏/节1罗（90）.wav'

# 2. Load the audio as a waveform `y`
#    Store the sampling rate as `sr`

codes = np.array(['[1000,1000;2000;1000,500,500;2000]',
                  '[2000;1000,1000;500,500,1000;2000]',
                  '[1000,1000;500,500,1000;1000,1000;2000]',
                  '[1000,--(1000);1000,--(1000);500,250,250,1000;--(1000),1000]',
                  '[500;1000,500,1000,500;500,500,500,250,250,500,500;250,250,500,500,1000]',
                  '[1000,--(1000);1000,--(1000);1000,-(500),500;1000,1000]',
                  '[750,250,500,500,500,-(500);500,1000,500,500,-(500);750,250,500,500,500,-(500)]',
                  '[500,1000,500,500,250,250;1000,500,750,250,500;3000]',
                  '[500,500,500;1000,500;500,500,500;1500;500,500,500;1000,500;500;1000;1500]',
                  '[500,500,1000;500,500;1000;375,125,250,250,375,125,250,250;500,500,1000]',
                  '[500,500,1000;500,500,1000;500,500,750,250;2000]',
                  '[1000,1000;500,500,1000;1000,500,500;2000]',
                  '[1000,1000;500,500,1000;500,250,250,250;2000]',
                  '[500,1000,500;250,250,250,250,500,500;500,500,500,500;2000]'])
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

def get_max_strength(chromagram):
    c_max = np.argmax(chromagram, axis=0)
    #print(c_max.shape[0])
    #print(c_max)
   # print(np.diff(c_max))
    # chromagram_diff = np.diff(chromagram,axis=0)
    # print(chromagram_diff)
    # sum_chromagram_diff = chromagram_diff.sum(axis=0)
    # test = np.array(sum_chromagram_diff)
    # plt.plot(test)

    img = np.zeros(chromagram.shape, dtype=np.float32)
    w, h = chromagram.shape
    for x in range(h):
        # img.item(x, c_max[x], 0)
        img.itemset((c_max[x], x), 1)
    return img

#y, sr = load_and_trim('F:/项目/花城音乐项目/样式数据/ALL/旋律/1.31MP3/旋律1.100分.wav')
y, sr = load_and_trim(filename)
# silence_threshold = 0.2
# need_vocal_separation = check_need_vocal_separation(y, silence_threshold)
# if need_vocal_separation:
#     y, sr = get_foreground(y, sr)  # 分离前景音

CQT = librosa.amplitude_to_db(librosa.cqt(y, sr=16000), ref=np.max)
w, h = CQT.shape
CQT[50:w, :] = -100
CQT[0:20, :] = -100

# 标准节拍时间点
type_index = get_onsets_index_by_filename(filename)
total_frames_number = get_total_frames_number(filename)
# base_frames = onsets_base_frames_rhythm(type_index,total_frames_number)
base_frames = onsets_base_frames(codes[type_index], total_frames_number)
base_onsets = librosa.frames_to_time(base_frames, sr=sr)

first_frame = base_frames[1] - base_frames[0]
rms = librosa.feature.rmse(y=y)[0]
rms = [x / np.std(rms) for x in rms]
min_waterline = find_min_waterline(rms, 8)
first_frame_rms = rms[0:first_frame]
first_frame_rms_max = np.max(first_frame_rms)
waterline = 0
if len(min_waterline) > 0:
    waterline = min_waterline[0][1]

if first_frame_rms_max == np.max(rms):
    #print("=====================================")
    threshold = first_frame_rms_max * 0.35
    if threshold > waterline:
        threshold = waterline + 0.2
        threshold = np.float64(threshold)
    rms = rms_smooth(rms, threshold, 6)
    # rms = [x if x > first_frame_rms_max * 0.35 else 0 for x in rms]
else:
    threshold = first_frame_rms_max * 0.6
    if threshold > waterline:
        threshold = waterline + 0.2
        threshold = np.float64(threshold)
    rms = rms_smooth(rms, threshold, 6)
    # rms = [x if x > first_frame_rms_max * 0.6 else 0 for x in rms]
# rms = [x / np.std(rms) if x / np.std(rms) > first_frame_rms_max*0.8 else 0 for x in rms]
# rms = rms/ np.std(rms)
rms_diff = np.diff(rms)
# print("rms_diff is {}".format(rms_diff))
#print("rms max is {}".format(np.max(rms)))
# all_peak_points = get_all_onsets_starts(rms,0.7)
# all_peak_points = get_onsets_by_cqt_rms(y,16000,base_frames,0.7)
topN = len(base_frames)
best_starts_waterline = []
threshold = first_frame_rms_max * 0.8
if len(min_waterline) > 0:
    # waterline = min_waterline[0][1]
    # waterline *= 1.5
    waterline,best_starts_waterline = find_best_waterline(rms, 4, topN)
    waterline += 0.3
    if waterline < 0.6:
        waterline = 0.6

    # waterline = 0.8
    if threshold < waterline:
        # waterline +=0.0000000001
        threshold = waterline + 0.5
        threshold = np.float64(threshold)
        # pass
    #print("waterline is {}".format(waterline))
    #print("threshold is {}".format(threshold))
all_peak_points, rms, threshold = get_topN_peak_by_denoise(rms, threshold, topN, waterline)
# all_peak_points,_ = get_topN_peak_by_denoise(rms, first_frame_rms_max * 0.8, topN)
# onsets_frames = get_real_onsets_frames_rhythm(y)
# _, onsets_frames = get_onset_rmse_viterbi(y, 0.35)
# onsets_frames = get_all_onsets_starts_for_beat(rms, 0.6)
onsets_frames = []

# all_peak_points = get_all_onsets_starts_for_beat(rms,0.6)
# all_trough_points = get_all_onsets_ends(rms,-0.4)
want_all_points = np.hstack((all_peak_points, onsets_frames))
want_all_points = list(set(want_all_points))
want_all_points.sort()
want_all_points_diff = np.diff(want_all_points)
if len(want_all_points) > 0:
    # 去掉挤在一起的线
    result = [want_all_points[0]]
    for i, v in enumerate(want_all_points_diff):
        if v > 4:
            result.append(want_all_points[i + 1])
        else:
            pass
want_all_points = result
#want_all_points = [x for i,x in enumerate(all_points) if i < len(all_points)-1 and (peak_trough_rms_diff[i]>1)]
if topN - len(want_all_points) >= 3 and topN - len(best_starts_waterline) <3:
    want_all_points = best_starts_waterline
print("want_all_points is {}".format(want_all_points))
want_all_points_time = librosa.frames_to_time(want_all_points)

#librosa.display.specshow(CQT)
plt.figure(figsize=(10, 6))
plt.subplot(5,1,1) #要生成两行两列，这是第一个图plt.subplot('行','列','编号')
# plt.colorbar(format='%+2.0f dB')
# plt.title('Constant-Q power spectrogram (note)')
librosa.display.specshow(CQT, y_axis='cqt_note',x_axis='time')
#onsets_frames =  librosa.onset.onset_detect(y)
#onsets_frames = get_real_onsets_frames_rhythm(y)
#onsets_frames = get_onsets_frames_by_cqt_for_rhythm(y,sr)

print(np.max(y))
#onstm = librosa.frames_to_time(onsets_frames, sr=sr)
plt.vlines(want_all_points_time, 0,sr, color='y', linestyle='solid')
print(CQT.shape)
q1,q2 = CQT.shape
#print(plt.figure)

plt.subplot(5,1,2) #要生成两行两列，这是第一个图plt.subplot('行','列','编号')
librosa.display.waveplot(y, sr=sr)
plt.vlines(want_all_points_time, -1*np.max(y),np.max(y), color='y', linestyle='solid')
    #plt.axhline(y=0.2, color='b')

# duration = librosa.get_duration(filename=filename)
# # 标准节拍时间点
# base_onsets = onsets_base(codes[11], duration, onstm[0])
# plt.vlines(base_onsets[:-1], -1*np.max(y),np.max(y), color='r', linestyle='dashed')
# plt.vlines(base_onsets[-1], -1*np.max(y),np.max(y), color='white', linestyle='dashed')
plt.subplot(5,1,3)
rms = librosa.feature.rmse(y=y)[0]
rms = [x / np.std(rms) for x in rms]
times = librosa.frames_to_time(np.arange(len(rms)))
plt.plot(times, rms)
#plt.axhline(0.02, color='r', alpha=0.5)
plt.xlabel('Time')
plt.ylabel('RMS')
plt.axis('tight')
plt.xlim(0,np.max(times))
plt.vlines(want_all_points_time, 0,np.max(rms), color='y', linestyle='solid')
if len(min_waterline)>0:
    xmin = librosa.frames_to_time([min_waterline[0][0]])
    xmin = float('%.2f' % xmin)
    xmax = librosa.frames_to_time([min_waterline[0][0]+8])
    xmax = float('%.2f' % xmax)
    print("min_waterline is {} at {}-{}".format(min_waterline[0][1],xmin,xmax))
    plt.axhline(y=min_waterline[0][1],color='r')
    plt.axhline(y=threshold,color='b')
# 标准节拍时间点
#base_frames = onsets_base_frames_rhythm(type_index,total_frames_number)
print("base_frames is {}".format(base_frames))

#min_d, best_y, onsets_frames = get_dtw_min(onsets_frames, base_frames, 65,move=False)
if len(want_all_points)>0:
    if base_frames[0] < want_all_points[0]:
        best_y = [x + (want_all_points[0] - base_frames[0]) for x in base_frames]
    else:
        best_y = base_frames
    base_onsets = librosa.frames_to_time(best_y, sr=sr)
    plt.vlines(base_onsets,  0,np.max(rms), color='r', linestyle='dashed')
    # 找出漏唱的线的帧
    standard_y = want_all_points.copy()
    recognize_y = best_y.copy()
    miss_onsets = get_mismatch_line(standard_y,recognize_y)
    miss_onsets_time = librosa.frames_to_time(miss_onsets[1], sr=sr)
    plt.vlines(miss_onsets_time,  0,np.max(rms), color='black', linestyle='dashed')


plt.subplot(5,1,4)
chromagram = librosa.feature.chroma_cqt(y, sr=sr)

c_max = np.argmax(chromagram,axis=0)
#print("c_max is {}".format(c_max))
c_max_diff = np.diff(c_max) # 一阶差分
img = np.zeros(chromagram.shape,dtype=np.float32)
w,h = chromagram.shape
for x in range(len(c_max_diff)):
    #img.item(x, c_max[x], 0)
    if x > 0 and (c_max_diff[x] == 1 or c_max_diff[x] == -1):
        c_max[x] = c_max[x-1]

for x in range(h):
    #img.item(x, c_max[x], 0)
    img.itemset((c_max[x],x), 1)
    img.itemset((c_max[x],x), 1)
    img.itemset((c_max[x],x), 1)
# 最强音色图
img  = get_max_strength(CQT)
librosa.display.specshow(img, x_axis='time', y_axis='chroma', cmap='coolwarm')
#plt.vlines(base_onsets, 0, sr, color='y', linestyle='solid')
plt.vlines(want_all_points_time, 0,sr, color='y', linestyle='solid')


plt.subplot(5,1,5)
c_max = np.argmax(CQT, axis=0)
#c_max = deburring(c_max, 3)
# note_start,note_end,note_number = find_note_number(c_max,all_peak_points[0],all_peak_points[1])
# note_start_time = librosa.frames_to_time([note_start])
#print("note_number is {}".format(note_number))
c_max_diff = np.diff(c_max)
times = range(len(c_max))
plt.plot(times,c_max)
plt.xlim(0,np.max(times))
onsets_frames = get_onsets_by_cqt_rms(y,16000,base_frames,0.7)
want_all_points_time = librosa.frames_to_time(onsets_frames)
#plt.vlines(want_all_points_time, np.min(c_max),np.max(c_max), color='y', linestyle='solid')
#plt.text(note_start_time, note_number, note_number)
for i in range(len(onsets_frames)-1):
    note_start,note_end,note_number = find_note_number_by_range(c_max,onsets_frames[i],onsets_frames[i+1])
    note_start_time = librosa.frames_to_time([note_start])
    #plt.text(note_start_time, note_number-4, note_number)
    if i ==0:
        first_note_number = note_number
#print(all_peak_points[-1])
note_start,note_end,note_number = find_note_number_by_range(c_max,onsets_frames[-1],len(c_max)-1)
note_start_time = librosa.frames_to_time([note_start])
find_note = find_note_number(note_number,3)
note_number_gap = first_note_number - find_note[0]
#plt.text(note_start_time, note_number-4, note_number)
# find_note = find_note_number(note_number,2)
# find_note = [x + note_number_gap for x in find_note]
# print("find_note is {}".format(find_note))
# print("c_max is {}".format(c_max))
# index = np.where(c_max == find_note[0])
# note_start_time = librosa.frames_to_time([index])
# plt.vlines(note_start_time, np.min(c_max),np.max(c_max), color='black', linestyle='solid')
# if len(find_note)>1:
#     index = np.where(c_max == find_note[1])
#     note_start_time = librosa.frames_to_time([index])
#     plt.vlines(note_start_time, np.min(c_max), np.max(c_max), color='black', linestyle='solid')

# plt.savefig('f:/a.jpg', bbox_inches='tight', pad_inches=0)
plt.show()
