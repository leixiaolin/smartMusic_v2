import parselmouth
from melody_util.parselmouth_sc_util import *
import matplotlib.pyplot as plt
import scipy.signal as signal
from melody_util.pingfen_sc_util import calculate_note_score,parse_pitch_code,parse_rhythm_code,get_standard_frames,get_score_with_absolute_pitch,get_score
from melody_util.cnn_bilstm_attention_model_for_onset_predict import predict_onset_frames_from_single_file
from melody_util.test_util import *
import os
import gc


filename = 'F:/项目/花城音乐项目/音符起始点检测/ourself/wav/4.wav'
filename = 'F:/项目/花城音乐项目/样式数据/20.03.16MP3/wav/c6e90530-1ea0-40d4-8994-0a58e6c8258b.wav'
filename = 'F:/项目/花城音乐项目/样式数据/20.03.16MP3/wav/42d3b2f6-4fd9-4a72-a242-a2da0053e61d.wav'
filename = 'F:/项目/花城音乐项目/样式数据/20.03.16MP3/wav/4c1590c4-8a8d-4920-80dd-e4fc9e7a1d95.wav'
filename = 'F:/项目/花城音乐项目/样式数据/1.31MP3/wav/旋律/done/旋律9.30分.wav'
filename = 'F:/项目/花城音乐项目/样式数据/9.08MP3/旋律/zx1.wav'
filename,rhythm_code,pitch_code = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/视唱1-02（90）.wav','[500,500,1000;500,500,1000;500,500,750,250;2000]','[3,3,3,3,3,3,3,5,1,2,3]'
# filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/8.12MP3/旋律/小学8题20190809-3492-5.wav', '[1000,1000;500,250,250,1000;1000,500,500;2000]', '[3,1,5,5,6,5,1+,6,3,5]'
# filename = 'F:/项目/花城音乐项目/样式数据/8.12MP3/旋律/小学8题20190809-3492-6.wav'
# filename = 'F:/项目/花城音乐项目/音符起始点检测/ourself/wav/z1_aug1-1.wav'
# filename = 'F:/项目/花城音乐项目/音符起始点检测/ourself/wav/x3_aug2--1.5.wav'
# filename = 'F:/项目/花城音乐项目/样式数据/1.31MP3/wav/旋律/旋律1.40分.wav'
# filename = 'F:/项目/花城音乐项目/样式数据/1.31MP3/wav/节奏/节奏1.20分.wav'
# filename = 'F:/项目/花城音乐项目/样式数据/1.31MP3/wav/旋律/done/旋律3_40分.wav'
# filename = 'F:/项目/花城音乐项目/样式数据/6.18MP3/旋律/旋律3，55.wav'
# filename = 'F:/项目/花城音乐项目/样式数据/6.18MP3/旋律/五年级1，98.wav'
# filename = 'F:/项目/花城音乐项目/样式数据/1.31MP3/wav/旋律/done/旋律1_90分.wav'
filename = 'F:/项目/花城音乐项目/样式数据/1.31MP3/wav/旋律/done/旋律1_92分.wav'
filename = 'F:/项目/花城音乐项目/样式数据/1.31MP3/wav/旋律/done/旋律10_90分.wav'
filename = 'F:/项目/花城音乐项目/样式数据/1.31MP3/wav/旋律/done/旋律1_75分.wav'
filename = 'F:/项目/花城音乐项目/样式数据/1.31MP3/wav/旋律/done/旋律9.100分.wav'

type_index = get_onsets_index_by_filename_rhythm(filename)
rhythm_code = get_code(type_index, 2)
pitch_code = get_code(type_index, 3)

snd = parselmouth.Sound(filename)
# intensity = snd.to_intensity()
# plt.figure()
# plt.plot(snd.xs(), snd.values.T)
# plt.xlim([snd.xmin, snd.xmax])
# plt.xlabel("time [s]")
# plt.ylabel("amplitude")
# plt.show()

rhythm_code = parse_rhythm_code(rhythm_code)
rhythm_code = [int(x) for x in rhythm_code]
pitch_code = parse_pitch_code(pitch_code)

pitch = get_pitch_by_parselmouth(filename)
print(pitch)
pitch_values = pitch.selected_array['frequency']
start_end = [i for i,p in enumerate(pitch_values) if p != 0.0]
print('start is {}, end is {}'.format(start_end[0],start_end[-1]))
total_rhythm = 8000
print('min is {}'.format(int((start_end[-1] - start_end[0])/total_rhythm*250)))


onset_frames = [89, 110, 132, 174, 250, 266, 303, 351, 386, 442, 479, 520, 546, 582, 609, 694]
onset_frames = [89, 110, 132, 174, 250, 266, 303, 351, 386, 442, 479, 520, 546, 582, 609, 694]
onset_frames = predict_onset_frames_from_single_file(filename)
# onset_frames = [55, 72, 97, 108, 134, 155, 187, 212, 221, 234, 282, 302, 313, 325]
onset_frames_time = [f*10e-3 for f in onset_frames]
standard_frames = get_standard_frames(rhythm_code,onset_frames[0],start_end[-1])
standard_frames_time = [f*10e-3 for f in standard_frames]
all_first_candidate_names, all_first_candidates, all_offset_types = get_all_numbered_notation_and_offset(pitch, onset_frames.copy())
all_first_candidates = [a for a in all_first_candidates if a is not None]
print("all_first_candidate_names is {},size is {}".format(all_first_candidate_names,len(all_first_candidate_names)))
print("all_first_candidates is {},size is {}".format(all_first_candidates,len(all_first_candidates)))
print("all_offset_types is {},size is {}".format(all_offset_types,len(all_offset_types)))
indexs = get_all_indexs_for_notation_names(all_first_candidate_names)
print("indexs is {}.size is {}".format(indexs,len(indexs)))
all_notation_names = get_all_notation_names_for_indexs(indexs)
print("all_first_candidate_names is {},size is {}".format(all_notation_names,len(all_notation_names)))
intensity = snd.to_intensity()

lengths,starts, lens = get_pitch_length_for_each_onset(pitch,onset_frames)
print("lengths is {},size is {}".format(lengths,len(lengths)))
tmp = del_onset_frames_for_too_short(pitch,onset_frames)
print("tmp is {},size is {}".format(tmp,len(tmp)))
# spectrogram = snd.to_spectrogram()
# plt.figure()
# draw_spectrogram(spectrogram)
# plt.twinx()
# draw_intensity(intensity)
# plt.xlim([snd.xmin, snd.xmax])
# plt.show()
# pic = draw_pitch_specified(intensity, pitch, pitch_values, 2,grain_size=1)
# pic.vlines(onset_frames_time,0,np.max(all_first_candidates))
# pic.vlines(standard_frames_time,0,50,colors='red')
# pic.show()

standard_notations = '[3,3,3,3,3,3,3,5,1,2,3]'
test_notations = ['C4', 'C3', 'C3', 'C4', 'C4', 'C4', 'C4', 'C4', None, 'C4', 'D4', 'G#3', 'G#3', 'A#3', 'C4', 'C4']
test_notations = all_first_candidate_names

best_numbered_notations,d = get_best_candidate_names_by_moved(standard_notations, test_notations)
print("best_numbered_notations is {},size is {}".format(best_numbered_notations,len(best_numbered_notations)))
print(np.array(d).shape)
standard_positions,test_positions = get_matched_positions(d)
print("standard_positions is {},size is {}".format(standard_positions,len(standard_positions)))
print("test_positions is {},size is {}".format(test_positions,len(test_positions)))

# threshold_score = 60
# total_score,detail = calculate_note_score(standard_notations, test_notations, threshold_score)
# print("total_score is {}".format(total_score))
# print("detail is {}".format(detail))
#
# start = onset_frames[0]
# end = start_end[-1]
# onset_frames = [f for i,f in enumerate(onset_frames) if test_notations[i] is not None]
# threshold_score = 40
# total_score,detail = calculate_onset_score(rhythm_code,onset_frames,standard_notations,best_numbered_notations,start,end,threshold_score)
# print("total_score is {}".format(total_score))
# print("detail is {}".format(detail))

print("===================================================")
filename,rhythm_code,pitch_code = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/视唱1-02（90）.wav','[500,500,1000;500,500,1000;500,500,750,250;2000]','[3,3,3,3,3,3,3,5,1,2,3]'
filename,rhythm_code,pitch_code = 'F:/项目/花城音乐项目/样式数据/1.31MP3/wav/旋律/done/旋律1.wav','[500,500,1000;500,500,1000;500,500,750,250;2000]','[3,3,3,3,3,3,3,5,1,2,3]'
filename,rhythm_code,pitch_code = 'F:/项目/花城音乐项目/样式数据/1.31MP3/wav/旋律/done/旋律1_75分.wav','[500,500,1000;500,500,1000;500,500,750,250;2000]','[3,3,3,3,3,3,3,5,1,2,3]'
filename,rhythm_code,pitch_code = 'F:/项目/花城音乐项目/样式数据/1.31MP3/wav/旋律/done/旋律10_90分.wav','[500,500,1000;500,500,1000;500,500,750,250;2000]','[3,3,3,3,3,3,3,5,1,2,3]'
filename,rhythm_code,pitch_code = 'F:/项目/花城音乐项目/样式数据/7.17MP3/旋律/test.wav','[500,500,1000;500,500,1000;500,500,750,250;2000]','[3,3,3,3,3,3,3,5,1,2,3]'
filename,rhythm_code,pitch_code = 'F:/项目/花城音乐项目/样式数据/8.28MP3/旋律/4.wav','[500,500,1000;500,500,1000;500,500,750,250;2000]','[3,3,3,3,3,3,3,5,1,2,3]'
filename,rhythm_code,pitch_code = 'F:/项目/花城音乐项目/样式数据/8.28MP3/旋律/8.wav','[500,500,1000;500,500,1000;500,500,750,250;2000]','[3,3,3,3,3,3,3,5,1,2,3]'
filename,rhythm_code,pitch_code = 'F:/项目/花城音乐项目/样式数据/8.28MP3/旋律/10.wav','[500,500,1000;500,500,1000;500,500,750,250;2000]','[3,3,3,3,3,3,3,5,1,2,3]'
filename,rhythm_code,pitch_code = 'F:/项目/花城音乐项目/样式数据/8.28MP3/旋律/1.wav','[500,500,1000;500,500,1000;500,500,750,250;2000]','[3,3,3,3,3,3,3,5,1,2,3]'
filename,rhythm_code,pitch_code = 'F:/项目/花城音乐项目/样式数据/1.31MP3/wav/旋律/done/旋律9.100分.wav','[500,500,1000;500,500,1000;500,500,750,250;2000]','[3,3,3,3,3,3,3,5,1,2,3]'

type_index = get_onsets_index_by_filename_rhythm(filename)
rhythm_code = get_code(type_index, 2)
pitch_code = get_code(type_index, 3)
# get_score(filename,rhythm_code,pitch_code)
get_score_with_absolute_pitch(filename,rhythm_code,pitch_code)

exit()
dir_list = ['F:/项目/花城音乐项目/样式数据/20.03.16MP3/wav/', 'F:/项目/花城音乐项目/样式数据/20.03.18MP3/wav/']
dir_list = ['F:/项目/花城音乐项目/样式数据/20.03.16MP3/wav/']
dir_list = ['F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/']
dir_list = ['F:/项目/花城音乐项目/样式数据/6.18MP3/旋律/']
# dir_list = ['F:/项目/花城音乐项目/样式数据/8.12MP3/旋律/']
# dir_list = ['F:/项目/花城音乐项目/音符起始点检测/ourself/wav/']
# dir_list = ['F:/项目/花城音乐项目/样式数据/1.31MP3/wav/节奏/']
# dir_list = ['F:/项目/花城音乐项目/样式数据/1.31MP3/wav/旋律/']
dir_list = ['F:/项目/花城音乐项目/样式数据/1.31MP3/wav/旋律/done/']
# dir_list = []
for dir in dir_list:
    file_list = os.listdir(dir)
    # shuffle(file_list)  # 将语音文件随机排列
    # file_list = ['节2罗（75）.wav']
    file_total = len(file_list)
    for filename in file_list:
        if filename.find('wav') >= 0:
            print(filename)
            # pic = get_predict_pic_from_single_file(os.path.join(dir, filename), model)
            type_index = get_onsets_index_by_filename_rhythm(os.path.join(dir, filename))
            # onset_code = get_code(type_index, 1)
            rhythm_code = get_code(type_index, 2)
            pitch_code = get_code(type_index, 3)
            get_score(os.path.join(dir, filename), rhythm_code, pitch_code)
            gc.collect()

# tmp = [71, 127, 150, 163, 179, 222, 271, 423, 474, 571, 668]
# tmp = [56, 276, 342, 394, 449, 599, 653, 756, 834, 923, 934, 953, 1027, 1109, 1289]
# tmp_time = [t*10e-3 for t in tmp]
# pitch = snd.to_pitch()
# pitch_df = get_pitch_diff(pitch)
# pitch_df = signal.medfilt(pitch_df, 3)  # 中值滤波
# pitch_df_indexs = [i for i,p in enumerate(pitch_df) if np.abs(p) > 3]
# pitch_df_indexs_times = [t*10e-3 for t in pitch_df_indexs]
# print(pitch_df)
# plt.vlines(tmp,0,5,colors='r')
# plt.plot(pitch_df)

# pitch_derivative = get_pitch_derivative(pitch,n=1)
# # pitch_derivative = signal.medfilt(pitch_derivative, 5)  # 中值滤波
# pitch_derivative = [p if np.abs(p) >= 3 else 0 for p in pitch_derivative ]
# # print(pitch_derivative)
# plt.plot(pitch_derivative)
# # If desired, pre-emphasize the sound fragment before calculating the spectrogram
# pre_emphasized_snd = snd.copy()
# pre_emphasized_snd.pre_emphasize()
# spectrogram = pre_emphasized_snd.to_spectrogram(window_length=0.01, maximum_frequency=8000)
# plt.figure()
# draw_spectrogram(spectrogram)
# plt.twinx()
# draw_pitch(pitch)
# plt.xlim([snd.xmin, snd.xmax])
# plt.vlines(pitch_df_indexs_times,0,100,colors='r')
# plt.show()