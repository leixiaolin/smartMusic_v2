import parselmouth

import numpy as np
import matplotlib.pyplot as plt
import librosa
from parselmouth_util import get_mean_pitch,draw_spectrogram,draw_pitch,get_pitch_by_parselmouth,draw_intensity
import os

result_path = 'e:/test_image/n/'

def batch_draw_samples(dir_list):
    for dir in dir_list:
        file_list = os.listdir(dir)
        # shuffle(file_list)  # 将语音文件随机排列
        #file_list = ['旋1王（98）.wav']
        file_total = len(file_list)
        total_length = 0
        for filename in file_list:
            if filename.find(".wav") > 0:
                print(dir + filename)
                snd = parselmouth.Sound(dir + filename)
                pitch = get_pitch_by_parselmouth(dir + filename)
                draw_pitch(pitch, draw_type=0,filename=filename)
                plt.xlim([snd.xmin, snd.xmax])
                plt.savefig(result_path  + filename.split('/')[-1].split('.wav')[0] + '.jpg', bbox_inches='tight', pad_inches=0)
                plt.clf()


filename = "F:/项目/花城音乐项目/音符起始点检测/note_onset_detection-master/audio/F-2.wav"
filename = "F:/项目/花城音乐项目/音高检测/Pitch-Class-Detection-using-ML-master/resources/training-set/Gb3.mp3"
filename,notation = "F:/项目/花城音乐项目/音高检测/Pitch-Class-Detection-using-ML-master/resources/training-set/C4.mp3",''
# filename = "F:/项目/花城音乐项目/样式数据/20.03.16MP3/wav/1a981498-eb19-4f0f-9b90-e17296d24613.wav"
# filename = "F:/项目/花城音乐项目/样式数据/20.03.16MP3/wav/2adcc430-0dc8-4c7e-bb24-ec36dd0f82ae.wav"
# filename = "F:/项目/花城音乐项目/样式数据/6.24MP3/旋律/两只老虎20190624-7881.wav" #[1,2,3,1,1,2,3,1,3,4,5,3,4,5]
# filename = "F:/项目/花城音乐项目/样式数据/6.24MP3/旋律/两只老虎20190624-8973.wav"
# filename,notation = "F:/项目/花城音乐项目/样式数据/20.03.16MP3/wav/7bce3e27-5239-4c79-b912-f2870fde5814.wav",''
# filename,notation = "F:/项目/花城音乐项目/样式数据/20.03.16MP3/wav/2f4a4416-dbb7-4584-b9f9-f952372161af.wav",''
# filename,notation = "F:/项目/花城音乐项目/样式数据/20.03.16MP3/wav/2adcc430-0dc8-4c7e-bb24-ec36dd0f82ae.wav",''
# filename,notation = "F:/项目/花城音乐项目/样式数据/20.03.16MP3/wav/1a981498-eb19-4f0f-9b90-e17296d24613.wav",'1,2,3,4,5,6,7,1+'

# filename,notation = "F:/项目/花城音乐项目/样式数据/12.05MP3/wav/旋律/B-5.wav",'3,3,1,3,4,5,5,6,7,1+'

dir_list = ['F:/项目/花城音乐项目/样式数据/20.03.16MP3/wav/']
dir_list = ['F:/项目/花城音乐项目/样式数据/12.05MP3/wav/旋律/']

y,sr = librosa.load(filename)
rms = librosa.feature.rmse(y=y)[0]
rms = [x / np.std(rms) for x in rms]
time = librosa.get_duration(filename=filename)
print("time is {}".format(time))
CQT = librosa.amplitude_to_db(librosa.cqt(y, sr=16000), ref = np.max)
w,h = CQT.shape
print("w.h is {},{}".format(w,h))

snd = parselmouth.Sound(filename)
intensity = snd.to_intensity()
# plt.figure()
# plt.plot(snd.xs(), snd.values.T)
# plt.xlim([snd.xmin, snd.xmax])
# plt.xlabel("time [s]")
# plt.ylabel("amplitude")
# plt.show() # or plt.savefig("sound.png"), or plt.savefig("sound.pdf")
# # print(snd.xs())
# print(snd.xs().size)
# print(snd.values.shape)

pitch = get_pitch_by_parselmouth(filename)
# print(pitch)
print(pitch.tmax)  #
print(pitch.n_frames) #总帧数
print(pitch.dt) #每帧时长
print(pitch.t1) #超始帧中点
print(pitch.duration) #持续时长
frames_total = int(np.floor((pitch.duration - pitch.t1)/pitch.dt) -1)
print("frames_total is {}".format(frames_total))

mean_pitch = get_mean_pitch(20,75,sr,pitch)
print("mean_pitch is {}".format(mean_pitch))
mean_pitch = get_mean_pitch(100,150,sr,pitch)
print("mean_pitch is {}".format(mean_pitch))

onset_frames = [31, 59, 72, 90, 110, 148, 185, 203, 221]
onset_times = librosa.frames_to_time(onset_frames, sr=sr)
pitch_values = pitch.selected_array['frequency']
# print(pitch_values)
onset_frames.append(350)
mean_pitchs = [get_mean_pitch(onset_frames[i],onset_frames[i+1],sr,pitch) for i in range(0,len(onset_frames)-1)]

print(mean_pitchs)
mean_pitchs = [p for p in mean_pitchs if p is not None]
print(pitch_values.size)
# onset_frames = [i for i in range(1,pitch_values.size) if pitch_values[i-1] == 0 and pitch_values[i] != 0]
# onset_times = [pitch.duration * o /pitch.n_frames for o in onset_frames]
# print(np.mean(pitch_values))
pre_emphasized_snd = snd.copy()
pre_emphasized_snd.pre_emphasize()
spectrogram = pre_emphasized_snd.to_spectrogram(window_length=0.03, maximum_frequency=4000)
plt.figure()
# draw_spectrogram(spectrogram)
# plt.twinx()
draw_pitch(pitch,draw_type=0,filename=filename.split("/")[-1],notation=notation)
plt.xlim([snd.xmin, snd.xmax])
# plt.twinx()  # 共X轴，用来画声音强度信息
# draw_intensity(intensity)
# plt.vlines(onset_times, 0, 500, color='y', linestyle='--')
# plt.hlines(mean_pitchs, 0, 8, color='r', linestyle='--')
plt.show()

# batch_draw_samples(dir_list)