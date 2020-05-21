import parselmouth

import numpy as np
import matplotlib.pyplot as plt
import librosa
from parselmouth_util import *
from xfyun.iat_ws_python3 import *
import os
from collections import Counter
from xfyun.wav2pcm import *
from pingfen_uitl import get_lossed_standard_notations,get_lcseque_and_position,get_lcseque_and_position_with_time_offset

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


filename,notation = "F:/项目/花城音乐项目/音符起始点检测/note_onset_detection-master/audio/F-2.wav",''
filename,notation = "F:/项目/花城音乐项目/音高检测/Pitch-Class-Detection-using-ML-master/resources/training-set/Gb3.mp3",''
filename,notation = "F:/项目/花城音乐项目/音高检测/Pitch-Class-Detection-using-ML-master/resources/training-set/C4.mp3",''
filename,notation = "F:/项目/花城音乐项目/样式数据/20.03.16MP3/wav/1a981498-eb19-4f0f-9b90-e17296d24613.wav",''
filename,notation = "F:/项目/花城音乐项目/样式数据/20.03.16MP3/wav/ef4f2b7e-6e2c-4af8-976a-1cf6c73f9f92.wav",''
# filename,notation = "F:/项目/花城音乐项目/样式数据/20.03.16MP3/wav/80ae0cb6-4fda-4d0e-8df4-a3c628857f6c.wav",''
# filename,notation = "F:/项目/花城音乐项目/样式数据/20.03.16MP3/wav/10604869-014c-4bf4-a7d9-fc25a84dd37e.wav",''
# filename,notation = "F:/项目/花城音乐项目/样式数据/20.03.16MP3/wav/2adcc430-0dc8-4c7e-bb24-ec36dd0f82ae.wav",''
# filename,notation = "F:/项目/花城音乐项目/样式数据/6.24MP3/旋律/两只老虎20190624-7881.wav",'' #[1,2,3,1,1,2,3,1,3,4,5,3,4,5]
# filename,notation = "F:/项目/花城音乐项目/样式数据/6.24MP3/旋律/两只老虎20190624-8973.wav",''
# filename,notation = "F:/项目/花城音乐项目/样式数据/20.03.16MP3/wav/7bce3e27-5239-4c79-b912-f2870fde5814.wav",''
# filename,notation = "F:/项目/花城音乐项目/样式数据/20.03.16MP3/wav/2f4a4416-dbb7-4584-b9f9-f952372161af.wav",''  #有个短音符没识别出来
# filename,notation = "F:/项目/花城音乐项目/样式数据/20.03.16MP3/wav/74c301cd-b57d-4fe9-b44d-667b587570f5.wav",'1+,7,6,5,6,5,4,3,2,1'  #有个音符没识别出来
# filename,notation = "F:/项目/花城音乐项目/样式数据/20.03.16MP3/wav/2adcc430-0dc8-4c7e-bb24-ec36dd0f82ae.wav",''
# filename,notation = "F:/项目/花城音乐项目/样式数据/20.03.16MP3/wav/1a981498-eb19-4f0f-9b90-e17296d24613.wav",'1,2,3,4,5,6,7,1+'

# filename,notation = "F:/项目/花城音乐项目/样式数据/12.05MP3/wav/旋律/B-5.wav",'3,3,1,3,4,5,5,6,7,1+'
# filename,notation = "F:/项目/花城音乐项目/样式数据/12.05MP3/wav/旋律/F-8.wav",'1,3,4,5,6,6,1+,7,6,1+'
# filename,notation = "F:/项目/花城音乐项目/样式数据/9.08MP3/旋律/zx1.wav",'3,3,1,3,4,5,5,6,7,1+'
# filename,notation = "F:/项目/花城音乐项目/样式数据/20.03.16MP3/wav/2f4a4416-dbb7-4584-b9f9-f952372161af.wav",'6,5,3,6,3,5,3,2,1,6-'
# filename,notation = "F:/项目/花城音乐项目/样式数据/20.03.16MP3/wav/522d003d-119d-4f4e-a0ba-f70c2ea351fb.wav",'1,3,5,3,1'
# filename,notation = "F:/项目/花城音乐项目/样式数据/20.03.16MP3/wav/9533c6e5-cd48-437b-8609-97038ff87af8.wav",'5'
# filename,notation = 'F:/项目/花城音乐项目/样式数据/6.24MP3/旋律/两只老虎20190624-2939.wav','1,2,3,1,1,2,3,1,3,4,5,3,4,5'
# filename,notation = 'F:/项目/花城音乐项目/样式数据/6.24MP3/旋律/两只老虎20190624-1089.wav',  '[1,2,3,1,1,2,3,1,3,4,5,3,4,5]'
# filename,notation = 'F:/项目/花城音乐项目/样式数据/20.03.26MP3/wav/CI1.1.wav','3,3,2,1,1,7-,6-,6-,6-,4,4,3,2,1,2,4,3'
# filename,notation = 'F:/项目/花城音乐项目/样式数据/20.03.26MP3/wav/CI1.wav','3,3,2,1,1,7-,6-,6-,6-,4,4,3,2,1,2,4,3,4,4,3,2,2,4,3,3,1,6-,6-,7-,3,2,1,7-,1,6-'
# filename,notation = 'F:/项目/花城音乐项目/样式数据/20.03.26MP3/wav/CI1_0010.wav','3,3,2,1,1,7-,6-,6-,6-,4,4,3,2,1,2'
# filename,notation = 'F:/项目/花城音乐项目/样式数据/20.03.26MP3/wav/CI2.wav','3,3,2,1,1,7-,6-,6-,6-,4,4,3,2,1,2,4,3,4,4,3,2,2,4,3,3,1,6-,6-,7-,3,2,1,7-,1,6-'
# filename,notation = 'F:/项目/花城音乐项目/样式数据/20.03.26MP3/wav/LCI1.wav','1,2,3,1,1,2,3,1,3,4,5,3,4,5'
# filename,notation = 'F:/项目/花城音乐项目/样式数据/20.03.26MP3/wav/MING1.wav','3,3,2,1,1,7-,6-,6-,6-,4,4,3,2,1,2,4,3,4,4,3,2,2,4,3,3,1,6-,6-,7-,3,2,1,7-,1,6-'
# filename,notation = 'F:/项目/花城音乐项目/样式数据/20.03.26MP3/wav/MING1_a.wav','3,3,2,1,1,7-,6-,6-,6-,4,4,3,2,1,2,4,3,4,4,3,2,2,4,3,3,1,6-,6-,7-,3,2,1,7-,1,6-'
# filename,notation = 'F:/项目/花城音乐项目/样式数据/20.03.26MP3/wav/MING1_0010.wav','[3,3,2,1,1,7-,6-,6-,6-,4,4,3,2,1,2]'
# filename, notation = 'F:/项目/花城音乐项目/样式数据/20.03.26MP3/wav/MING2.wav',  '3,3,2,1,1,7-,6-,6-,6-,4,4,3,2,1,2,4,3,4,4,3,2,2,4,3,3,1,6-,6-,7-,3,2,1,7-,1,6-'

# filename, notation = 'F:/项目/花城音乐项目/样式数据/12.05MP3/wav/旋律/B-6.wav',  '3,3,1,3,4,5,5,6,7,1+'
# filename, notation = 'F:/项目/花城音乐项目/样式数据/12.05MP3/wav/旋律/D-6.wav',  '1+,7,6,5,4,5,4,3,2,1'
# filename, notation = 'F:/项目/花城音乐项目/样式数据/12.05MP3/wav/旋律/F-9.wav',  '1,3,4,5,6,6,1+,7,6,1+'
# filename, notation = 'F:/项目/花城音乐项目/样式数据/12.05MP3/wav/旋律/H-5.wav',  '1+,7,6,5,6,5,4,3,2,1'
# filename, notation = 'F:/项目/花城音乐项目/样式数据/20.04.01MP3/乐器演奏打分.wav',  '3,3,2,1,1,7-,6-,6-,6-,4,4,3,2,1,2,4,3,4,4,3,2,2,4,3,3,1,6-,6-,7-,3,2,1,7-,1,6-'
# filename, notation = 'F:/项目/花城音乐项目/样式数据/20.04.01MP3/人声打分.wav',  '3,3,2,1,1,7-,6-,6-,6-,4,4,3,2,1,2,4,3,4,4,3,2,2,4,3,3,1,6-,6-,7-,3,2,1,7-,1,6-'

######################202-04-03#########################
# filename, notation = "F:/项目/花城音乐项目/样式数据/20.03.16MP3/wav/2f4a4416-dbb7-4584-b9f9-f952372161af.wav", '[6,5,3,6,3,5,3,2,1,6-]'
# filename, notation  = 'F:/项目/花城音乐项目/样式数据/20.03.16MP3/wav/4c1590c4-8a8d-4920-80dd-e4fc9e7a1d95.wav','[3,3,1,3,4,5,5,6,7,1+]'
# filename, notation = 'F:/项目/花城音乐项目/样式数据/12.05MP3/wav/旋律/H-5.wav',  '1+,7,6,5,6,5,4,3,2,1'
# # filename, notation = 'F:/项目/花城音乐项目/样式数据/20.04.01MP3/人声打分.wav', '3,3,2,1,1,7-,6-,6-,6-,4,4,3,2,1,2,4,3,4,4,3,2,2,4,3,3,1,6-,6-,7-,3,2,1,7-,1,6-'
# filename, notation = 'F:/项目/花城音乐项目/样式数据/12.05MP3/wav/旋律/F-5.wav',  '1,3,4,5,6,6,1+,7,6,1+'
# filename,notation = 'F:/项目/花城音乐项目/样式数据/20.03.26MP3/wav/CI1_0010.wav','3,3,2,1,1,7-,6-,6-,6-,4,4,3,2,1,2'
# filename,notation = 'F:/项目/花城音乐项目/样式数据/20.04.07MP3/人声_1549.wav','3,3,2,1,1,7-,6-,6-,6-,4,4,3,2,1,2,4,3,4,4,3,2,2,4,3,3,1,6-,6-,7-,3,2,1,7-,1,6-'
# filename,notation = 'F:/项目/花城音乐项目/样式数据/20.04.07MP3/人声_1649.wav','3,3,2,1,1,7-,6-,6-,6-,4,4,3,2,1,2,4,3,4,4,3,2,2,4,3,3,1,6-,6-,7-,3,2,1,7-,1,6-'
######################202-04-03#########################

######################202-04-08#########################
# filename,notation = 'F:/项目/花城音乐项目/样式数据/20.04.08MP3/2段词-标准.wav','3,3,2,1,1,7-,6-,6-,6-,4,4,3,2,1,2,4,3,4,4,3,2,2,4,3,3,1,6-,6-,7-,3,2,1,7-,1,6-'
filename,notation = 'F:/项目/花城音乐项目/样式数据/20.04.08MP3/2段词-标准1648.wav','3,3,2,1,1,7-,6-,6-,6-,4,4,3,2,1,2,4,3,4,4,3,2,2,4,3,3,1,6-,6-,7-,3,2,1,7-,1,6-'
# filename,notation = 'F:/项目/花城音乐项目/样式数据/20.04.08MP3/2段词-标准4882.wav','3,3,2,1,1,7-,6-,6-,6-,4,4,3,2,1,2,4,3,4,4,3,2,2,4,3,3,1,6-,6-,7-,3,2,1,7-,1,6-'
# filename,notation = 'F:/项目/花城音乐项目/样式数据/20.04.08MP3/2段词-标准1681.wav','3,3,2,1,1,7-,6-,6-,6-,4,4,3,2,1,2,4,3,4,4,3,2,2,4,3,3,1,6-,6-,7-,3,2,1,7-,1,6-'
# filename,notation = 'F:/项目/花城音乐项目/样式数据/20.04.08MP3/2段词-不标准.wav','3,3,2,1,1,7-,6-,6-,6-,4,4,3,2,1,2,4,3,4,4,3,2,2,4,3,3,1,6-,6-,7-,3,2,1,7-,1,6-'
# filename,notation = 'F:/项目/花城音乐项目/样式数据/20.04.08MP3/2段词-不标准1648.wav','3,3,2,1,1,7-,6-,6-,6-,4,4,3,2,1,2,4,3,4,4,3,2,2,4,3,3,1,6-,6-,7-,3,2,1,7-,1,6-'
# filename,notation = 'F:/项目/花城音乐项目/样式数据/20.04.08MP3/2段词-不标准4882.wav','3,3,2,1,1,7-,6-,6-,6-,4,4,3,2,1,2,4,3,4,4,3,2,2,4,3,3,1,6-,6-,7-,3,2,1,7-,1,6-'
######################202-04-08#########################

######################202-04-28#########################
# filename, notation = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋3罗（80）.wav', '3,3,2,1,1,7-,6-,6-,6-,4,4,3,2,1,2,4,3,4,4,3,2,2,4,3,3,1,6-,6-,7-,3,2,1,7-,1,6-'
# filename, notation = 'F:/项目/花城音乐项目/样式数据/12.05MP3/wav/旋律/H-5.wav', '3,3,2,1,1,7-,6-,6-,6-,4,4,3,2,1,2,4,3,4,4,3,2,2,4,3,3,1,6-,6-,7-,3,2,1,7-,1,6-'
# ######################202-04-28#########################
#
# ######################202-04-30#########################
# filename, notation = 'F:/项目/花城音乐项目/样式数据/20.04.29MP3/wav/CI1.wav', '3,3,2,1,1,7-,6-,6-,6-,4,4,3,2,1,2,4,3,4,4,3,2,2,4,3,3,1,6-,6-,7-,3,2,1,7-,1,6-'
# filename, notation = 'F:/项目/花城音乐项目/样式数据/20.04.29MP3/wav/CI2.wav', '3,3,2,1,1,7-,6-,6-,6-,4,4,3,2,1,2,4,3,4,4,3,2,2,4,3,3,1,6-,6-,7-,3,2,1,7-,1,6-'
# filename, notation = 'F:/项目/花城音乐项目/样式数据/20.04.29MP3/wav/test1-1547.wav', '3,3,2,1,1,7-,6-,6-,6-,4,4,3,2,1,2,4,3,4,4,3,2,2,4,3,3,1,6-,6-,7-,3,2,1,7-,1,6-'
# filename, notation = 'F:/项目/花城音乐项目/样式数据/20.04.29MP3/wav/test2-1548.wav', '3,3,2,1,1,7-,6-,6-,6-,4,4,3,2,1,2,4,3,4,4,3,2,2,4,3,3,1,6-,6-,7-,3,2,1,7-,1,6-'
# filename, notation = 'F:/项目/花城音乐项目/样式数据/20.04.29MP3/wav/test3-1547.wav', '3,3,2,1,1,7-,6-,6-,6-,4,4,3,2,1,2,4,3,4,4,3,2,2,4,3,3,1,6-,6-,7-,3,2,1,7-,1,6-'
# ######################202-04-30#########################
#
# filename, notation = 'F:/项目/花城音乐项目/样式数据/20.05.01MP3/wav/6749-1133.wav', '3,3,2,1,1,7-,6-,6-,6-,4,4,3,2,1,2,4,3,4,4,3,2,2,4,3,3,1,6-,6-,7-,3,2,1,7-,1,6-'
# filename, notation = 'F:/项目/花城音乐项目/样式数据/20.05.12MP3/wav/200508-4710-1548.wav', '3,3,2,1,1,7-,6-,6-,6-,4,4,3,2,1,2,4,3,4,4,3,2,2,4,3,3,1,6-,6-,7-,3,2,1,7-,1,6-'
# filename, notation = 'F:/项目/花城音乐项目/样式数据/20.05.12MP3/wav/200508-8312-1548.wav', '3,3,2,1,1,7-,6-,6-,6-,4,4,3,2,1,2,4,3,4,4,3,2,2,4,3,3,1,6-,6-,7-,3,2,1,7-,1,6-'
# filename, notation = 'F:/项目/花城音乐项目/样式数据/20.05.12MP3/wav/20200518-8354-1132.wav', '3,3,2,1,1,7-,6-,6-,6-,4,4,3,2,1,2,4,3,4,4,3,2,2,4,3,3,1,6-,6-,7-,3,2,1,7-,1,6-'
# filename, notation = 'F:/项目/花城音乐项目/样式数据/20.05.01MP3/dbg/3141/seg1.wav', '3,3,2,1,1,7-,6-,6-,6-,4,4,3,2,1,2,4,3,4,4,3,2,2,4,3,3,1,6-,6-,7-,3,2,1,7-,1,6-'
# filename, notation = 'F:/项目/花城音乐项目/样式数据/20.05.20MP3/wav/20200520-2360-1548.wav', '3,3,2,1,1,7-,6-,6-,6-,4,4,3,2,1,2,4,3,4,4,3,2,2,4,3,3,1,6-,6-,7-,3,2,1,7-,1,6-'
# filename, notation = 'F:/项目/花城音乐项目/样式数据/20.04.29MP3/wav/CI1.wav', '3,3,2,1,1,7-,6-,6-,6-,4,4,3,2,1,2,4,3,4,4,3,2,2,4,3,3,1,6-,6-,7-,3,2,1,7-,1,6-'
# filename, notation = 'F:/项目/花城音乐项目/样式数据/20.04.29MP3/wav/CI2.wav', '3,3,2,1,1,7-,6-,6-,6-,4,4,3,2,1,2,4,3,4,4,3,2,2,4,3,3,1,6-,6-,7-,3,2,1,7-,1,6-'


dir_list = ['F:/项目/花城音乐项目/样式数据/20.03.16MP3/wav/']
dir_list = ['F:/项目/花城音乐项目/样式数据/12.05MP3/wav/旋律/']


#标准时间线
standard_notation_time = [0,1,1.5,2,3,3.5,4,5,6,8,9,9.5,10,10.5,11,11.5,12,16,17,17.5,18,19,19.5,20,21,21.5,22,23,24,25,26,26.5,27,27.5,28]
# standard_notation_time = [0,0.6818181818181817,1.0227272727272734,1.363636363636365,2.0454545454545467,2.3863636363636385,2.72727272727273,3.409090909090912,4.0909090909090935,5.454545454545459,6.136363636363642,6.477272727272732,6.818181818181822,7.159090909090912,7.500000000000002,7.840909090909092,8.181818181818182,10.909090909090908,11.590909090909092,11.931818181818182,12.272727272727272,12.954545454545455,13.295454545454545,13.636363636363635,14.318181818181818,14.659090909090908,14.999999999999998,15.681818181818182,16.363636363636367,17.045454545454547,17.727272727272734,18.06818181818182,18.409090909090914,18.75,19.090909090909093,21.81818181818182]
standard_notations = '3,3,2,1,1,7-,6-,6-,6-,4,4,3,2,1,2,4,3,4,4,3,2,2,4,3,3,1,6-,6-,7-,3,2,1,7-,1,6- '
sns_list = standard_notations.split(',')

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
print("pitch.dt is {}".format(pitch.dt)) #每帧时长
print(pitch.t1) #超始帧中点
print(pitch.duration) #持续时长
frames_total = int(np.floor((pitch.duration - pitch.t1)/pitch.dt) -1)
print("frames_total is {}".format(frames_total))

mean_pitch = get_mean_pitch(20,75,sr,pitch)
print("mean_pitch is {}".format(mean_pitch))
mean_pitch = get_mean_pitch(100,150,sr,pitch)
print("mean_pitch is {}".format(mean_pitch))

# onset_frames = [35, 41, 90, 111, 137, 253, 261, 287, 308, 313, 335, 340]
onset_frames = [20, 44, 63, 78, 92, 109, 166, 182, 192, 218, 242]
onset_times = librosa.frames_to_time(onset_frames, sr=sr)
#换算成parselmouth的帧数
changed_onset_frames = [int(pitch.n_frames * t /pitch.duration) for t in onset_times]
print("changed_onset_frames is {},size is {}".format(changed_onset_frames,len(changed_onset_frames)))
pitch_values = pitch.selected_array['frequency']
# pitch_values_candidate = get_pitch_values(pitch_values)
# test_frames = [i for i in range(len(pitch_values) - 30) if np.abs(pitch_values_candidate[i] - pitch_values_candidate[i+1]) > 5 and Counter(pitch_values_candidate[i+1:i+30]).most_common(1)[0][1] > 24 and Counter(pitch_values_candidate[i+1:i+30]).most_common(1)[0][0] > 75]
# print("test_frames is {},size is {}".format(test_frames,len(test_frames)))
# test_onset_times = [pitch.duration * t /pitch.n_frames for t in test_frames]
# test_frames,test_onset_times = get_starts_by_absolute_pitch_with_filename(filename)
small_or_big = 'small'
# test_frames,test_onset_times = get_short_starts_by_absolute_pitch(pitch,small_or_big)
# test_frames,test_onset_times = get_starts_by_absolute_pitch(pitch,small_or_big)
test_frames,test_onset_times = get_all_starts_by_absolute_pitch(pitch,small_or_big)
start_frames_on_librosa = librosa.time_to_frames(test_onset_times)
start_frames_on_librosa = list(start_frames_on_librosa)
print("1 test_frames is {},size is {}".format(test_frames,len(test_frames)))
all_first_candidate_names, all_first_candidates, all_offset_types = get_all_numbered_notation_and_offset(pitch,start_frames_on_librosa)
print("1 all_first_candidate_names is {},size is {}".format(all_first_candidate_names,len(all_first_candidate_names)))
# test_frames,test_onset_times =check_all_starts(pitch,start_frames_on_librosa,test_onset_times)
print("1 test_frames is {},size is {}".format(test_frames,len(test_frames)))
all_first_candidate_names,all_second_candidate_names,all_first_candidates,all_second_candidates = get_all_pitch_candidate(pitch,onset_frames.copy(),sr)
# print("all_first_candidate_names is {},size is {}".format(all_first_candidate_names,len(all_first_candidate_names)))
# print("onset_frames is {},size is {}".format(onset_frames,len(onset_frames)))
result = change_pitch_seque(all_first_candidate_names,all_first_candidates)
print(result)

# ###### 语音识别===========
if filename.find(".wav") >= 0:
    wav2pcm(filename)
pcmfile = filename.split(".wav")[0] + ".pcm"
all_message, all_detail = get_result_from_xfyun(pcmfile)
detail_time = [round((value)/100,2) for value in all_detail.keys() if value > 0]
# detail_time.sort()
print(all_detail)
print(detail_time)
# print(all_message)
# all_detail = {20: ('惜', 0), 144: ('爱', 1), 236: ('春天', 2), 392: ('的', 4), 440: ('人', 5), 552: ('儿', 6), 640: ('时', 7), 824: ('心地', 8), 1040: ('纯洁', 10), 1188: ('的', 12), 1624: ('相思', 13), 1832: ('罗', 15), 1936: ('兰花', 16), 2100: ('花儿', 18), 2240: ('一样', 20), 2425: ('是', 22), 2545: ('我', 23), 2645: ('知心', 24), 2745: ('朋友', 26), 0: ('', 28)}
# detail_time = [0.2, 1.44, 2.36, 3.92, 4.4, 5.52, 6.4, 8.24, 10.4, 11.88, 16.24, 18.32, 19.36, 21.0, 22.4, 24.25, 25.45, 26.45, 27.45]
# all_message = '惜爱春天的人儿时心地纯洁的相思罗兰花花儿一样是我知心朋友'

first_time,last_time,_ = get_start_and_end_with_parselmouth(filename)
onset_times = detail_time
# offset = onset_times[0] - first_time
# offset = 0
# onset_times = [o - offset for o in onset_times]
print(all_detail)
print(detail_time)
print(all_message)

###### 语音识别 end ===========

pitch_code = '[3,3,1,3,4,5,5,6,7,1+]'
get_best_relative_pitch(pitch,pitch_code)

onset_frames = start_frames_on_librosa
all_first_candidate_names, all_first_candidates, all_offset_types = get_all_numbered_notation_and_offset(pitch,onset_frames)
print("2 all_first_candidate_names is {},size is {}".format(all_first_candidate_names,len(all_first_candidate_names)))
print("2 all_first_candidate_names is {},size is {}".format(all_offset_types,len(all_offset_types)))
numbered_notations,detail = get_all_numbered_musical_notation_by_moved(3,all_first_candidate_names,test_onset_times)
print("2 numbered_notations is {},size is {}".format(numbered_notations,len(numbered_notations)))
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
# draw_pitch(pitch,draw_type=0,filename=filename.split("/")[-1],notation=notation,grain_size=0)
# pitch_values = pitch_values + 120
import scipy.signal as signal
# b, a = signal.butter(8, 0.2, analog=False)
# pitch_values = signal.filtfilt(b, a, pitch_values)
pitch_values = signal.medfilt(pitch_values, 35)
print("pitch_values size is {}".format(len(pitch_values)))
pitch_values_on_test_frames = [np.abs(round(pitch_values[t-2] - pitch_values[t+2],2)) for t in test_frames]
print("pitch_values_on_test_frames is {}".format(pitch_values_on_test_frames))
intensity = snd.to_intensity()
draw_pitch_specified(intensity,pitch,pitch_values,draw_type=0,filename=filename.split("/")[-1],notation=notation,grain_size=1)
plt.xlim([snd.xmin, snd.xmax])
starts_by_parselmouth_rms,starts_by_parselmouth_rms_times = get_starts_by_parselmouth_rms(intensity,pitch)
print("starts_by_parselmouth_rms is {},size is {}".format(starts_by_parselmouth_rms,len(starts_by_parselmouth_rms)))
# plt.twinx()  # 共X轴，用来画声音强度信息
# draw_intensity(intensity)
# 平移语音识别的时间点
onset_times = [t - (onset_times[0] - test_onset_times[0]) for t in onset_times]
plt.vlines(onset_times, 0, 40, color='y', linestyle='-.')
# plt.vlines(test_onset_times, 0, 500, color='r', linestyle='--')
# plt.vlines(starts_by_parselmouth_rms_times, 0, 500, color='b', linestyle='--')
# plt.hlines(mean_pitchs, 0, 8, color='r', linestyle='--')
merge_times = merge_times_from_iat_plm_rms(onset_times,test_onset_times,starts_by_parselmouth_rms_times)
print("3 merge_times is {},size is {}".format(merge_times,len(merge_times)))
merge_frames = librosa.time_to_frames(merge_times)
plt.vlines(merge_times, 0, 30, color='b', linestyle='-')
all_first_candidate_names, all_first_candidates, all_offset_types = get_all_numbered_notation_and_offset(pitch,merge_frames)
print("3 all_first_candidate_names is {},size is {}".format(all_first_candidate_names,len(all_first_candidate_names)))
numbered_notations,numbered_notations_detail = get_all_numbered_musical_notation_by_moved(3,all_first_candidate_names,merge_times)
print("3 numbered_notations is {},size is {}".format(numbered_notations,len(numbered_notations)))
print("3 numbered_notations_detail is {},size is {}".format(numbered_notations_detail,len(numbered_notations_detail)))
start_time,end_time = 17.1,25.78
selected_numbered_notations_detail = get_notation_detail_by_times(numbered_notations_detail,start_time,end_time)
print("3 selected_numbered_notations_detail is {},size is {}".format(selected_numbered_notations_detail,len(selected_numbered_notations_detail)))
# 打印音高(字母)
for i,c in enumerate(all_first_candidate_names):
    k = merge_times[i]
    plt.text(k, 20, c, size='8',color='r')

# # 打印音高(数字)
# for i,c in enumerate(numbered_notations):
#     k = merge_times[i]
#     plt.text(k, 12, c, size='8',color='r')

# 打印歌词
t_offset = (detail_time[0] - test_onset_times[0])
for (k,v) in  all_detail.items():
    plt.text(k/100 - t_offset, 3, v[0], size='8',color='r')

#标准时间线
# standard_notation_time = [0,1,1.5,2,3,3.5,4,5,6,8,9,9.5,10,10.5,11,11.5,12,16,17,17.5,18,19,19.5,20,21,21.5,22,23,24,25,26,26.5,27,27.5,28]
# standard_time = [t for t in standard_notation_time]
tmp_points = [t for i,t in enumerate(merge_times) if numbered_notations[i] is not None]
firt_offset = tmp_points[0]
standard_time = [t+firt_offset for t in standard_notation_time]
plt.vlines(standard_time, 0, 500, color='g', linestyle=':')
standard_notations = '3,3,2,1,1,7-,6-,6-,6-,4,4,3,2,1,2,4,3,4,4,3,2,2,4,3,3,1,6-,6-,7-,3,2,1,7-,1,6-'
standard_notations = standard_notations.split(',')
standard_notations_times = zip(standard_time,standard_notations)

standard_notations = [s[0] for s in standard_notations]
standard_notations = ''.join(standard_notations)
print("standard_notations is {},size is {}".format(standard_notations, len(standard_notations)))
loss_positions, loss_notations_in_standard = get_lossed_standard_notations(standard_notations, numbered_notations)

#找出未匹配的音高，并对未匹配的每个音高进行分析
time_offset_threshold = 2
lcseque, standard_positions, test_positions = get_lcseque_and_position(standard_notations, numbered_notations)
print("standard_notations is {}".format(standard_notations))
print("numbered_notations is {}".format(numbered_notations))
print("standard_notation_time is {}".format(standard_notation_time))
print("merge_times is {}".format(merge_times))
lcseque, standard_positions, test_positions = get_lcseque_and_position_with_time_offset(standard_notations, numbered_notations, standard_notation_time, merge_times)
# standard_positions_times = [standard_notation_time[i] for i in standard_positions]
# test_positions_times = [merge_times[i] for i in test_positions]
# # 比较匹配点的时间偏差值
# times_offset = [np.abs(standard_positions_times[i] - t) for i, t in enumerate(test_positions_times)]
# # 如果时间偏差值较大，该匹配点记为未匹配
# loss_positions_by_times = [standard_positions[i] for i, t in enumerate(times_offset) if t > time_offset_threshold]
# loss_notations_by_times = [lcseque[i] for i, t in enumerate(times_offset) if t > time_offset_threshold]
# # loss_positions = loss_positions + loss_positions_by_times
# loss_positions.sort()

score_seted = 30
standard_notations = '3,3,2,1,1,7-,6-,6-,6-,4,4,3,2,1,2,4,3,4,4,3,2,2,4,3,3,1,6-,6-,7-,3,2,1,7-,1,6-'
# pitch_total_score, pitch_score_detail = pitch_score(standard_notations,numbered_notations,score_seted)
# print("pitch_total_score is {}".format(pitch_total_score))
# print("pitch_score_detail is {}".format(pitch_score_detail))
# 打印标准音符
for i,(k,v) in enumerate(standard_notations_times):
    if i not in standard_positions: # 未匹配的
        plt.text(k, 6, v[0], size='20', color='r')
    else:
        plt.text(k, 6, v[0], size='12',color='g')

# 打印音高(数字)
for i,c in enumerate(numbered_notations):
    k = merge_times[i]
    if i not in test_positions:  # 未匹配的
        plt.text(k, 12, c, size='20', color='r')
    else:
        plt.text(k, 12, c, size='12', color='y')
plt.show()

# batch_draw_samples(dir_list)