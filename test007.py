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
filename, notation = 'F:/项目/花城音乐项目/样式数据/12.05MP3/wav/旋律/H-5.wav', '3,3,2,1,1,7-,6-,6-,6-,4,4,3,2,1,2,4,3,4,4,3,2,2,4,3,3,1,6-,6-,7-,3,2,1,7-,1,6-'
# ######################202-04-28#########################
#
# ######################202-04-30#########################
# filename, notation = 'F:/项目/花城音乐项目/样式数据/20.04.29MP3/wav/CI1.wav', '3,3,2,1,1,7-,6-,6-,6-,4,4,3,2,1,2,4,3,4,4,3,2,2,4,3,3,1,6-,6-,7-,3,2,1,7-,1,6-'
# filename, notation = 'F:/项目/花城音乐项目/样式数据/20.04.29MP3/wav/CI2.wav', '3,3,2,1,1,7-,6-,6-,6-,4,4,3,2,1,2,4,3,4,4,3,2,2,4,3,3,1,6-,6-,7-,3,2,1,7-,1,6-'
# filename, notation = 'F:/项目/花城音乐项目/样式数据/20.04.29MP3/wav/test1-1547.wav', '3,3,2,1,1,7-,6-,6-,6-,4,4,3,2,1,2,4,3,4,4,3,2,2,4,3,3,1,6-,6-,7-,3,2,1,7-,1,6-'
# filename, notation = 'F:/项目/花城音乐项目/样式数据/20.04.29MP3/wav/test2-1548.wav', '3,3,2,1,1,7-,6-,6-,6-,4,4,3,2,1,2,4,3,4,4,3,2,2,4,3,3,1,6-,6-,7-,3,2,1,7-,1,6-'
filename, notation = 'F:/项目/花城音乐项目/样式数据/20.04.29MP3/wav/test3-1547.wav', '3,3,2,1,1,7-,6-,6-,6-,4,4,3,2,1,2,4,3,4,4,3,2,2,4,3,3,1,6-,6-,7-,3,2,1,7-,1,6-'
# ######################202-04-30#########################
#
# filename, notation = 'F:/项目/花城音乐项目/样式数据/20.05.01MP3/wav/6749-1133.wav', '3,3,2,1,1,7-,6-,6-,6-,4,4,3,2,1,2,4,3,4,4,3,2,2,4,3,3,1,6-,6-,7-,3,2,1,7-,1,6-'
# filename, notation = 'F:/项目/花城音乐项目/样式数据/20.05.12MP3/wav/200508-4710-1548.wav', '3,3,2,1,1,7-,6-,6-,6-,4,4,3,2,1,2,4,3,4,4,3,2,2,4,3,3,1,6-,6-,7-,3,2,1,7-,1,6-'
# filename, notation = 'F:/项目/花城音乐项目/样式数据/20.05.12MP3/wav/200508-8312-1548.wav', '3,3,2,1,1,7-,6-,6-,6-,4,4,3,2,1,2,4,3,4,4,3,2,2,4,3,3,1,6-,6-,7-,3,2,1,7-,1,6-'
# filename, notation = 'F:/项目/花城音乐项目/样式数据/20.05.12MP3/wav/20200518-8354-1132.wav', '3,3,2,1,1,7-,6-,6-,6-,4,4,3,2,1,2,4,3,4,4,3,2,2,4,3,3,1,6-,6-,7-,3,2,1,7-,1,6-'
filename, notation = 'F:/项目/花城音乐项目/样式数据/20.05.01MP3/dbg/3141/seg1.wav', '3,3,2,1,1,7-,6-,6-,6-,4,4,3,2,1,2,4,3,4,4,3,2,2,4,3,3,1,6-,6-,7-,3,2,1,7-,1,6-'
# filename, notation = 'F:/项目/花城音乐项目/样式数据/20.05.20MP3/wav/20200520-2360-1548.wav', '3,3,2,1,1,7-,6-,6-,6-,4,4,3,2,1,2,4,3,4,4,3,2,2,4,3,3,1,6-,6-,7-,3,2,1,7-,1,6-'
# filename, notation = 'F:/项目/花城音乐项目/样式数据/20.04.29MP3/wav/CI1.wav', '3,3,2,1,1,7-,6-,6-,6-,4,4,3,2,1,2,4,3,4,4,3,2,2,4,3,3,1,6-,6-,7-,3,2,1,7-,1,6-'
# filename, notation = 'F:/项目/花城音乐项目/样式数据/20.04.29MP3/wav/CI2.wav', '3,3,2,1,1,7-,6-,6-,6-,4,4,3,2,1,2,4,3,4,4,3,2,2,4,3,3,1,6-,6-,7-,3,2,1,7-,1,6-'
# filename, notation = 'F:/项目/花城音乐项目/样式数据/20.05.26MP3/wav/20200526-8406.wav', '3,3,2,1,1,7-,6-,6-,6-,4,4,3,2,1,2,4,3,4,4,3,2,2,4,3,3,1,6-,6-,7-,3,2,1,7-,1,6-'
filename, notation = 'F:/项目/花城音乐项目/样式数据/20.05.26MP3/wav/20200526-1002.wav', '3,3,2,1,1,7-,6-,6-,6-,4,4,3,2,1,2,4,3,4,4,3,2,2,4,3,3,1,6-,6-,7-,3,2,1,7-,1,6-'

snd = parselmouth.Sound(filename)
intensity = snd.to_intensity()
pitch = get_pitch_by_parselmouth(filename)
print(pitch.duration) #持续时长
pitch_values = pitch.selected_array['frequency']
import scipy.signal as signal
# b, a = signal.butter(8, 0.2, analog=False)
# pitch_values = signal.filtfilt(b, a, pitch_values)
pitch_values = signal.medfilt(pitch_values, 35)
print("pitch_values size is {}".format(len(pitch_values)))
intensity = snd.to_intensity()
# draw_intensity(intensity)
values = intensity.values.T.copy()
values = list(values)
values = [v[0] for v in values]
values = signal.medfilt(values, 11)
tmp = len(values)
# values[int(tmp/2):] = 0
plt.plot(intensity.xs(), intensity.values.T, linewidth=1)
plt.plot(intensity.xs(), values, linewidth=1)
plt.grid(False)
plt.ylim(0)
plt.ylabel("intensity [dB]")

print("intensity len is {}".format(len(intensity.xs())))
values = intensity.values.T.copy()
values = list(values)
print("intensity values len is {}".format(len(values)))
print("intensity values max is {}".format(np.max(values)))
# draw_pitch_specified(intensity,pitch,pitch_values,draw_type=0,filename=filename.split("/")[-1],notation=notation,grain_size=1)
plt.xlim([snd.xmin, snd.xmax])
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.title(filename, fontsize=16)
plt.hlines(40, 0, pitch.duration, color='r', linestyle='--')
plt.text(1, 41, '评判线（背景噪声线）', size='12', color='r')
plt.show()

# batch_draw_samples(dir_list)