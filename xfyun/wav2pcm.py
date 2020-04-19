# -*- coding:utf-8 -*-
import os
import numpy as np
import numpy as np
def wav2pcm(wavfile):
    str = "F:/ffmpeg/bin/ffmpeg -y -i " + wavfile + " -acodec pcm_s16le -f s16le -ac 1 -ar 16000 " + wavfile.split('.wav')[0] + ".pcm"
    print(str)
    # os.system("ffmpeg -i "+ m4a_path + m4a + " " + m4a_path + m4a.split('.')[0] + ".wav" )   # ffmpeg 可以用来处理视频和音频文件，可               #以设置一些参数，如采样频率等
    os.system(str)
if __name__ == "__main__":

    wavfile = 'F:/项目/花城音乐项目/样式数据/20.04.01MP3/人声打分.wav'
    wavfile = 'F:/项目/花城音乐项目/样式数据/12.05MP3/wav/旋律/B-6.wav'
    wavfile = 'F:/项目/花城音乐项目/样式数据/20.04.07MP3/人声_1549.wav'
    wavfile = 'F:/项目/花城音乐项目/样式数据/20.04.08MP3/录音.wav'
    wav2pcm(wavfile)