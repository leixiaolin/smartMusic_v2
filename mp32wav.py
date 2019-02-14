import os

mp3_path = "F:/项目/花城音乐项目/样式数据/音乐样本2019-01-29/ALL节奏所有的/"
save_path= 'F:/项目/花城音乐项目/样式数据/ALL/节奏/2019-01-29/'

mp3_file = os.listdir(mp3_path)
for mp3 in mp3_file:
    if mp3.find("节奏") >= 0:
        #print(mp3_path + mp3.split('.')[0])
        str = "F:/ffmpeg/bin/ffmpeg -i "+ mp3_path + mp3 + " " + save_path + mp3.split('mp3')[0] + "wav"
        print(str)
        #os.system("ffmpeg -i "+ mp3_path + mp3 + " " + mp3_path + mp3.split('.')[0] + ".wav" )   # ffmpeg 可以用来处理视频和音频文件，可               #以设置一些参数，如采样频率等
        os.system(str)
