import os

mp3_path = "F:/项目/花城音乐项目/音高检测/Pitch-Class-Detection-using-ML-master/resources/training-set/"
save_path = "F:/项目/花城音乐项目/音高检测/Pitch-Class-Detection-using-ML-master/resources/wav/train/"

mp3_file = os.listdir(mp3_path)
for mp3 in mp3_file:
    if mp3.find("mp3") >= 0:
        mp3 = mp3.replace(" ","-")
        #print(mp3_path + mp3.split('.')[0])
        str = "F:/ffmpeg/bin/ffmpeg -i "+ mp3_path + mp3 + " " + save_path + mp3.split('mp3')[0] + "wav"
        print(str)
        #os.system("ffmpeg -i "+ mp3_path + mp3 + " " + mp3_path + mp3.split('.')[0] + ".wav" )   # ffmpeg 可以用来处理视频和音频文件，可               #以设置一些参数，如采样频率等
        os.system(str)


# ===========================================================================================
# 旋律
# ===========================================================================================
# mp3_path = "F:/项目/花城音乐项目/样式数据/2.18MP3/"
# save_path= 'F:/项目/花城音乐项目/样式数据/ALL/旋律/2.18MP3/'
#
# mp3_file = os.listdir(mp3_path)
# for mp3 in mp3_file:
#     if mp3.find("旋律") >= 0:
#         #print(mp3_path + mp3.split('.')[0])
#         str = "F:/ffmpeg/bin/ffmpeg -i "+ mp3_path + mp3 + " " + save_path + mp3.split('mp3')[0] + "wav"
#         print(str)
#         #os.system("ffmpeg -i "+ mp3_path + mp3 + " " + mp3_path + mp3.split('.')[0] + ".wav" )   # ffmpeg 可以用来处理视频和音频文件，可               #以设置一些参数，如采样频率等
#         os.system(str)
