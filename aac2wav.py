import os

m4a_path = "F:/项目/花城音乐项目/样式数据/20.03.26MP3/aac/"
wav_path = "F:/项目/花城音乐项目/样式数据/20.03.26MP3/wav/"
m4a_file = os.listdir(m4a_path)
m4a_file = [name for name in m4a_file if name.endswith(".aac")]

for i, m4a in enumerate(m4a_file):
    #print(m4a_path + m4a.split('.')[0])
    if m4a.find(".aac")>=0:
        m4a = m4a.replace(" ","-")
        str = "F:/ffmpeg/bin/ffmpeg -i "+ m4a_path + m4a + " " + wav_path + m4a.split('.aac')[0] + ".wav"
    print(str)
    #os.system("ffmpeg -i "+ m4a_path + m4a + " " + m4a_path + m4a.split('.')[0] + ".wav" )   # ffmpeg 可以用来处理视频和音频文件，可               #以设置一些参数，如采样频率等
    os.system(str)