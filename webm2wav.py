import os

webm_path = "F:/项目/花城音乐项目/样式数据/20.05.12MP3/"
wav_path = "F:/项目/花城音乐项目/样式数据/20.05.12MP3/wav/"
webm_file = os.listdir(webm_path)
webm_file = [name for name in webm_file if name.endswith(".webm")]

for i, m4a in enumerate(webm_file):
    #print(m4a_path + m4a.split('.')[0])
    if m4a.find(".webm")>=0:
        m4a = m4a.replace(" ","-")
        str = "F:/ffmpeg/bin/ffmpeg -i "+ webm_path + m4a + " " + wav_path + m4a.split('.webm')[0] + ".wav"
    print(str)
    #os.system("ffmpeg -i "+ m4a_path + m4a + " " + m4a_path + m4a.split('.')[0] + ".wav" )   # ffmpeg 可以用来处理视频和音频文件，可               #以设置一些参数，如采样频率等
    os.system(str)