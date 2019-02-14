import os

m4a_path = "F:\\项目\\花城音乐项目\\样式数据\\节奏样本\\节奏一\\"

m4a_file = os.listdir(m4a_path)

for i, m4a in enumerate(m4a_file):
    #print(m4a_path + m4a.split('.')[0])
    str = "F:/ffmpeg/bin/ffmpeg -i "+ m4a_path + m4a + " " + m4a_path + m4a.split('.')[0] + ".wav"
    print(str)
    #os.system("ffmpeg -i "+ m4a_path + m4a + " " + m4a_path + m4a.split('.')[0] + ".wav" )   # ffmpeg 可以用来处理视频和音频文件，可               #以设置一些参数，如采样频率等
    os.system(str)