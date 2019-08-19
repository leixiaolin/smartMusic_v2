import librosa.display

from base_helper import *
from create_base import *
from pitch_helper import *


# 1. Get the file path to the included audio example
# Sonify detected beat events
# 定义加载语音文件并去掉两端静音的函数
def load_and_trim(path):
    audio, sr = librosa.load(path)
    energy = librosa.feature.rmse(audio)
    frames = np.nonzero(energy >= np.max(energy) / 5)
    indices = librosa.core.frames_to_samples(frames)[1]
    audio = audio[indices[0]:indices[-1]] if indices.size else audio[0:0]

    return audio, sr
test_codes = np.array(['[1000,1000;2000;1000,500,500;2000]',
                       '[2000;1000,1000;500,500,1000;2000]',
                       '[1000,1000;500,500,1000;1000,1000;2000]',
                       '[1000,--(1000);1000,--(1000);500,250,250,1000;--(1000),1000]',
                       '[500;1000,500,1000,500;500,500,500,250,250,500,500;250,250,500,500,1000]',
                       '[1000,--(1000);1000,--(1000);1000,-(500),500;1000,1000]',
                       '[750,250,500,500,500,-(500);500,1000,500,500,-(500);750,250,500,500,500,-(500)]',
                       '[500,1000,500,500,250,250;1000,500,750,250,500;3000]',
                       '[500,500,500;1000,500;500,500,500;1500;500,500,500;1000,500;500;1000;1500]',
                       '[500,500,1000;500,500;1000;375,125,250,250,375,125,250,250;500,500,1000]'])
test_note_codes = np.array(['[3,3,3,3,3,3,3,5,1,2,3]',
                            '[5,5,3,2,1,2,5,3,2]',
                            '[5,5,3,2,1,2,2,3,2,6-,5-]',
                            '[5,1+,7,1+,2+,1+,7,6,5,2,4,3,6,5]',
                            '[3,6,7,1+,2+,1+,7,6,3]',
                            '[1+,7,1+,2+,3+,2+,1+,7,6,7,1+,2+,7,1+,7,1+,2+,1+]',
                            '[5,6,1+,6,2,3,1,6-,5-]',
                            '[5,5,6,5,6,5,1,3,0,2,2,5-,2,1]',
                            '[3,2,1,2,1,1,2,3,4,5,3,6,5,5,3]',
                            '[3,4,5,1+,7,6,5]'])
test_rhythm_codes = np.array(['[500,500,1000;500,500,1000;500,500,750,250;2000]',
                              '[1000,1000;500,500,1000;1000,500,500; 2000]',
                              '[1000,1000;500,500,1000;500,250,250,500,500;2000]',
                              '[500,1000,500;250,250,250,250,500,500;500,500,500,500;2000]',
                              '[1000;500,500,1000;500,500,500,500;2000]',
                              '[500;500,500,500,500;500,500,500,500;500,500,500,500;250,250,250,250,500]',
                              '[1000,750,250,2000;500,500,500,500,2000]',
                              '[1000,1000,1000,500,500;1000,1000,1000,--(1000);1000,1000,1000;1000,4000]',
                              '[1500,500,500,500;2500,500;1000,500,500,500,500;2500,500]',
                              '[500,500;1500,500,500,500;2000]'])


def get_code(index, type):
    if type == 1:
        code = test_codes[index]
    if type == 2:
        code = test_rhythm_codes[index]
    if type == 3:
        code = test_note_codes[index]
    # code = code.replace(";", ',')
    # code = code.replace("[", '')
    # code = code.replace("]", '')
    # code = [x for x in code.split(',')]
    return code


def get_onsets_index_by_filename(filename):
    if filename.find("节奏10") >= 0 or filename.find("节奏十") >= 0 or filename.find("节奏题十") >= 0 or filename.find(
            "节奏题10") >= 0 or filename.find("节10") >= 0:
        return 9
    elif filename.find("节奏1") >= 0 or filename.find("节奏一") >= 0 or filename.find("节奏题一") >= 0 or filename.find(
            "节奏题1") >= 0 or filename.find("节1") >= 0:
        return 0
    elif filename.find("节奏2") >= 0 or filename.find("节奏二") >= 0 or filename.find("节奏题二") >= 0 or filename.find(
            "节奏题2") >= 0 or filename.find("节2") >= 0:
        return 1
    elif filename.find("节奏3") >= 0 or filename.find("节奏三") >= 0 or filename.find("节奏题三") >= 0 or filename.find(
            "节奏题3") >= 0 or filename.find("节3") >= 0:
        return 2
    elif filename.find("节奏4") >= 0 or filename.find("节奏四") >= 0 or filename.find("节奏题四") >= 0 or filename.find(
            "节奏题4") >= 0 or filename.find("节4") >= 0:
        return 3
    elif filename.find("节奏5") >= 0 or filename.find("节奏五") >= 0 or filename.find("节奏题五") >= 0 or filename.find(
            "节奏题5") >= 0 or filename.find("节5") >= 0:
        return 4
    elif filename.find("节奏6") >= 0 or filename.find("节奏六") >= 0 or filename.find("节奏题六") >= 0 or filename.find(
            "节奏题6") >= 0 or filename.find("节6") >= 0:
        return 5
    elif filename.find("节奏7") >= 0 or filename.find("节奏七") >= 0 or filename.find("节奏题七") >= 0 or filename.find(
            "节奏题7") >= 0 or filename.find("节7") >= 0:
        return 6
    elif filename.find("节奏8") >= 0 or filename.find("节奏八") >= 0 or filename.find("节奏题八") >= 0 or filename.find(
            "节奏题8") >= 0 or filename.find("节8") >= 0:
        return 7
    elif filename.find("节奏9") >= 0 or filename.find("节奏九") >= 0 or filename.find("节奏题九") >= 0 or filename.find(
            "节奏题9") >= 0 or filename.find("节9") >= 0:
        return 8
    else:
        return -1


def get_onsets_index_by_filename_rhythm(filename):
    if filename.find("旋律10") >= 0 or filename.find("旋律十") >= 0 or filename.find("视唱十") >= 0 or filename.find(
            "视唱10") >= 0 or filename.find("旋10") >= 0:
        return 9
    elif filename.find("旋律1") >= 0 or filename.find("旋律一") >= 0 or filename.find("视唱一") >= 0 or filename.find(
            "视唱1") >= 0 or filename.find("旋1") >= 0:
        return 0
    elif filename.find("旋律2") >= 0 or filename.find("旋律二") >= 0 or filename.find("视唱二") >= 0 or filename.find(
            "视唱2") >= 0 or filename.find("旋2") >= 0:
        return 1
    elif filename.find("旋律3") >= 0 or filename.find("旋律三") >= 0 or filename.find("视唱三") >= 0 or filename.find(
            "视唱3") >= 0 or filename.find("旋3") >= 0:
        return 2
    elif filename.find("旋律4") >= 0 or filename.find("旋律四") >= 0 or filename.find("视唱四") >= 0 or filename.find(
            "视唱4") >= 0 or filename.find("旋4") >= 0:
        return 3
    elif filename.find("旋律5") >= 0 or filename.find("旋律五") >= 0 or filename.find("视唱五") >= 0 or filename.find(
            "视唱5") >= 0 or filename.find("旋5") >= 0:
        return 4
    elif filename.find("旋律6") >= 0 or filename.find("旋律六") >= 0 or filename.find("视唱六") >= 0 or filename.find(
            "视唱6") >= 0 or filename.find("旋6") >= 0:
        return 5
    elif filename.find("旋律7") >= 0 or filename.find("旋律七") >= 0 or filename.find("视唱七") >= 0 or filename.find(
            "视唱7") >= 0 or filename.find("旋7") >= 0:
        return 6
    elif filename.find("旋律8") >= 0 or filename.find("旋律八") >= 0 or filename.find("视唱八") >= 0 or filename.find(
            "视唱8") >= 0 or filename.find("旋8") >= 0:
        return 7
    elif filename.find("旋律9") >= 0 or filename.find("旋律九") >= 0 or filename.find("视唱九") >= 0 or filename.find(
            "视唱9") >= 0 or filename.find("旋9") >= 0:
        return 8
    else:
        return -1
def init_test_data():
    files, rhythm_codes, pitch_codes = [], [], []

    filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/6.24MP3/旋律/两只老虎20190624-1089.wav', '[500,500,500,500;500,500,500,500;500,500,1000;500,500;1000]', '[1,2,3,1,1,2,3,1,3,4,5,3,4,5]'  # 音准节奏均正确，给分偏低  94      =======================
    files.append(filename)
    rhythm_codes.append(rhythm_code)
    pitch_codes.append(pitch_code)
    filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/6.24MP3/旋律/两只老虎20190624-1328.wav', '[500,500,500,500;500,500,500,500;500,500,1000;500,500;1000]', '[1,2,3,1,1,2,3,1,3,4,5,3,4,5]'  # 音准节奏均正确，给分偏低  95
    files.append(filename)
    rhythm_codes.append(rhythm_code)
    pitch_codes.append(pitch_code)
    filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/6.24MP3/旋律/两只老虎20190624-1586.wav', '[500,500,500,500;500,500,500,500;500,500,1000;500,500;1000]', '[1,2,3,1,1,2,3,1,3,4,5,3,4,5]'  # 100  95
    files.append(filename)
    rhythm_codes.append(rhythm_code)
    pitch_codes.append(pitch_code)
    filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/6.24MP3/旋律/两只老虎20190624-2939.wav', '[500,500,500,500;500,500,500,500;500,500,1000;500,500;1000]', '[1,2,3,1,1,2,3,1,3,4,5,3,4,5]'  # 音准节奏均正确，给分偏低 90
    files.append(filename)
    rhythm_codes.append(rhythm_code)
    pitch_codes.append(pitch_code)
    filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/6.24MP3/旋律/两只老虎20190624-7881.wav', '[500,500,500,500;500,500,500,500;500,500,1000;500,500;1000]', '[1,2,3,1,1,2,3,1,3,4,5,3,4,5]'  # 音准节奏均正确，给分偏低 90
    files.append(filename)
    rhythm_codes.append(rhythm_code)
    pitch_codes.append(pitch_code)
    filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/6.24MP3/旋律/两只老虎20190624-8973.wav', '[500,500,500,500;500,500,500,500;500,500,1000;500,500;1000]', '[1,2,3,1,1,2,3,1,3,4,5,3,4,5]'  # 100  ========
    files.append(filename)
    rhythm_codes.append(rhythm_code)
    pitch_codes.append(pitch_code)
    filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/6.24MP3/旋律/小学8题20190624-3898-6.wav', '[1000,500,500;2000;250,250,500,500,500;2000]', '[6,5,3,6,3,5,3,2,1,6-]'  # 故意把最后一个音唱错了，节奏全对,扣0.5左右即可 85
    files.append(filename)
    rhythm_codes.append(rhythm_code)
    pitch_codes.append(pitch_code)
    filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/6.24MP3/旋律/小学8题20190624-3898-7.wav', '[2000;250,250,250,250,1000;2000;500,500,1000]', '[6,5,6,3,5,6,3,2,1,6-]'  # 这一条故意唱错了两个音，节奏是对的，这个扣一分即可 72
    files.append(filename)
    rhythm_codes.append(rhythm_code)
    pitch_codes.append(pitch_code)
    filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/6.24MP3/旋律/小学8题20190624-3898-8.wav', '[1000,250,250,250,250;2000;1000,500,500;2000]', '[1,3,5,1+,6,5,1,3,2,1]'  # 这一条节奏不是太稳，但音高基本正确,9.5分是没问题的 93
    files.append(filename)
    rhythm_codes.append(rhythm_code)
    pitch_codes.append(pitch_code)
    filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/7.12MP3/旋律/小学8题20190702-2647-5.wav', '[1000,1000;500,250,250,1000;1000,500,500;2000]', '[3,1,5,5,6,5,1+,6,3,5]'  # 准确 96
    files.append(filename)
    rhythm_codes.append(rhythm_code)
    pitch_codes.append(pitch_code)
    filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/7.12MP3/旋律/小学8题20190702-2647-6.wav', '[1000,500,500;2000;250,250,500,500,500;2000]', '[6,5,3,6,3,5,3,2,1,6]'  # 基本可给满分 59
    files.append(filename)
    rhythm_codes.append(rhythm_code)
    pitch_codes.append(pitch_code)
    filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/7.12MP3/旋律/小学8题20190702-2647-7.wav', '[2000;250,250,250,250,1000;2000;500,500,1000]', '[6,5,6,3,5,6,3,2,1,6]'  # 第一个节奏应该扣分，最后一个音没唱，应该没分  78
    files.append(filename)
    rhythm_codes.append(rhythm_code)
    pitch_codes.append(pitch_code)
    filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/7.12MP3/旋律/小学8题20190702-2647-8.wav', '[1000,250,250,250,250;2000;1000,500,500;2000]', '[1,3,5,1+,6,5,1,3,2,1]'  # 100       =======================
    files.append(filename)
    rhythm_codes.append(rhythm_code)
    pitch_codes.append(pitch_code)
    filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/7.17MP3/旋律/小学8题20190717-2776-5.wav', '[1000,1000;500,250,250,1000;1000,500,500;2000]', '[3,1,5,5,6,5,1+,6,3,5]'  # 100  ===========
    files.append(filename)
    rhythm_codes.append(rhythm_code)
    pitch_codes.append(pitch_code)
    filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/7.17MP3/旋律/小学8题20190717-2776-6.wav', '[1000,500,500;2000;250,250,500,500,500;2000]', '[6,5,3,6,3,5,3,2,1,6-]'  #
    files.append(filename)
    rhythm_codes.append(rhythm_code)
    pitch_codes.append(pitch_code)
    filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/7.17MP3/旋律/小学8题20190717-2776-7.wav', '[2000;250,250,250,250,1000;2000;500,500,1000]', '[6,5,6,3,5,6,3,2,1,6-]'  #
    files.append(filename)
    rhythm_codes.append(rhythm_code)
    pitch_codes.append(pitch_code)
    filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/7.17MP3/旋律/小学8题20190717-2776-8.wav', '[1000,250,250,250,250;2000;1000,500,500;2000]', '[1,3,5,1+,6,5,1,3,2,1]'  #
    files.append(filename)
    rhythm_codes.append(rhythm_code)
    pitch_codes.append(pitch_code)
    filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/7.17MP3/旋律/小学8题20190717-5668-5.wav', '[1000,1000;500,250,250,1000;1000,500,500;2000]', '[3,1,5,5,6,5,1+,6,3,5]'  #
    files.append(filename)
    rhythm_codes.append(rhythm_code)
    pitch_codes.append(pitch_code)
    filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/7.17MP3/旋律/小学8题20190717-5668-6.wav', '[1000,500,500;2000;250,250,500,500,500;2000]', '[6,5,3,6,3,5,3,2,1,6-]'  #
    files.append(filename)
    rhythm_codes.append(rhythm_code)
    pitch_codes.append(pitch_code)
    filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/7.17MP3/旋律/小学8题20190717-5668-7.wav', '[2000;250,250,250,250,1000;2000;500,500,1000]', '[6,5,6,3,5,6,3,2,1,6-]'  #
    files.append(filename)
    rhythm_codes.append(rhythm_code)
    pitch_codes.append(pitch_code)
    filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/7.17MP3/旋律/小学8题20190717-5668-8.wav', '[1000,250,250,250,250;2000;1000,500,500;2000]', '[1,3,5,1+,6,5,1,3,2,1]'  # 100      =======================
    files.append(filename)
    rhythm_codes.append(rhythm_code)
    pitch_codes.append(pitch_code)
    filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/7.17MP3/旋律/小学8题20190717-6249-5.wav', '[1000,1000;500,250,250,1000;1000,500,500;2000]', '[3,1,5,5,6,5,1+,6,3,5]'  #
    files.append(filename)
    rhythm_codes.append(rhythm_code)
    pitch_codes.append(pitch_code)
    filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/7.17MP3/旋律/小学8题20190717-6249-6.wav', '[1000,500,500;2000;250,250,500,500,500;2000]', '[6,5,3,6,3,5,3,2,1,6-]'  #
    files.append(filename)
    rhythm_codes.append(rhythm_code)
    pitch_codes.append(pitch_code)
    filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/7.17MP3/旋律/小学8题20190717-6249-7.wav', '[2000;250,250,250,250,1000;2000;500,500,1000]', '[6,5,6,3,5,6,3,2,1,6-]'  #
    files.append(filename)
    rhythm_codes.append(rhythm_code)
    pitch_codes.append(pitch_code)
    filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/7.17MP3/旋律/小学8题20190717-6249-8.wav', '[1000,250,250,250,250;2000;1000,500,500;2000]', '[1,3,5,1+,6,5,1,3,2,1]'  #
    files.append(filename)
    rhythm_codes.append(rhythm_code)
    pitch_codes.append(pitch_code)
    filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/7.17MP3/旋律/小学8题20190717-6285-5.wav', '[1000,1000;500,250,250,1000;1000,500,500;2000]', '[3,1,5,5,6,5,1+,6,3,5]'  #
    files.append(filename)
    rhythm_codes.append(rhythm_code)
    pitch_codes.append(pitch_code)
    filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/7.17MP3/旋律/小学8题20190717-6285-6.wav', '[1000,500,500;2000;250,250,500,500,500;2000]', '[6,5,3,6,3,5,3,2,1,6-]'  # 62  =======*******============?????????
    files.append(filename)
    rhythm_codes.append(rhythm_code)
    pitch_codes.append(pitch_code)
    filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/7.17MP3/旋律/小学8题20190717-6285-7.wav', '[2000;250,250,250,250,1000;2000;500,500,1000]', '[6,5,6,3,5,6,3,2,1,6]'  #
    files.append(filename)
    rhythm_codes.append(rhythm_code)
    pitch_codes.append(pitch_code)
    filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/7.17MP3/旋律/小学8题20190717-6285-8.wav', '[1000,250,250,250,250;2000;1000,500,500;2000]', '[1,3,5,1+,6,5,1,3,2,1]'  #
    files.append(filename)
    rhythm_codes.append(rhythm_code)
    pitch_codes.append(pitch_code)
    filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/7.12MP3/旋律/小学8题20190712-4290-5.wav', '[1000,1000;500,250,250,1000;1000,500,500;2000]', '[3,1,5,5,6,5,1+,6,3,5]'  #
    files.append(filename)
    rhythm_codes.append(rhythm_code)
    pitch_codes.append(pitch_code)
    filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/7.12MP3/旋律/小学8题20190712-4290-6.wav', '[1000,500,500;2000;250,250,500,500,500;2000]', '[6,5,3,6,3,5,3,2,1,6]'  #
    files.append(filename)
    rhythm_codes.append(rhythm_code)
    pitch_codes.append(pitch_code)
    filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/7.12MP3/旋律/小学8题20190712-4290-7.wav', '[2000;250,250,250,250,1000;2000;500,500,1000]', '[6,5,6,3,5,6,3,2,1,6]'  #
    files.append(filename)
    rhythm_codes.append(rhythm_code)
    pitch_codes.append(pitch_code)
    filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/7.12MP3/旋律/小学8题20190712-4290-8.wav', '[1000,250,250,250,250;2000;1000,500,500;2000]', '[1,3,5,1+,6,5,1,3,2,1]'  # 100   ???????????==================
    files.append(filename)
    rhythm_codes.append(rhythm_code)
    pitch_codes.append(pitch_code)
    filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋3罗（80）.wav', '[1000,1000;500,500,1000;500,250,250,500,500;2000]', '[5,5,3,2,1,2,2,3,2,6-,5-]'
    files.append(filename)
    rhythm_codes.append(rhythm_code)
    pitch_codes.append(pitch_code)
    filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/6.18MP3/旋律/01，98.wav', '[500,250,250,500,500;250,250,250,250,500,500;500,250,250,500,500;500,250,250,1000]', '[5,5,6,5,3,4,5,4,5,4,2,3,3,4,3,1,2,3,5,1]'
    files.append(filename)
    rhythm_codes.append(rhythm_code)
    pitch_codes.append(pitch_code)
    filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/7.17MP3/旋律/test.wav', '[1000,250,250,250,250;2000;1000,500,500;2000]', '[1,3,5,1+,6,5,1,3,2,1]'  #
    files.append(filename)
    rhythm_codes.append(rhythm_code)
    pitch_codes.append(pitch_code)

    filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/8.05MP3/旋律/小学8题20190805-9112-5.wav', '[1000,1000;500,250,250,1000;1000,500,500;2000]', '[3,1,5,5,6,5,1+,6,3,5]'  # 100
    files.append(filename)
    rhythm_codes.append(rhythm_code)
    pitch_codes.append(pitch_code)
    filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/8.05MP3/旋律/小学8题20190805-9112-6.wav', '[1000,500,500;2000;250,250,500,500,500;2000]', '[6,5,3,6,3,5,3,2,1,6]'  # 84
    files.append(filename)
    rhythm_codes.append(rhythm_code)
    pitch_codes.append(pitch_code)
    filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/8.05MP3/旋律/小学8题20190805-9112-7.wav', '[2000;250,250,250,250,1000;2000;500,500,1000]', '[6,5,6,3,5,6,3,2,1,6]'  # 90
    files.append(filename)
    rhythm_codes.append(rhythm_code)
    pitch_codes.append(pitch_code)
    filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/8.05MP3/旋律/小学8题20190805-9112-8.wav', '[1000,250,250,250,250;2000;1000,500,500;2000]', '[1,3,5,1+,6,5,1,3,2,1]'  # 100
    files.append(filename)
    rhythm_codes.append(rhythm_code)
    pitch_codes.append(pitch_code)
    filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/8.05MP3/旋律/中学8题20190805-6858-5.wav', '[500,250,250,500,500;1500,500;1000,1000;2000]', '[3,3,1,3,4,5,5,6,7,1+]'  # 92
    files.append(filename)
    rhythm_codes.append(rhythm_code)
    pitch_codes.append(pitch_code)
    filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/8.05MP3/旋律/中学8题20190805-6858-6.wav', '[1000,1000;1500,500;500,250,250,500,500;2000]', '[1+,7,6,5,4,5,4,3,2,1]'  # 100
    files.append(filename)
    rhythm_codes.append(rhythm_code)
    pitch_codes.append(pitch_code)
    filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/8.05MP3/旋律/中学8题20190805-6858-7.wav', '[500,1000,500;2000;500,250,250,500,500;2000]', '[1,3,4,5,6,6,1+,7,6,1+]'  # 88
    files.append(filename)
    rhythm_codes.append(rhythm_code)
    pitch_codes.append(pitch_code)
    filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/8.05MP3/旋律/中学8题20190805-6858-8.wav', '[500,1000,500;2000;500,500,500,250,250;2000]', '[1+,7,6,5,6,5,4,3,2,1]'  # 94
    files.append(filename)
    rhythm_codes.append(rhythm_code)
    pitch_codes.append(pitch_code)

    filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/8.12MP3/旋律/小学8题20190809-3492-5.wav', '[1000,1000;500,250,250,1000;1000,500,500;2000]', '[3,1,5,5,6,5,1+,6,3,5]'  #
    files.append(filename)
    rhythm_codes.append(rhythm_code)
    pitch_codes.append(pitch_code)
    filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/8.12MP3/旋律/小学8题20190809-3492-6.wav', '[1000,500,500;2000;250,250,500,500,500;2000]', '[6,5,3,6,3,5,3,2,1,6]'  #
    files.append(filename)
    rhythm_codes.append(rhythm_code)
    pitch_codes.append(pitch_code)
    filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/8.12MP3/旋律/小学8题20190809-3492-7.wav', '[2000;250,250,250,250,1000;2000;500,500,1000]', '[6,5,6,3,5,6,3,2,1,6]'  #
    files.append(filename)
    rhythm_codes.append(rhythm_code)
    pitch_codes.append(pitch_code)
    filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/8.12MP3/旋律/小学8题20190809-3492-8.wav', '[1000,250,250,250,250;2000;1000,500,500;2000]', '[1,3,5,1+,6,5,1,3,2,1]'  #
    files.append(filename)
    rhythm_codes.append(rhythm_code)
    pitch_codes.append(pitch_code)

    filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/8.12MP3/旋律/小学8题20190812-6117-5.wav', '[1000,1000;500,250,250,1000;1000,500,500;2000]', '[3,1,5,5,6,5,1+,6,3,5]'  #
    files.append(filename)
    rhythm_codes.append(rhythm_code)
    pitch_codes.append(pitch_code)
    filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/8.12MP3/旋律/小学8题20190812-6117-6.wav', '[1000,500,500;2000;250,250,500,500,500;2000]', '[6,5,3,6,3,5,3,2,1,6]'  #
    files.append(filename)
    rhythm_codes.append(rhythm_code)
    pitch_codes.append(pitch_code)
    filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/8.12MP3/旋律/小学8题20190812-6117-7.wav', '[2000;250,250,250,250,1000;2000;500,500,1000]', '[6,5,6,3,5,6,3,2,1,6]'  #
    files.append(filename)
    rhythm_codes.append(rhythm_code)
    pitch_codes.append(pitch_code)
    filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/8.12MP3/旋律/小学8题20190812-6117-8.wav', '[1000,250,250,250,250;2000;1000,500,500;2000]', '[1,3,5,1+,6,5,1,3,2,1]'  #
    files.append(filename)
    rhythm_codes.append(rhythm_code)
    pitch_codes.append(pitch_code)
    return files, rhythm_codes, pitch_codes
def test_batch_samples():
    files, rhythm_codes, pitch_codes = init_test_data()

    total_length = 0

    for filename, rhythm_code, pitch_code in zip(files,rhythm_codes,pitch_codes):
        print(filename)

        # total_score,all_starts, detail = calcalate_total_score(filename, rhythm_code,pitch_code)
        # print("total_score is {}".format(total_score))
        # print("detail is {}".format(detail))
        total_score, all_starts, detail = calcalate_total_score(filename, rhythm_code, pitch_code)
        print("总分 is {}".format(total_score))
        print("detail is {}".format(detail))
        plt = draw_detail(filename, rhythm_code, pitch_code)
        # plt.show()
        if total_score < 80:
            result_path = 'e:/test_image/n/'
        else:
            result_path = 'e:/test_image/n2/'
            #plt.savefig(result_path  + filename.split('/')[-1].split('.wav')[0] + '-' + str(total_score) + '-' + str(lost_score) + '-' + str(ex_score) + '-' + str(min_d) + '.jpg', bbox_inches='tight', pad_inches=0)
        plt.savefig(result_path  + filename.split('/')[-1].split('.wav')[0] + '-' + str(total_score) + '分.jpg', bbox_inches='tight', pad_inches=0)
        plt.clf()

        length = get_total_length(filename)
        # print(length)
        total_length = total_length + length
    file_total = len(files)
    each_length = total_length / file_total
    # print("each_length is {}".format(each_length))

def batch_draw_samples():
    files, rhythm_codes, pitch_codes = init_test_data()

    for filename, rhythm_code, pitch_code in zip(files,rhythm_codes,pitch_codes):
        print(filename)

        plt = draw_detail(filename, rhythm_code, pitch_code)
        plt.savefig(result_path  + filename.split('/')[-1].split('.wav')[0] + '.jpg', bbox_inches='tight', pad_inches=0)
        plt.clf()



def test_total_length(dir_list):
    for dir in dir_list:
        file_list = os.listdir(dir)
        # shuffle(file_list)  # 将语音文件随机排列
        #file_list = ['旋1王（98）.wav']
        file_total = len(file_list)
        total_length = 0
        for filename in file_list:
            if filename.find(".wav") > 0:
                print(dir + filename)
                tmp = get_total_length(dir + filename)
                total_length = total_length + tmp
        each_length = total_length/file_total
        print("each_length is {}".format(each_length))
result_path = 'e:/test_image/n/'
#y, sr = load_and_trim('F:/项目/花城音乐项目/样式数据/ALL/旋律/1.31MP3/旋律1.100分.wav')
filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律2.1(80).wav'
filename = 'F:/项目/花城音乐项目/样式数据/ALL/旋律/1.31MP3/旋律3.100分.wav'
filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律一（9）（100）.wav'
#filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律一（14）（95）.wav'
#filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/节奏/节2罗（75）.wav'
#filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏一（4）（96）.wav'
filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋2熙(0).wav'
filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋1罗（96）.wav'
#filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋10罗（92）.wav'
#filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋2录音4(72).wav'
#filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋10罗（92）.wav'
#filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律一（13）（98）.wav'
#filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋1熙(90).wav'
#filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋1谭（98）.wav'
#filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋3.3(96).wav'
# filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋1王（98）.wav'
# filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋9.1(73).wav'
type_index = get_onsets_index_by_filename_rhythm(filename)
# onset_code = get_code(type_index, 1)
rhythm_code = get_code(type_index, 2)
pitch_code = get_code(type_index, 3)
# filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/6.24MP3/旋律/两只老虎20190624-1089.wav', '[500,500,500,500;500,500,500,500;500,500,1000;500,500;1000]', '[1,2,3,1,1,2,3,1,3,4,5,3,4,5]'  # =======================
filename,rhythm_code,pitch_code = 'F:/项目/花城音乐项目/样式数据/6.24MP3/旋律/两只老虎20190624-1328.wav','[500,500,500,500;500,500,500,500;500,500,1000;500,500;1000]','[1,2,3,1,1,2,3,1,3,4,5,3,4,5]'       #
# filename,rhythm_code,pitch_code = 'F:/项目/花城音乐项目/样式数据/6.24MP3/旋律/两只老虎20190624-1586.wav','[500,500,500,500;500,500,500,500;500,500,1000;500,500;1000]','[1,2,3,1,1,2,3,1,3,4,5,3,4,5]'      #
filename,rhythm_code,pitch_code = 'F:/项目/花城音乐项目/样式数据/6.24MP3/旋律/两只老虎20190624-2939.wav','[500,500,500,500;500,500,500,500;500,500,1000;500,500;1000]','[1,2,3,1,1,2,3,1,3,4,5,3,4,5]'      #81
filename,rhythm_code,pitch_code = 'F:/项目/花城音乐项目/样式数据/6.24MP3/旋律/两只老虎20190624-7881.wav','[500,500,500,500;500,500,500,500;500,500,1000;500,500;1000]','[1,2,3,1,1,2,3,1,3,4,5,3,4,5]'          #100
filename,rhythm_code,pitch_code = 'F:/项目/花城音乐项目/样式数据/6.24MP3/旋律/两只老虎20190624-8973.wav','[500,500,500,500;500,500,500,500;500,500,1000;500,500;1000]','[1,2,3,1,1,2,3,1,3,4,5,3,4,5]'       #100


filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/6.24MP3/旋律/小学8题20190624-3898-6.wav', '[1000,500,500;2000;250,250,500,500,500;2000]', '[6,5,3,6,3,5,3,2,1,6-]'  #72
filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/6.24MP3/旋律/小学8题20190624-3898-7.wav', '[2000;250,250,250,250,1000;2000;500,500,1000]', '[6,5,6,3,5,6,3,2,1,6-]'  #82
filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/6.24MP3/旋律/小学8题20190624-3898-8.wav', '[1000,250,250,250,250;2000;1000,500,500;2000]', '[1,3,5,1+,6,5,1,3,2,1]'  #100

filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/7.12MP3/旋律/小学8题20190702-2647-5.wav', '[1000,1000;500,250,250,1000;1000,500,500;2000]', '[3,1,5,5,6,5,1+,6,3,5]'  #86
filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/7.12MP3/旋律/小学8题20190702-2647-6.wav', '[1000,500,500;2000;250,250,500,500,500;2000]', '[6,5,3,6,3,5,3,2,1,6]'  # 84
filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/7.12MP3/旋律/小学8题20190702-2647-7.wav', '[2000;250,250,250,250,1000;2000;500,500,1000]', '[6,5,6,3,5,6,3,2,1,6]'  #74
filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/7.12MP3/旋律/小学8题20190702-2647-8.wav', '[1000,250,250,250,250;2000;1000,500,500;2000]', '[1,3,5,1+,6,5,1,3,2,1]'  # 94


filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/7.17MP3/旋律/小学8题20190717-2776-5.wav', '[1000,1000;500,250,250,1000;1000,500,500;2000]', '[3,1,5,5,6,5,1+,6,3,5]'  # 100
filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/7.17MP3/旋律/小学8题20190717-2776-6.wav', '[1000,500,500;2000;250,250,500,500,500;2000]', '[6,5,3,6,3,5,3,2,1,6-]'  #94
filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/7.17MP3/旋律/小学8题20190717-2776-7.wav', '[2000;250,250,250,250,1000;2000;500,500,1000]', '[6,5,6,3,5,6,3,2,1,6-]'  #100
filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/7.17MP3/旋律/小学8题20190717-2776-8.wav', '[1000,250,250,250,250;2000;1000,500,500;2000]', '[1,3,5,1+,6,5,1,3,2,1]'  #84

filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/7.17MP3/旋律/小学8题20190717-5668-5.wav', '[1000,1000;500,250,250,1000;1000,500,500;2000]', '[3,1,5,5,6,5,1+,6,3,5]'  #100
filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/7.17MP3/旋律/小学8题20190717-5668-6.wav', '[1000,500,500;2000;250,250,500,500,500;2000]', '[6,5,3,6,3,5,3,2,1,6-]'  #100
filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/7.17MP3/旋律/小学8题20190717-5668-7.wav', '[2000;250,250,250,250,1000;2000;500,500,1000]', '[6,5,6,3,5,6,3,2,1,6-]'  #100
filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/7.17MP3/旋律/小学8题20190717-5668-8.wav', '[1000,250,250,250,250;2000;1000,500,500;2000]', '[1,3,5,1+,6,5,1,3,2,1]'  # 88

filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/7.17MP3/旋律/小学8题20190717-6249-5.wav', '[1000,1000;500,250,250,1000;1000,500,500;2000]', '[3,1,5,5,6,5,1+,6,3,5]'  #76
filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/7.17MP3/旋律/小学8题20190717-6249-6.wav', '[1000,500,500;2000;250,250,500,500,500;2000]', '[6,5,3,6,3,5,3,2,1,6-]'  #94
filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/7.17MP3/旋律/小学8题20190717-6249-7.wav', '[2000;250,250,250,250,1000;2000;500,500,1000]', '[6,5,6,3,5,6,3,2,1,6-]'  #100
filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/7.17MP3/旋律/小学8题20190717-6249-8.wav', '[1000,250,250,250,250;2000;1000,500,500;2000]', '[1,3,5,1+,6,5,1,3,2,1]'  #50

filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/7.17MP3/旋律/小学8题20190717-6285-5.wav', '[1000,1000;500,250,250,1000;1000,500,500;2000]', '[3,1,5,5,6,5,1+,6,3,5]'  #100
filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/7.17MP3/旋律/小学8题20190717-6285-6.wav', '[1000,500,500;2000;250,250,500,500,500;2000]', '[6,5,3,6,3,5,3,2,1,6-]'  #  90
filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/7.17MP3/旋律/小学8题20190717-6285-7.wav', '[2000;250,250,250,250,1000;2000;500,500,1000]', '[6,5,6,3,5,6,3,2,1,6]'  #90
filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/7.17MP3/旋律/小学8题20190717-6285-8.wav', '[1000,250,250,250,250;2000;1000,500,500;2000]', '[1,3,5,1+,6,5,1,3,2,1]'  #8

filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/7.12MP3/旋律/小学8题20190712-4290-5.wav', '[1000,1000;500,250,250,1000;1000,500,500;2000]', '[3,1,5,5,6,5,1+,6,3,5]'  #100
filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/7.12MP3/旋律/小学8题20190712-4290-6.wav', '[1000,500,500;2000;250,250,500,500,500;2000]', '[6,5,3,6,3,5,3,2,1,6]'  #84
# filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/7.12MP3/旋律/小学8题20190712-4290-7.wav','[2000;250,250,250,250,1000;2000;500,500,1000]', '[6,5,6,3,5,6,3,2,1,6]'  # 88
# filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/7.12MP3/旋律/小学8题20190712-4290-8.wav', '[1000,250,250,250,250;2000;1000,500,500;2000]', '[1,3,5,1+,6,5,1,3,2,1]'  #100

# filename, rhythm_code, pitch_code  = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋3罗（80）.wav', '[1000,1000;500,500,1000;500,250,250,500,500;2000]', '[5,5,3,2,1,2,2,3,2,6-,5-]' # 67 背景噪声造成音高不准
# filename, rhythm_code, pitch_code  = 'F:/项目/花城音乐项目/样式数据/6.18MP3/旋律/01，98.wav', '[500,250,250,500,500;250,250,250,250,500,500;500,250,250,500,500;500,250,250,1000]', '[5,5,6,5,3,4,5,4,5,4,2,3,3,4,3,1,2,3,5,1]'  # 25 背景噪声太大

# filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/7.17MP3/旋律/test.wav','[1000,250,250,250,250;2000;1000,500,500;2000]', '[1,3,5,1+,6,5,1,3,2,1]'  #94
# filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋4.4(0).wav'
# filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋2.3(95).wav'
# filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋2.4(50).wav'
# filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋2录音1(90).wav'
# type_index = get_onsets_index_by_filename_rhythm(filename)
# rhythm_code = get_code(type_index, 2)
# pitch_code = get_code(type_index, 3)

# filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/8.05MP3/旋律/小学8题20190805-9112-5.wav', '[1000,1000;500,250,250,1000;1000,500,500;2000]', '[3,1,5,5,6,5,1+,6,3,5]'  #38?????????????????????????????
# filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/8.05MP3/旋律/小学8题20190805-9112-6.wav', '[1000,500,500;2000;250,250,500,500,500;2000]', '[6,5,3,6,3,5,3,2,1,6]'  #72???????????????????????? 倍频问题
# filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/8.05MP3/旋律/小学8题20190805-9112-7.wav','[2000;250,250,250,250,1000;2000;500,500,1000]', '[6,5,6,3,5,6,3,2,1,6]'  #62????????????????????????
# filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/8.05MP3/旋律/小学8题20190805-9112-8.wav', '[1000,250,250,250,250;2000;1000,500,500;2000]', '[1,3,5,1+,6,5,1,3,2,1]'  #94????????????

# filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/8.05MP3/旋律/中学8题20190805-6858-5.wav', '[500,250,250,500,500;1500,500;1000,1000;2000]', '[3,3,1,3,4,5,5,6,7,1+]'  #100
# filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/8.05MP3/旋律/中学8题20190805-6858-6.wav', '[1000,1000;1500,500;500,250,250,500,500;2000]', '[1+,7,6,5,4,5,4,3,2,1]'  #100
# filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/8.05MP3/旋律/中学8题20190805-6858-7.wav','[500,1000,500;2000;500,250,250,500,500;2000]', '[1,3,4,5,6,6,1+,7,6,1+]'  #100
# filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/8.05MP3/旋律/中学8题20190805-6858-8.wav', '[500,1000,500;2000;500,500,500,250,250;2000]', '[1+,7,6,5,6,5,4,3,2,1]'  #94


# filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/8.12MP3/旋律/小学8题20190809-3492-5.wav', '[1000,1000;500,250,250,1000;1000,500,500;2000]', '[3,1,5,5,6,5,1+,6,3,5]'  #74 静默区起始点的伪节拍引起的
# filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/8.12MP3/旋律/小学8题20190809-3492-6.wav', '[1000,500,500;2000;250,250,500,500,500;2000]', '[6,5,3,6,3,5,3,2,1,6]'  # 84
# filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/8.12MP3/旋律/小学8题20190809-3492-7.wav','[2000;250,250,250,250,1000;2000;500,500,1000]', '[6,5,6,3,5,6,3,2,1,6]'  # 72
# filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/8.12MP3/旋律/小学8题20190809-3492-8.wav', '[1000,250,250,250,250;2000;1000,500,500;2000]', '[1,3,5,1+,6,5,1,3,2,1]'  #68????????????????????????????? 将1000也判断为静默区

# filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/8.12MP3/旋律/小学8题20190812-6117-5.wav', '[1000,1000;500,250,250,1000;1000,500,500;2000]', '[3,1,5,5,6,5,1+,6,3,5]'  #94
# filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/8.12MP3/旋律/小学8题20190812-6117-6.wav', '[1000,500,500;2000;250,250,500,500,500;2000]', '[6,5,3,6,3,5,3,2,1,6]'  # 90
# filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/8.12MP3/旋律/小学8题20190812-6117-7.wav','[2000;250,250,250,250,1000;2000;500,500,1000]', '[6,5,6,3,5,6,3,2,1,6]'  #88
# filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/8.12MP3/旋律/小学8题20190812-6117-8.wav', '[1000,250,250,250,250;2000;1000,500,500;2000]', '[1,3,5,1+,6,5,1,3,2,1]'  #94

def batch_test(dir_list):
    result_path = 'e:/test_image/n/'
    for dir in dir_list:
        file_list = os.listdir(dir)
        # shuffle(file_list)  # 将语音文件随机排列
        # file_list = ['16，48.wav']
        file_total = len(file_list)
        for filename in file_list:
            if filename.find('wav') <= 0:
                continue
            print(filename)
            type_index = get_onsets_index_by_filename_rhythm(dir + filename)
            rhythm_code = get_code(type_index, 2)
            pitch_code = get_code(type_index, 3)

            plt = draw_plt(dir + filename,rhythm_code, pitch_code)
            #plt.savefig(result_path  + filename.split('/')[-1].split('.wav')[0] + '-' + str(total_score) + '-' + str(lost_score) + '-' + str(ex_score) + '-' + str(min_d) + '.jpg', bbox_inches='tight', pad_inches=0)
            plt.savefig(result_path  + filename.split('/')[-1].split('.wav')[0] + '-' + str(total_score) + '分.jpg', bbox_inches='tight', pad_inches=0)
            plt.clf()
    pass

if __name__ == '__main__':

    # plt.vlines(base_frames_time, 0, 30, color='r', linestyle='solid')


    # threshold = 8
    # all_maxs,all_mins = get_all_max_min_points_on_heigth_cqt(cqt_bak,threshold)
    # all_maxs_time = librosa.frames_to_time(all_maxs)
    # plt.vlines(all_maxs_time, 0, 60, color='b', linestyle='dashed')
    # onset_types,all_starts = get_onset_from_heights_v2(cqt_bak,rhythm_code)
    # print("onset_types is {},size {}".format(onset_types,len(onset_types)))
    # all_symbols = get_all_symbols(onset_types)
    # #print(all_symbols)
    # code = parse_rhythm_code(rhythm_code)
    # code = [int(x) for x in code]
    # print("code  is {} ,size {}".format(code, len(code)))
    # base_symbols = get_all_symbols(code)
    # threshold_score = 40
    # onset_score,detail = calculate_onset_score_from_symbols(base_symbols, all_symbols,threshold_score)
    # print("onset_score is {}".format(onset_score))
    # print("detail is {}".format(detail))
    # note_score,detail = calculate_note_score(cqt_bak,rhythm_code,pitch_code,60)
    # print("note_score is {}".format(note_score))
    # print("detail is {}".format(detail))
    print(filename)
    # total_score,all_starts,detail = calcalate_total_score(filename, rhythm_code,pitch_code)
    # print("总分 is {}".format(total_score))
    # print("detail is {}".format(detail))
    length = get_total_length(filename)
    # print(length)

    # my_plt,total_score = draw_plt(filename,rhythm_code, pitch_code)
    # plt.show()
    total_score, all_starts, detail = calcalate_total_score(filename, rhythm_code, pitch_code)
    print("总分 is {}".format(total_score))
    print("detail is {}".format(detail))
    detail = draw_detail(filename, rhythm_code, pitch_code)
    detail.show()

    dir_list = ['F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/','F:/项目/花城音乐项目/样式数据/8.12MP3/旋律/','F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/','F:/项目/花城音乐项目/样式数据/3.19MP3/旋律/','F:/项目/花城音乐项目/样式数据/6.18MP3/旋律/']
    # batch_test(dir_list)

    test_batch_samples()

    # test_total_length(dir_list)

    # batch_draw_samples();