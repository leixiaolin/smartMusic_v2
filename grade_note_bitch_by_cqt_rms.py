# -*- coding: utf-8 -*-

import shutil
import librosa.display
from cqt_rms import *
from create_labels_files import *


def clear_dir(dis_dir):
    shutil.rmtree(dis_dir)
    os.mkdir(dis_dir)


def load_and_trim(path):
    audio, sr = librosa.load(path)
    energy = librosa.feature.rmse(audio)
    frames = np.nonzero(energy >= np.max(energy) / 5)
    indices = librosa.core.frames_to_samples(frames)[1]
    audio = audio[indices[0]:indices[-1]] if indices.size else audio[0:0]

    return audio, sr




if __name__ ==  '__main__':

    # filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏1.3(95).wav'
    # filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏1（二）(100).wav'
    # filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏1_40227（100）.wav'
    # filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏1林(70).wav'
    # filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏2_40314（100）.wav'
    # filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏2_40409（98）.wav'
    # filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏2林(25).wav'
    # filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏2语(85).wav'
    # filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏10_40411（85）.wav'
    # filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏10-04（80）.wav'
    # filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏九（2）（95）.wav'


    savepath = './single_notes/data/test/'



    '''
    测试多个文件
    '''
    dir_list = ['F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/']
    dir_list = ['F:/项目/花城音乐项目/样式数据/3.19MP3/节奏/']
    dir_list = ['F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/']
    #dir_list = []
    total_accuracy = 0
    total_num = 0
    result_path = 'e:/test_image/t/'
    for dir in dir_list:
        file_list = os.listdir(dir)
        #file_list = ['节奏1（二）(100).wav','节奏1（四）(100).wav','节奏1（三）(95).wav']
        for filename in file_list:
            # wavname = re.findall(pattern,filename)[0]
            print(dir + filename)
            y, sr = librosa.load(dir + filename)
            onset_frames_cqt, best_y,best_threshold, plt = get_detail_cqt_rms_secondary_optimised(dir + filename)
            plt.savefig(result_path + filename.split(".wav")[0]+'.jpg', bbox_inches='tight', pad_inches=0)
            plt.clf()


