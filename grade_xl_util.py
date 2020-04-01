# -*- coding: UTF-8 -*-
import argparse
import sys
from pitch_helper import *
import warnings
import time
warnings.simplefilter('ignore')

def write_txt(content, filename, mode='w'):
    """保存txt数据
    :param content:需要保存的数据,type->list
    :param filename:文件名
    :param mode:读写模式:'w' or 'a'
    :return: void
    """
    with open(filename, mode) as f:
        f.write(content)

parser = argparse.ArgumentParser(description='Process input the file path and file code.')   # 首先创建一个ArgumentParser对象
parser.add_argument("file_path", help="音频文件的路径",type=str)
parser.add_argument("rhythm_code", help="音频文件的节奏编码",type=str)
parser.add_argument("melody_code", help="音频文件的音高编码",type=str)

args = parser.parse_args()
#print(len(sys.argv))
if len(sys.argv) != 4:
     parser.print_help()
     sys.exit(1)
else:
    file_path = args.file_path
    rhythm_code = args.rhythm_code
    melody_code = args.melody_code
    print("file_path is {}".format(file_path))
    print("rhythm_code is {}".format(rhythm_code))
    print("melody_code is {}".format(melody_code))
    # file_path = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋3罗（80）.wav'
    # rhythm_code = '[1000,1000;500,500,1000;500,250,250,500,500;2000]'
    # melody_code = '[5,5,3,2,1,2,2,3,2,6-,5-]'
    # total_score,  onsets_frames,detail_content = calcalate_total_score(file_path,rhythm_code,melody_code)
    start = time.clock()
    total_score, onsets_frames, detail_content,total_score_absolute_pitch,detail_absolute_pitch,change_points = calcalate_total_score_by_alexnet(file_path, rhythm_code, melody_code)
    print("time used is {}".format(time.clock() - start))
    print("total_score, is {}".format(total_score))
    filepath, fullflname = os.path.split(file_path)
    output_file = fullflname.split('.wav')[0] + '-out.txt'
    content = 'total_score is ' + str(total_score)
    content += "\n"
    save_path = os.path.join(filepath,output_file)
    write_txt(content, save_path, mode='w')
    #detail = '整体节奏偏离较大，未能误别部分节拍，流畅平稳性有待加强'
    detail = detail_content
    write_txt(detail, save_path, mode='a')
    # 绝对音高的评分结果
    content = "\n" + 'total_score is ' + str(total_score_absolute_pitch)
    content += "\n"
    write_txt(content, save_path, mode='a')
    detail = detail_absolute_pitch
    write_txt(detail, save_path, mode='a')
    #python grade_xl_util.py F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋3罗（80）.wav [1000,1000;500,500,1000;500,250,250,500,500;2000] [5,5,3,2,1,2,2,3,2,6-,5-]
    # python grade_xl_util.py F:/项目/花城音乐项目/样式数据/6.18MP3/旋律/01，98.wav [500,250,250,500,500;250,250,250,250,500,500;500,250,250,500,500;500,250,250,1000] [5,5,6,5,3,4,5,4,5,4,2,3,3,4,3,1,2,3,5,1]
    # python grade_xl_util.py F:/项目/花城音乐项目/样式数据/7.17MP3/旋律/小学8题20190717-2776-6.wav [1000,500,500;2000;250,250,500,500,500;2000] [6,5,3,6,3,5,3,2,1,6-]
    # python grade_xl_util.py F:/项目/花城音乐项目/样式数据/7.17MP3/旋律/test.wav [1000,250,250,250,250;2000;1000,500,500;2000] [1,3,5,1+,6,5,1,3,2,1]