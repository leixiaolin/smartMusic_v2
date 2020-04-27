# -*- coding: UTF-8 -*-
import argparse
import sys
from parselmouth_util import score_all
import warnings
import time
import json
import os
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
parser.add_argument("standard_kc", help="音频文件的标准歌词",type=str)
parser.add_argument("standard_kc_time", help="音频文件的标准歌词时间点",type=str)
parser.add_argument("standard_notations", help="音频文件的标准音符",type=str)
parser.add_argument("standard_notation_time", help="音频文件的标准音符时间点",type=str)

args = parser.parse_args()
print(len(sys.argv))
if len(sys.argv) != 6:
     parser.print_help()
     sys.exit(1)
else:
    file_path = args.file_path
    standard_kc = args.standard_kc
    standard_kc_time = args.standard_kc_time
    standard_kc_time = list(standard_kc_time)
    standard_notations = args.standard_notations
    standard_notation_time = args.standard_notation_time
    standard_notation_time = list(standard_notation_time)
    # print("file_path is {}".format(file_path))
    # print("rhythm_code is {}".format(rhythm_code))
    # print("melody_code is {}".format(melody_code))
    # file_path = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋3罗（80）.wav'
    # rhythm_code = '[1000,1000;500,500,1000;500,250,250,500,500;2000]'
    # melody_code = '[5,5,3,2,1,2,2,3,2,6-,5-]'
    # total_score,  onsets_frames,detail_content = calcalate_total_score(file_path,rhythm_code,melody_code)
    start = time.clock()
    total_score, pitch_total_score, notation_duration_total_score, kc_duration_total_score, pitch_score_detail, notation_duration_score_detail, kc_rhythm_sscore_detail = score_all(
        file_path, standard_kc, standard_kc_time, standard_notations, standard_notation_time)
    print("total_score is {}".format(total_score))
    score_detail = "音高评分结果为{}，{}，音符节奏评分结果为{}，{}，歌词节奏评分结果为{}，{}".format(pitch_total_score,
                                                                       pitch_score_detail,
                                                                       notation_duration_total_score,
                                                                       notation_duration_score_detail,
                                                                       kc_duration_total_score,
                                                                       kc_rhythm_sscore_detail)
    # print("time used is {}".format(time.clock() - start))
    # print("total_score, is {}".format(total_score))
    out_result = {'file_path': file_path, 'rhythm_code': total_score, 'melody_code': total_score,
                  'time_used': (time.clock() - start), 'detail': score_detail}
    x = json.dumps(out_result)
    print(json.dumps(out_result,ensure_ascii=False))
    filepath, fullflname = os.path.split(file_path)
    output_file = fullflname.split('.wav')[0] + '-out.txt'
    content = 'total_score is ' + str(total_score)
    content += "\n"
    save_path = os.path.join(filepath,output_file)
    write_txt(content, save_path, mode='w')
    #detail = '整体节奏偏离较大，未能误别部分节拍，流畅平稳性有待加强'
    detail = score_detail
    write_txt(detail, save_path, mode='a')

    #python grade_triple_item_util.py F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋3罗（80）.wav [1000,1000;500,500,1000;500,250,250,500,500;2000] [5,5,3,2,1,2,2,3,2,6-,5-]
    # python grade_triple_item_util.py F:/项目/花城音乐项目/样式数据/20.04.08MP3/2段词-不标准1648.wav '喜爱春天的人儿是心地纯洁的人像紫罗兰花儿一样是我知心朋友' '0,1,2,3,3.5,4,5,6,8,9,10,11,11.5,12,16,17,18,19,20,21,22,23,24,25,26,26.5,27,28,32' '3,3,2,1,1,7-,6-,6-,6-,4,4,3,2,1,2,4,3,4,4,3,2,2,4,3,3,1,6-,6-,7-,3,2,1,7-,1,6-' '0,1,1.5,2,3,3.5,4,5,6,8,9,9.5,10,10.5,11,11.5,12,16,17,17.5,18,19,19.5,20,21,21.5,22,23,24,25,26,26.5,27,27.5,28,32'
    # python grade_triple_item_util.py F:/项目/花城音乐项目/样式数据/7.17MP3/旋律/小学8题20190717-2776-6.wav [1000,500,500;2000;250,250,500,500,500;2000] [6,5,3,6,3,5,3,2,1,6-]
    # python grade_triple_item_util.py F:/项目/花城音乐项目/样式数据/7.17MP3/旋律/test.wav [1000,250,250,250,250;2000;1000,500,500;2000] [1,3,5,1+,6,5,1,3,2,1]