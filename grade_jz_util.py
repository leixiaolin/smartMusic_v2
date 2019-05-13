import argparse
import sys
from find_mismatch import *

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
parser.add_argument("file_code", help="音频文件的节奏编码",type=str)

args = parser.parse_args()
print(len(sys.argv))
if len(sys.argv) != 3:
     parser.print_help()
     sys.exit(1)
else:
    file_path = args.file_path
    file_code = args.file_code
    print("file_path is {}".format(file_path))
    print("file_code is {}".format(file_code))
    #file_path = 'F:/项目/花城音乐项目/样式数据/3.06MP3/节奏/节4欧(95).wav'
    #file_code = '[19,60,92,128,161,178,197,230,263]'
    score, lost_score, ex_score, min_d = get_score_jz(file_path,file_code)
    print("score, lost_score, ex_score, min_d is {},{},{},{}".format(score, lost_score, ex_score, min_d))
    filepath, fullflname = os.path.split(file_path)
    output_file = fullflname.split('.wav')[0] + '-out.txt'
    content = 'score, lost_score, ex_score, min_d is ' + str(score) + ' ' + str(lost_score) + str(ex_score) + ' '+ str(min_d)
    save_path = os.path.join(filepath,output_file)
    write_txt(content, save_path, mode='w')
    #python grade_jz_util.py F:/项目/花城音乐项目/样式数据/3.06MP3/节奏/节4欧(95).wav [19,60,92,128,161,178,197,230,263]