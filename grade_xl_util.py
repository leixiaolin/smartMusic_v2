import argparse
import sys
from note_lines_helper import *
import warnings
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
    #file_path = 'F:/项目/花城音乐项目/样式数据/3.06MP3/节奏/节4欧(95).wav'
    #file_code = '[19,60,92,128,161,178,197,230,263]'
    total_score, onset_score, note_scroe = get_melody_score(file_path,rhythm_code,melody_code)
    print("total_score,onset_score, note_scroe is {},{},{}".format(total_score,onset_score, note_scroe))
    filepath, fullflname = os.path.split(file_path)
    output_file = fullflname.split('.wav')[0] + '-out.txt'
    content = 'total_score,onset_score, note_scroe is ' + str(total_score) + ' ' + str(onset_score) + ' ' + str(note_scroe)
    content += "\n"
    save_path = os.path.join(filepath,output_file)
    write_txt(content, save_path, mode='w')
    detail = '整体节奏偏离较大，未能误别部分节拍，流畅平稳性有待加强'
    write_txt(detail, save_path, mode='a')
    #python grade_xl_util.py F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋3罗（80）.wav [1000,1000;500,500,1000;500,250,250,500,500;2000] [5,5,3,2,1,2,2,3,2,6-,5-]