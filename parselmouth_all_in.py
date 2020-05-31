
from parselmouth_util import score_all

# 速度为88的切分时间点
slice0_seg0_start = 11.41
slice0_seg0_end = 33.23
slice0_seg1_start = 33.23
slice0_seg1_end = 55.05

# 速度为60的切分时间点
slice1_seg0_start = 16.5
slice1_seg0_end = 48.5
slice1_seg1_start = 48.5
slice1_seg1_end = 80.5

from pydub import AudioSegment
import os

def slice_file(save_path,raw_file,save_file, start,end):
    audio = AudioSegment.from_wav(raw_file)
    audio[start * 1000:end * 1000].export(save_path + save_file, format="wav")

def slice_file_to_segs(filename,type):
    save_path = 'F:/tmp/'
    if type == 0:
        seg0_filename = filename.split('/')[-1].split(".wav")[0] + '-1133.wav'
        start, end = slice0_seg0_start,slice0_seg0_end
        slice_file(save_path,filename, seg0_filename, start, end)   # 切分第一段
        seg1_filename = filename.split('/')[-1].split(".wav")[0] + '-3355.wav'
        start, end = slice0_seg1_start, slice0_seg1_end
        slice_file(save_path,filename,  seg1_filename, start, end)   # 切分第二段
        return save_path + seg0_filename, save_path + seg1_filename
    elif type == 1:
        seg0_filename = filename.split('/')[-1].split(".wav")[0] + '-1648.wav'
        start, end = slice1_seg0_start, slice1_seg0_end
        slice_file(save_path,filename,  seg0_filename, start, end)  # 切分第一段
        seg1_filename = filename.split('/')[-1].split(".wav")[0] + '-4880.wav'
        start, end = slice1_seg1_start, slice1_seg1_end
        slice_file(save_path,filename,  seg1_filename, start, end)  # 切分第二段
        return save_path + seg0_filename, save_path + seg1_filename

def webm2wav(filename):
    tmp = filename.split('.webm')[0] + ".wav"
    if os.path.exists(tmp):
        os.remove(tmp)
    if filename.find(".webm") >= 0:
        filename = filename.replace(" ", "-")
        str = "F:/ffmpeg/bin/ffmpeg -i " + filename + " " + filename.split('.webm')[0] + ".wav"
    print(str)
    # os.system("ffmpeg -i "+ m4a_path + m4a + " " + m4a_path + m4a.split('.')[0] + ".wav" )   # ffmpeg 可以用来处理视频和音频文件，可               #以设置一些参数，如采样频率等
    os.system(str)
    return filename.split('.webm')[0] + ".wav"

def score_by_type(seg0_filename,seg1_filename,type):
    if type == 0:
        seg0_duration = 21.8
        seg1_duration = 21.8

         #第一段的参数
        standard_kc = '喜爱春天的人儿是心地纯洁的人像紫罗兰花儿一样是我知心朋友'
        standard_kc_time = [0,0.6818181818181817,1.363636363636365,2.0454545454545467,2.3863636363636385,2.72727272727273,3.409090909090912,4.0909090909090935,5.454545454545459,6.136363636363642,6.818181818181822,7.500000000000002,7.840909090909092,8.181818181818182,10.909090909090908,11.590909090909092,12.272727272727272,12.954545454545455,13.636363636363635,14.318181818181818,14.999999999999998,15.681818181818182,16.363636363636367,17.045454545454547,17.727272727272734,18.06818181818182,18.409090909090914,19.090909090909093,20.454545454545453,21.81818181818182]
        standard_notations = '3,3,2,1,1,7-,6-,6-,6-,4,4,3,2,1,2,4,3,4,4,3,2,2,4,3,3,1,6-,6-,7-,3,2,1,7-,1,6-'
        standard_notation_time = [0,0.6818181818181817,1.0227272727272734,1.363636363636365,2.0454545454545467,2.3863636363636385,2.72727272727273,3.409090909090912,4.0909090909090935,5.454545454545459,6.136363636363642,6.477272727272732,6.818181818181822,7.159090909090912,7.500000000000002,7.840909090909092,8.181818181818182,10.909090909090908,11.590909090909092,11.931818181818182,12.272727272727272,12.954545454545455,13.295454545454545,13.636363636363635,14.318181818181818,14.659090909090908,14.999999999999998,15.681818181818182,16.363636363636367,17.045454545454547,17.727272727272734,18.06818181818182,18.409090909090914,18.75,19.090909090909093,21.81818181818182]
        total_score0, pitch_total_score, notation_duration_total_score, kc_duration_total_score,kc_express_total_score,fluency_total_score, pitch_score_detail, notation_duration_score_detail, kc_rhythm_sscore_detail,kc_express_sscore_detail,fluency_sscore_detail = score_all(seg0_filename, standard_kc,standard_kc_time, standard_notations, standard_notation_time)
        print("total_score0 is {}".format(total_score0))
        score_detail = "音高评分结果为{}，{}，音符节奏评分结果为{}，{}，歌词节奏评分结果为{}，{}，歌词表达评分结果为{}，{}，流畅度评分结果为{}，{}".format(pitch_total_score,
                                                                           pitch_score_detail,
                                                                           notation_duration_total_score,
                                                                           notation_duration_score_detail,
                                                                           kc_duration_total_score,
                                                                           kc_rhythm_sscore_detail,
                                                                           kc_express_total_score,
                                                                           kc_express_sscore_detail,
                                                                           fluency_total_score,
                                                                           fluency_sscore_detail)
        print(score_detail)

        # 第二段的参数
        standard_kc = '喜爱夏天的人儿是意志坚强的人像冲打岩礁的波浪一样是我敬爱的父亲'
        standard_kc_time = [0,0.6818181818181799,1.3636363636363669,2.0454545454545467,2.3863636363636402,2.7272727272727337,3.4090909090909136,4.0909090909090935,5.45454545454546,6.13636363636364,6.818181818181827,7.500000000000014,7.840909090909108,8.181818181818201,10.909090909090928,11.590909090909108,11.931818181818201,12.272727272727295,12.954545454545475,13.295454545454568,13.636363636363662,14.318181818181841,15.000000000000028,15.681818181818208,16.363636363636388,17.045454545454568,17.727272727272748,18.06818181818184,18.238636363636388,18.409090909090935,19.090909090909122,20.45454545454549,21.81818181818185 ]
        standard_notations = '3,3,2,1,1,7-,6-,6-,6-,4,4,3,2,1,2,4,3,4,4,3,2,2,4,3,3,1,6-,6-,7-,3,2,1,7-,1,6-'
        standard_notation_time = [0,0.6818181818181799,1.0227272727272734,1.3636363636363669,2.0454545454545467,2.3863636363636402,2.7272727272727337,3.4090909090909136,4.0909090909090935,5.45454545454546,6.13636363636364,6.477272727272734,6.818181818181827,7.159090909090921,7.500000000000014,7.840909090909108,8.181818181818201,10.909090909090928,11.590909090909108,11.931818181818201,12.272727272727295,12.954545454545475,13.295454545454568,13.636363636363662,14.318181818181841,14.659090909090935,15.000000000000028,15.681818181818208,16.363636363636388,17.045454545454568,17.727272727272748,18.06818181818184,18.409090909090935,18.75000000000003,19.090909090909122,21.81818181818185]
        total_score1, pitch_total_score, notation_duration_total_score, kc_duration_total_score,kc_express_total_score,fluency_total_score, pitch_score_detail, notation_duration_score_detail, kc_rhythm_sscore_detail,kc_express_sscore_detail,fluency_sscore_detail = score_all(seg0_filename, standard_kc,standard_kc_time, standard_notations, standard_notation_time)
        print("total_score1 is {}".format(total_score0))
        score_detail = "音高评分结果为{}，{}，音符节奏评分结果为{}，{}，歌词节奏评分结果为{}，{}，歌词表达评分结果为{}，{}，流畅度评分结果为{}，{}".format(pitch_total_score,
                                                                           pitch_score_detail,
                                                                           notation_duration_total_score,
                                                                           notation_duration_score_detail,
                                                                           kc_duration_total_score,
                                                                           kc_rhythm_sscore_detail,
                                                                           kc_express_total_score,
                                                                           kc_express_sscore_detail,
                                                                           fluency_total_score,
                                                                           fluency_sscore_detail)
        print(score_detail)

    elif type == 1:
        seg0_duration = 32
        seg1_duration = 32

        #第一段的参数
        standard_kc = '喜爱春天的人儿是心地纯洁的人像紫罗兰花儿一样是我知心朋友'
        standard_kc_time = [0,1,2,3,3.5,4,5,6,8,9,10,11,11.5,12,16,17,18,19,20,21,22,23,24,25,26,26.5,27,28,32]
        standard_notations = '3,3,2,1,1,7-,6-,6-,6-,4,4,3,2,1,2,4,3,4,4,3,2,2,4,3,3,1,6-,6-,7-,3,2,1,7-,1,6-'
        standard_notation_time = [0,1,1.5,2,3,3.5,4,5,6,8,9,9.5,10,10.5,11,11.5,12,16,17,17.5,18,19,19.5,20,21,21.5,22,23,24,25,26,26.5,27,27.5,28,32]
        total_score0, pitch_total_score, notation_duration_total_score, kc_duration_total_score,kc_express_total_score,fluency_total_score, pitch_score_detail, notation_duration_score_detail, kc_rhythm_sscore_detail,kc_express_sscore_detail,fluency_sscore_detail = score_all(seg0_filename, standard_kc,standard_kc_time, standard_notations, standard_notation_time)
        print("total_score0 is {}".format(total_score0))
        score_detail = "音高评分结果为{}，{}，音符节奏评分结果为{}，{}，歌词节奏评分结果为{}，{}，歌词表达评分结果为{}，{}，流畅度评分结果为{}，{}".format(pitch_total_score,
                                                                           pitch_score_detail,
                                                                           notation_duration_total_score,
                                                                           notation_duration_score_detail,
                                                                           kc_duration_total_score,
                                                                           kc_rhythm_sscore_detail,
                                                                           kc_express_total_score,
                                                                           kc_express_sscore_detail,
                                                                           fluency_total_score,
                                                                           fluency_sscore_detail)
        print(score_detail)

        # 第二段的参数
        standard_kc = '喜爱夏天的人儿是意志坚强的人像冲打岩礁的波浪一样是我敬爱的父亲'
        standard_kc_time = [0,1,2,3,3.5,4,5,6,8,9,10,11,11.5,12,16,17,17.5,18,19,19.5,20,21,22,23,24,25,26,26.5,26.75,27,28,32]
        standard_notations = '3,3,2,1,1,7-,6-,6-,6-,4,4,3,2,1,2,4,3,4,4,3,2,2,4,3,3,1,6-,6-,7-,3,2,1,7-,1,6-'
        standard_notation_time = [0,1,1.5,2,3,3.5,4,5,6,8,9,9.5,10,10.5,11,11.5,12,16,17,17.5,18,19,19.5,20,21,21.5,22,23,24,25,26,26.5,27,27.5,28,32]
        total_score1, pitch_total_score, notation_duration_total_score, kc_duration_total_score,kc_express_total_score,fluency_total_score, pitch_score_detail, notation_duration_score_detail, kc_rhythm_sscore_detail,kc_express_sscore_detail,fluency_sscore_detail = score_all(seg0_filename, standard_kc,standard_kc_time, standard_notations, standard_notation_time)
        print("total_score1 is {}".format(total_score0))
        score_detail = "音高评分结果为{}，{}，音符节奏评分结果为{}，{}，歌词节奏评分结果为{}，{}，歌词表达评分结果为{}，{}，流畅度评分结果为{}，{}".format(pitch_total_score,
                                                                           pitch_score_detail,
                                                                           notation_duration_total_score,
                                                                           notation_duration_score_detail,
                                                                           kc_duration_total_score,
                                                                           kc_rhythm_sscore_detail,
                                                                           kc_express_total_score,
                                                                           kc_express_sscore_detail,
                                                                           fluency_total_score,
                                                                           fluency_sscore_detail)
        print(score_detail)
    total_duration = round(seg0_duration + seg1_duration,2)
    total_score = round((total_score0*seg0_duration + total_score1*seg1_duration)/total_duration,2)
    print("total_score is {}".format(total_score))
    return total_score

if __name__ == '__main__':
    filename,type = 'F:/项目/花城音乐项目/样式数据/20.05.20MP3/20200520-2360.webm',0
    filename,type = 'F:/项目/花城音乐项目/样式数据/20.05.20MP3/20200520-8187.webm',0
    filename,type = 'F:/项目/花城音乐项目/样式数据/20.05.26MP3/20200526-8406.webm',0
    filename, type = 'F:/项目/花城音乐项目/样式数据/20.05.30MP3/20200530-2202.webm', 0
    filename, type = 'F:/项目/花城音乐项目/样式数据/20.05.30MP3/20200530-9053.webm', 0
      # filename,type = 'F:/项目/花城音乐项目/样式数据/20.05.20MP3/20200520-8605.webm',0
    # filename,type = 'F:/项目/花城音乐项目/样式数据/20.05.01MP3/dbg/1701/m.webm',1
    # filename,type = 'F:/项目/花城音乐项目/样式数据/20.05.26MP3/20200526-1002.webm',1
    # filename,type = 'F:/项目/花城音乐项目/样式数据/20.04.08MP3/2段词-标准.webm',1

    filename = webm2wav(filename)
    seg0_filename,seg1_filename = slice_file_to_segs(filename, type)

    # seg0_filename,type = 'F:/项目/花城音乐项目/样式数据/20.04.08MP3/2段词-标准1648.wav',1
    # seg1_filename, type = 'F:/项目/花城音乐项目/样式数据/20.04.08MP3/2段词-标准4882.wav', 1
    # seg0_filename, type = 'F:/项目/花城音乐项目/样式数据/20.04.08MP3/2段词-不标准1648.wav', 1
    # seg1_filename, type = 'F:/项目/花城音乐项目/样式数据/20.04.08MP3/2段词-不标准4882.wav', 1
    # seg0_filename, type = 'F:/项目/花城音乐项目/样式数据/20.05.26MP3/tmp-25522-1DENXMdu4JwK-.wav', 0
    # seg1_filename, type = 'F:/项目/花城音乐项目/样式数据/20.05.26MP3/tmp-25522-1JAPZZrhv9u9-.wav', 0
    print(seg0_filename)
    print(seg1_filename)
    total_score = score_by_type(seg0_filename, seg1_filename, type)