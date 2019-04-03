import re
import numpy, wave,matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import librosa
import librosa.display
from PIL import Image
import re
import shutil
from create_base import *
from create_labels_files import *
from myDtw import *
import tensorflow as tf
import numpy as np
import pdb
import cv2
import os
import glob
import slim.nets.alexnet as alaxnet
import os
from create_tf_record import *
import tensorflow.contrib.slim as slim
import heapq
import re
from denoise import denoise_by_rms


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


def get_single_onsets(filename):
    y, sr = librosa.load(filename)
    #librosa.display.waveplot(y, sr=sr)
    #onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
    onset_frames,onsets_frames_strength = get_onsets_by_all(y,sr)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    #plt.vlines(onset_times, 0, y.max(), color='r', linestyle='--')
    onset_samples = librosa.time_to_samples(onset_times)
    #print(onset_samples)
    #plt.subplot(len(onset_times),1,1)
    #plt.show()
    curr_num = 1
    for i in range(0, len(onset_times)):
        start = onset_samples[i] - sr/2
        if start < 0:
            start =0
        end = onset_samples[i] + sr/2
        #y2 = [x if i> start and i<end else 0 for i,x in enumerate(y)]
        tmp = [x if i > start and i < end else 0 for i, x in enumerate(y)]
        y2 = np.zeros(len(tmp))
        middle = int(len(y2) / 2)
        offset = middle - onset_samples[i]
        for j in range(len(tmp)):
            if tmp[j] > 0:
                y2[j + offset] = tmp[j]
        # y2 = [tmp[i + offset] for i in range(len(tmp)) if tmp[i]>0]
        # y2 = [0.03 if i> start and i<end else 0.02 for i,x in enumerate(y)]
        # y2[int(len(y2) / 2)] = np.max(y)  # 让图片展示归一化
        t = librosa.samples_to_time([middle], sr=sr)
        plt.vlines(t, -1 * np.max(y), np.max(y), color='r', linestyle='--')  # 标出节拍位置
        y2 = np.array(y2)
        #print("len(y2) is {}".format(len(y2)))

        #print("(end - start)*sr is {}".format((end - start)*sr))
        #plt.subplot(len(onset_times),1,i+1)
        #y, sr = librosa.load(filename, offset=2.0, duration=3.0)
        librosa.display.waveplot(y2, sr=sr)
        fig = matplotlib.pyplot.gcf()
        fig.set_size_inches(4, 4)
        if "." in filename:
            Filename = filename.split(".")[0]
        plt.axis('off')
        plt.axes().get_xaxis().set_visible(False)
        plt.axes().get_yaxis().set_visible(False)
        plt.savefig(savepath + str(curr_num) + '.png', bbox_inches='tight', pad_inches=0)
        plt.clf()
        curr_num += 1
    #plt.show()
    return onset_frames,onsets_frames_strength

'''
分割节奏图
'''
def split_onset_image(filename,onset_frames,onset_frames_strength,savepath):
    y, sr = librosa.load(filename)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    onset_samples = librosa.time_to_samples(onset_times)
    clear_dir(savepath)
    curr_num = 1
    for i in range(0, len(onset_times)):
        start = onset_samples[i] - sr / 2
        if start < 0:
            start = 0
        end = onset_samples[i] + sr / 2
        # y2 = [x if i> start and i<end else 0 for i,x in enumerate(y)]
        tmp = [x if i > start and i < end else 0 for i, x in enumerate(y)]
        y2 = np.zeros(len(tmp))
        middle = int(len(y2) / 2)
        offset = middle - onset_samples[i]
        for j in range(len(tmp)):
            if tmp[j] > 0:
                y2[j + offset] = tmp[j]

        t = librosa.samples_to_time([middle], sr=sr)
        plt.vlines(t, -1 * np.max(y), np.max(y), color='r', linestyle='--')  # 标出节拍位置
        y2 = np.array(y2)

        librosa.display.waveplot(y2, sr=sr)
        fig = matplotlib.pyplot.gcf()
        fig.set_size_inches(4, 4)
        if "." in filename:
            Filename = filename.split(".")[0]
        plt.axis('off')
        plt.axes().get_xaxis().set_visible(False)
        plt.axes().get_yaxis().set_visible(False)
        plt.savefig(savepath + str(curr_num) + '.png', bbox_inches='tight', pad_inches=0)
        plt.clf()
        curr_num += 1

    return onset_frames, onset_frames_strength

def  predict(filename,image_dir,onset_frames,onsets_frames_strength,models_path,f_range):
    if onset_frames:
        class_nums = 2
        onsets = []
        onsets_strength = {}
        tf.reset_default_graph()

        batch_size = 1  #
        resize_height = 224  # 指定存储图片高度
        resize_width = 224  # 指定存储图片宽度
        depths = 3
        data_format = [batch_size, resize_height, resize_width, depths]

        [batch_size, resize_height, resize_width, depths] = data_format

        #labels = np.loadtxt(labels_filename, str, delimiter='\t')
        input_images = tf.placeholder(dtype=tf.float32, shape=[None, resize_height, resize_width, depths], name='input')

        with slim.arg_scope(alaxnet.alexnet_v2_arg_scope()):
            out, end_points = alaxnet.alexnet_v2(inputs=input_images, num_classes=class_nums, dropout_keep_prob=1.0, is_training=False)

        # 将输出结果进行softmax分布,再求最大概率所属类别
        score = tf.nn.softmax(out,name='pre')
        class_id = tf.argmax(score, 1)

        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess, models_path)
        images_list=sorted(glob.glob(os.path.join(image_dir,'*.png')), key=os.path.getmtime)
        # images_list = glob.glob(os.path.join(image_dir, '*.png'))
        #sorted(glob.glob('*.png'), key=os.path.getmtime)
        score_total = 0
        index = 0
        key_type = type(list(onsets_frames_strength.keys())[0])
        for image_path in images_list:
            im=read_image(image_path,resize_height,resize_width,normalization=True)
            im=im[np.newaxis,:]
            #pred = sess.run(f_cls, feed_dict={x:im, keep_prob:1.0})
            pre_score,pre_label = sess.run([score,class_id], feed_dict={input_images:im})
            max_score=pre_score[0,pre_label]
            #print("{} is: pre labels:{},name:{} score: {}".format(image_path, pre_label, labels[pre_label], max_score))
            print("{} is predicted as label::{} ".format(image_path,pre_label[0]))

            # 将判断为yes的节拍加入onsets
            if 1 == pre_label[0]:
                #score_total += 1
                onsets.append(onset_frames[index])
                onsets_strength[key_type(onset_frames[index])] = onsets_frames_strength.get(key_type(onset_frames[index]))
            else:
                pass
            index += 1
        #accuracy = 0
        #print("valuation accuracy is {}".format(accuracy))
        sess.close()
        # if len(onsets)!=0:
        #     y, sr = librosa.load(filename)
        #     rms = librosa.feature.rmse(y=y)[0]
        #     rms = [x / np.std(rms) for x in rms]
        #     onsets,onsets_strength = remove_crowded_frames_by_rms(onsets,onsets_strength,rms,int(f_range*2)+2)
        return onsets,onsets_strength#,accuracy
    return [],{}

# '''
# 测试方法
# '''
# def test(filename,models_path):
#     import re
#     image_dir = '../single_onsets/data/test/test'
#     clear_dir(image_dir)
#     pattern = 'WAV/(.+)'
#     wavname = re.findall(pattern,filename)[0]
#     curr_num = 1
#     onset_frames, onsets_frames_strength, curr_num = get_single_onsets(filename, curr_num)
#     print("onset_frames,onsets_frames_strength is {},{}".format(onset_frames, onsets_frames_strength))
#     if onset_frames:
#         pre_onset_frames, pre_onset_strength, accuracy = predict(wavname,image_dir,onset_frames,onsets_frames_strength,models_path)
#         return onset_frames, pre_onset_frames, accuracy

'''
根据cqt判断节拍
'''
def detect_onset_by_cqt(filename):
    savepath = '../single_notes/data/test/'
    image_dir = '../single_notes/data/test/'
    # 清空文件夹
    if not os.path.exists(image_dir):
        os.mkdir(image_dir)
    clear_dir(image_dir)

    onset_frames= get_single_notes(filename,savepath,True)
    y, sr = librosa.load(filename)
    onset_strength = librosa.onset.onset_strength(y, sr)
    onset_strength = [x / np.std(onset_strength) for x in onset_strength]

    if onset_frames:
        #print("onsets, onsets_strength is {},{}".format(onset_frames, onsets_strength))
        y, sr = librosa.load(filename)
        #onset_frames = del_note_end_by_cqt(y,onset_frames)
        #onset_frames = del_note_middle_by_cqt(y,onset_frames)
        if len(onset_frames) > 0:
            min_width = get_min_width_onsets(filename)
            print("min_width is {}".format(min_width))
            onset_frames = del_overcrowding(onset_frames, min_width/3)
        #librosa.display.waveplot(y, sr=sr)
        CQT = librosa.amplitude_to_db(librosa.cqt(y, sr=16000), ref=np.max)
        w, h = CQT.shape
        CQT[50:w, :] = np.min(CQT)
        CQT[0:20, :] = np.min(CQT)
        for i in range(w):
            for j in range(h):
                if CQT[i][j] > -20:
                    CQT[i][j] = np.max(CQT)
                # else:
                #     CQT[i][j] = np.min(CQT)
    onset_strength2 = {}
    for i in onset_frames:
        onset_strength2[i] = onset_strength[i]
    onset_frames.sort()
    return onset_frames,onset_strength2

'''
通过rms删去节拍中间的伪节拍
'''
def del_onset_middle_by_rms(onset_frames,onset_frames_strength,rms,f_range):
    K = 10
    topK_avg_rms = np.sum(np.array(rms)[np.argpartition(np.array(rms), -K)[-K:]]) / K
    threshold = topK_avg_rms * 0.2
    key_type = type(list(onset_frames_strength.keys())[0])
    result = []
    result_strength = {}

    for frame in onset_frames:
        rms_before = []
        for i in range(frame - f_range,frame + 1):
            if i < 0:
                continue
            rms_before.append(rms[i])
        local_rms_min = np.min(rms_before)

        # 若局部最小rms小于阈值则是节拍 否则为节拍中部
        if local_rms_min < threshold:
            result.append(frame)
            result_strength[frame] = onset_frames_strength.get(key_type(frame))

    return result,result_strength

'''
根据rms来判断节拍
'''
def detect_onset_by_rms(filename,onset_frames,f_range):
    y, sr = librosa.load(filename)
    rms = librosa.feature.rmse(y=y)[0]
    rms = [x / np.std(rms) for x in rms]
    rms = denoise_by_rms(rms)
    K = 10
    topK_avg_rms = np.sum(np.array(rms)[np.argpartition(np.array(rms),-K)[-K:]])/K
    threshold = topK_avg_rms * 0.2
    remove_list = []
    onset_strength = librosa.onset.onset_strength(y, sr)
    onset_strength = [x / np.std(onset_strength) for x in onset_strength]
    for frame in onset_frames:
        local_max_rms = -1
        local_max_rms_frame = 0
        for i in range(frame - f_range, frame + f_range + 1):
            if i > len(rms) - 1:
                break
            if i < 0:
                continue
            if rms[i] > local_max_rms:
                local_max_rms = rms[i]
                local_max_rms_frame = i
        if local_max_rms < threshold:
            remove_list.append(frame)
        else:
            onset_frames = [local_max_rms_frame if i == frame else i for i in onset_frames]
    onset_frames = list(set(onset_frames))
    for frame in remove_list:
        onset_frames.remove(frame)

    onset_frames_strength = {}
    for frame in onset_frames:
        onset_frames_strength[str(frame)] = onset_strength[frame]
    if len(onset_frames)!=0:
        onset_frames, onset_frames_strength = remove_crowded_frames_by_rms(onset_frames, onset_frames_strength,rms,f_range+3)
        onset_frames, onset_frames_strength = del_onset_middle_by_rms(onset_frames, onset_frames_strength, rms, f_range+6)
    return onset_frames,onset_frames_strength

def denoise_by_rms(rms):

    total_rms = []

    for i in range(0,5):
        if rms[i] != 0:
            total_rms.append(rms[i])
        if rms[len(rms)-1-i]!=0:
            total_rms.append(rms[len(rms)-1-i])

    threshold = np.average(total_rms)

    rms = [0 if i < threshold else i for i in rms]

    return rms

'''
根据strength判断节拍
'''
# def detect_onset_by_strength(filename,onset_frames,f_range):
#     y, sr = librosa.load(filename)
#     onset_strength = librosa.onset.onset_strength(y, sr)
#     onset_strength = [x / np.std(onset_strength) for x in onset_strength]
#     K = 8
#     topK_avg_strength = np.sum(np.array(onset_strength)[np.argpartition(np.array(onset_strength), -K)[-K:]]) / K
#     threshold = topK_avg_strength * 0.32
#     remove_list = []
#     for frame in onset_frames:
#         local_max_strength = -1
#         local_max_strength_frame = 0
#         for i in range(frame - f_range, frame + f_range):
#             if i > len(onset_strength) - 1:
#                 break
#             if i < 0:
#                 continue
#             if local_max_strength > onset_strength[i]:
#                 local_max_strength = onset_strength[i]
#                 local_max_strength_frame = i
#         if local_max_strength < threshold:
#             remove_list.append(frame)
#         else:
#             onset_frames = [local_max_strength_frame if i == frame else i for i in onset_frames]
#     for frame in remove_list:
#         onset_frames.remove(frame)
#     onset_frames_strength = {}
#     for frame in onset_frames:
#         onset_frames_strength[str(frame)] = onset_strength[frame]
#     if len(onset_frames)!=0:
#         onset_frames,onset_frames_strength = remove_crowded_frames(onset_frames,onset_frames_strength,int(f_range*2)+2)
#     return onset_frames,onset_frames_strength

'''
去掉挤在一起的线
'''
def remove_crowded_frames_by_rms(onset_frames,onset_frames_strength,rms,crowded_range):
    frames_remove_list = []
    key_type = type(list(onset_frames_strength.keys())[0])
    onset_frames.sort()
    for i in range(0,len(onset_frames)-1):
        if abs(onset_frames[i] - onset_frames[i+1])<crowded_range:
            if rms[onset_frames[i]] >= rms[onset_frames[i+1]]:
                frames_remove_list.append(onset_frames[i+1])
            else:
                frames_remove_list.append(onset_frames[i])
    # 去重
    frames_remove_list = list(set(frames_remove_list))
    onset_frames = list(set(onset_frames))
    for i in range(0,len(frames_remove_list)):
        onset_frames.remove(frames_remove_list[i])
        onset_frames_strength.pop(key_type(frames_remove_list[i]))
    return onset_frames,onset_frames_strength


if __name__ ==  '__main__':
    '''
    create_base参数修改为
    gap1 = 0.2
    gap2 = 0.3
    gap3 = 0.3
    gap4 = 2.8
    '''
    savepath = '../single_onsets/data/test/test/'
    labels_filename = '../single_onsest/data/label.txt'
    #models_path = '../single_onsets/models/wzd/model.ckpt-10000'
    models_path = '../single_onsets/models/alex/model.ckpt-10000'
    image_dir = savepath

    '''================================================================================================================='''
    '''
    测试单个音频文件
    '''
    # filename = '../mp3/2.27WAV/节奏3_40207（100）.wav'
    # filename = '../mp3/2.27WAV/节奏1-01（70）.wav'
    # filename = '../mp3/3.19WAV/节奏4录音4（55）.wav'
    filename = '../mp3/1.31WAV/节奏3.20分.wav'
    clear_dir(image_dir)
    wavname = filename
    y, sr = librosa.load(filename)
    rms = librosa.feature.rmse(y=y)[0]
    rms = [x / np.std(rms) for x in rms]
    type_index = get_onsets_index_by_filename(filename)
    total_frames_number = get_total_frames_number(filename)
    base_frames = onsets_base_frames(codes[type_index], total_frames_number)
    f_range = int(len(rms)/(10*len(base_frames)))
    print('f_range is {}'.format(f_range))


    # 获取一次判断的节拍帧和强度
    onset_frames, onset_frames_strength = get_single_onsets(filename)
    # 用cqt修正判断
    cqt_onset_frames, cqt_onset_frames_strength = detect_onset_by_cqt(filename)
    onset_frames = list(set(cqt_onset_frames + onset_frames))
    onset_frames_strength.update(cqt_onset_frames_strength)
    onset_frames_time = librosa.frames_to_time(onset_frames, sr=sr)

    if onset_frames:
        onset_frames_time = librosa.frames_to_time(onset_frames, sr=sr)


        # rms判断
        rms_onset_frames, rms_onset_frames_strength = detect_onset_by_rms(filename, onset_frames,f_range)
        rms_onset_frames_time = librosa.frames_to_time(rms_onset_frames, sr=sr)

        # 模型判断
        after_rms_onset_frames,after_rms_onset_frames_strength= split_onset_image(filename,rms_onset_frames,rms_onset_frames_strength,savepath)
        pre_onset_frames, pre_onset_strength = predict(filename,
                                                       image_dir,
                                                       after_rms_onset_frames,
                                                       after_rms_onset_frames_strength,
                                                       models_path,
                                                       f_range)
        pre_onset_time = librosa.frames_to_time(pre_onset_frames, sr=sr)


        # # 模型判断后调用rms方法二次判断
        # after_model_rms_onset_frames, after_model_rms_onset_frames_strength = detect_onset_by_rms(filename, pre_onset_frames,f_range)
        # after_model_rms_onset_frames_time = librosa.frames_to_time(rms_onset_frames, sr=sr)

        # # 模型判断后调用strength方法二次判断
        # after_model_strength_onset_frames, strength_onset_frames_strength = detect_onset_by_strength(filename, pre_onset_frames,f_range)
        # after_model_strength_onset_frames_time = librosa.frames_to_time(after_model_strength_onset_frames, sr=sr)

        '''绘图'''
        plt.subplot(3, 1, 1)
        plt.ylabel('O')
        librosa.display.waveplot(y, sr=sr)
        plt.vlines(onset_frames_time, -1 * np.max(y), np.max(y), color='r', linestyle='dashed')

        plt.subplot(3, 1, 2)
        plt.ylabel('O-R')
        librosa.display.waveplot(y, sr=sr)
        plt.vlines(rms_onset_frames_time, -1 * np.max(y), np.max(y), color='r', linestyle='dashed')

        plt.subplot(3, 1, 3)
        plt.ylabel('O-R-M')
        librosa.display.waveplot(y, sr=sr)
        plt.vlines(pre_onset_time, -1 * np.max(y), np.max(y), color='r', linestyle='dashed')

        # plt.subplot(4, 1, 4)
        # plt.ylabel('O-R-M-C')
        # librosa.display.waveplot(y, sr=sr)
        # plt.vlines(cqt_onset_frames_time, -1 * np.max(y), np.max(y), color='r', linestyle='solid')



        # plt.subplot(4, 1, 4)
        # plt.ylabel('O-M-R')
        # librosa.display.waveplot(y, sr=sr)
        # plt.vlines(after_model_rms_onset_frames_time, -1 * np.max(y), np.max(y), color='r', linestyle='solid')

        # plt.subplot(5, 1, 5)
        # plt.ylabel('O-M-S')
        # librosa.display.waveplot(y, sr=sr)
        # plt.vlines(after_model_strength_onset_frames_time, -1 * np.max(y), np.max(y), color='r', linestyle='solid')

        plt.show()
    '''================================================================================================================='''
    '''
    测试多个文件
    '''
    # import re
    # import os
    #
    # dir_list = ['../mp3/1.31WAV/','../mp3/2.2WAV/','../mp3/2.18WAV/','../mp3/2.27WAV/','../mp3/3.19WAV/']
    # total_accuracy = 0
    # total_num = 0
    # pre_savepath = '../single_onsets/data/compare_pre_onsets/'
    # # 要测试的数量
    # test_num = 10000
    # if not os.path.exists(pre_savepath):
    #     os.mkdir(pre_savepath)
    # for dir in dir_list:
    #     file_list = os.listdir(dir)
    #     # 将识别图存在相应的日期文件下
    #     pattern = 'mp3/(.+)WAV'
    #     date = re.findall(pattern, dir)[0]
    #     if not os.path.exists(pre_savepath + date):
    #         os.mkdir(pre_savepath + date)
    #     clear_dir(pre_savepath + date)
    #     for filename in file_list:
    #
    #         print("-------current test dir is {},filename is {}-------".format(dir, filename))
    #         #onsets_frames,pre_onsets, accuracy = test(dir + filename, models_path)  # 参数必须是完整路径
    #         # pre_onsets2,accuracy = test(dir+filename,models_path2)
    #         # pre_onsets = list(set(pre_onsets1+pre_onsets2))
    #         # total_accuracy += accuracy
    #         # total_num += 1
    #         # mean_accuracy = total_accuracy / total_num
    #         #print('-------current test num is {},current mean accuracy is {}-------'.format(total_num, mean_accuracy))
    #
    #         clear_dir(image_dir)
    #         wavname = filename
    #         curr_num = 1
    #         y, sr = librosa.load(dir + filename)
    #         rms = librosa.feature.rmse(y=y)[0]
    #         rms = [x / np.std(rms) for x in rms]
    #         type_index = get_onsets_index_by_filename(dir + filename)
    #         total_frames_number = get_total_frames_number(dir + filename)
    #         base_frames = onsets_base_frames(codes[type_index], total_frames_number)
    #         f_range = int(len(rms) / (10 * len(base_frames)))
    #         print('f_range is {}'.format(f_range))
    #         # 获取一次判断的节拍帧和强度
    #         onset_frames, onset_frames_strength = get_single_onsets(dir + filename)
    #         # 用cqt修正判断
    #         cqt_onset_frames, cqt_onset_frames_strength = detect_onset_by_cqt(dir + filename)
    #         onset_frames = list(set(cqt_onset_frames + onset_frames))
    #         onset_frames_strength.update(cqt_onset_frames_strength)
    #         onset_frames_time = librosa.frames_to_time(onset_frames, sr=sr)
    #         if onset_frames:
    #             onset_frames_time = librosa.frames_to_time(onset_frames, sr=sr)
    #
    #             # rms判断
    #             rms_onset_frames, rms_onset_frames_strength = detect_onset_by_rms(dir + filename, onset_frames, f_range)
    #             rms_onset_frames_time = librosa.frames_to_time(rms_onset_frames, sr=sr)
    #
    #             # 模型判断
    #             after_rms_onset_frames, after_rms_onset_frames_strength = split_onset_image(dir + filename, rms_onset_frames,
    #                                                                                         rms_onset_frames_strength,
    #                                                                                         savepath)
    #             pre_onset_frames, pre_onset_strength = predict(filename,
    #                                                            image_dir,
    #                                                            after_rms_onset_frames,
    #                                                            after_rms_onset_frames_strength,
    #                                                            models_path,
    #                                                            f_range)
    #             pre_onset_time = librosa.frames_to_time(pre_onset_frames, sr=sr)
    #
    #             # # 模型判断后调用rms方法二次判断
    #             # after_model_rms_onset_frames, after_model_rms_onset_frames_strength = detect_onset_by_rms(dir + filename,
    #             #                                                                                           pre_onset_frames,
    #             #                                                                                           f_range)
    #             # after_model_rms_onset_frames_time = librosa.frames_to_time(rms_onset_frames, sr=sr)
    #
    #             # # 模型判断后调用strength方法二次判断
    #             # after_model_strength_onset_frames, strength_onset_frames_strength = detect_onset_by_strength(dir + filename,
    #             #                                                                                              pre_onset_frames,
    #             #                                                                                              f_range)
    #             # after_model_strength_onset_frames_time = librosa.frames_to_time(after_model_strength_onset_frames,
    #             #                                                                 sr=sr)
    #
    #             '''绘图'''
    #             plt.subplot(3, 1, 1)
    #             plt.ylabel('O')
    #             librosa.display.waveplot(y, sr=sr)
    #             plt.vlines(onset_frames_time, -1 * np.max(y), np.max(y), color='r', linestyle='dashed')
    #
    #             plt.subplot(3, 1, 2)
    #             plt.ylabel('O-R')
    #             librosa.display.waveplot(y, sr=sr)
    #             plt.vlines(rms_onset_frames_time, -1 * np.max(y), np.max(y), color='r', linestyle='dashed')
    #
    #             plt.subplot(3, 1, 3)
    #             plt.ylabel('O-R-M')
    #             librosa.display.waveplot(y, sr=sr)
    #             plt.vlines(pre_onset_time, -1 * np.max(y), np.max(y), color='r', linestyle='dashed')
    #
    #             # plt.subplot(4, 1, 4)
    #             # plt.ylabel('O-M-R')
    #             # librosa.display.waveplot(y, sr=sr)
    #             # plt.vlines(after_model_rms_onset_frames_time, -1 * np.max(y), np.max(y), color='r', linestyle='solid')
    #
    #             # plt.subplot(5, 1, 5)
    #             # plt.ylabel('O-M-S')
    #             # librosa.display.waveplot(y, sr=sr)
    #             # plt.vlines(after_model_strength_onset_frames_time, -1 * np.max(y), np.max(y), color='r',
    #             #            linestyle='solid')
    #
    #
    #         # 保存识别对比图
    #         # plt.show()
    #         plt.savefig(pre_savepath + date + '/' + filename.strip('.wav') + '.png', bbox_inches='tight', pad_inches=0)
    #
    #         if total_num > test_num:
    #             break
    #     if total_num > test_num:
    #         break

    '''================================================================================================================='''
    '''
    测试cqt
    '''
    # # filename = '../mp3/2.27WAV/节奏3_40207（100）.wav'
    # # filename = '../mp3/2.27WAV/节奏1-01（70）.wav'
    # # filename = '../mp3/3.19WAV/节奏4录音4（55）.wav'
    # filename = '../mp3/1.31WAV/节奏5.80分.wav'
    # clear_dir(image_dir)
    # wavname = filename
    # y, sr = librosa.load(filename)
    # rms = librosa.feature.rmse(y=y)[0]
    # rms = [x / np.std(rms) for x in rms]
    # type_index = get_onsets_index_by_filename(filename)
    # total_frames_number = get_total_frames_number(filename)
    # base_frames = onsets_base_frames(codes[type_index], total_frames_number)
    # f_range = int(len(rms) / (10 * len(base_frames)))
    # print('f_range is {}'.format(f_range))
    #
    # # 用cqt修正判断
    # cqt_onset_frames, cqt_onset_frames_strength = detect_onset_by_cqt(filename)
    # cqt_onset_frames_time = librosa.frames_to_time(cqt_onset_frames, sr=sr)
    # librosa.display.waveplot(y, sr=sr)
    # plt.vlines(cqt_onset_frames_time, -1 * np.max(y), np.max(y), color='r', linestyle='dashed')
    # plt.show()

