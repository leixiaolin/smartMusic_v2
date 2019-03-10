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

from create_tf_record import *
import tensorflow.contrib.slim as slim

#savepath = 'e:/test_image/'
savepath = './single_onsets/data/test/'
filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏1.3(95).wav'
filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏1（二）(100).wav'
filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏1_40227（100）.wav'
filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏1林(70).wav'
filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏2_40314（100）.wav'
filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏2_40409（98）.wav'
#filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏2林(25).wav'
#filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏2语(85).wav'
#filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏10_40411（85）.wav'
#filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏10-04（80）.wav'
filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏九（2）（95）.wav'


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

    for i in range(0, len(onset_times)):
        start = onset_samples[i] - sr/2
        if start < 0:
            start =0
        end = onset_samples[i] + sr/2
        #y2 = [x if i> start and i<end else 0 for i,x in enumerate(y)]
        y2 = [x for i,x in enumerate(y) if i> start and i<end]
        #y2 = [0.03 if i> start and i<end else 0.02 for i,x in enumerate(y)]
        y2[int(len(y2) / 2)] = np.max(y)  # 让图片展示归一化
        t = librosa.samples_to_time([onset_samples[i]-start], sr=sr)
        plt.vlines(t, -1*np.max(y), np.max(y), color='r', linestyle='--') # 标出节拍位置
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
        plt.savefig(savepath + str(i+1) + '.jpg', bbox_inches='tight', pad_inches=0)
        plt.clf()
    #plt.show()
    return onset_frames,onsets_frames_strength

def  predict(image_dir,onset_frames,onsets_frames_strength,models_path):
    class_nums = 2
    onsets = []
    onsets_strength = {}


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
    images_list=sorted(glob.glob(os.path.join(image_dir,'*.jpg')), key=os.path.getmtime)
    #sorted(glob.glob('*.png'), key=os.path.getmtime)
    score_total = 0
    index = 0
    for image_path in images_list:
        im=read_image(image_path,resize_height,resize_width,normalization=True)
        im=im[np.newaxis,:]
        #pred = sess.run(f_cls, feed_dict={x:im, keep_prob:1.0})
        pre_score,pre_label = sess.run([score,class_id], feed_dict={input_images:im})
        max_score=pre_score[0,pre_label]
        #print("{} is: pre labels:{},name:{} score: {}".format(image_path, pre_label, labels[pre_label], max_score))
        print("{} is predicted as label::{} ".format(image_path,pre_label[0]))
        if 1 == pre_label[0]:
            #score_total += 1
            onsets.append(onset_frames[index])
            onsets_strength[onset_frames[index]] = onsets_frames_strength.get(onset_frames[index])
        else:
            pass
        index +=1
    print("valuation accuracy is {}".format(score_total/len(images_list)))
    sess.close()
    return onsets,onsets_strength

if __name__ ==  '__main__':

    labels_filename = './single_onsest/data/label.txt'
    models_path = './single_onsets/models/alex/model.ckpt-10000'

    # 清空文件夹
    image_dir = './single_onsets/data/test'
    clear_dir(image_dir)
    onset_frames, onsets_frames_strength = get_single_onsets(filename)
    print("onset_frames,onsets_frames_strength is {},{}".format(onset_frames,onsets_frames_strength))
    if onset_frames:
        onsets, onsets_strength = predict(image_dir,onset_frames,onsets_frames_strength,models_path)
        print("onsets, onsets_strength is {},{}".format(onsets, onsets_strength))
        y, sr = librosa.load(filename)
        librosa.display.waveplot(y, sr=sr)
        onsets_time = librosa.frames_to_time(onsets, sr=sr)
        onset_frames_time = librosa.frames_to_time(onset_frames,sr = sr)
        plt.vlines(onsets_time, -1 * np.max(y), np.max(y), color='r', linestyle='solid')
        plt.vlines(onset_frames_time, -1 * np.max(y), np.max(y), color='b', linestyle='dashed')
        plt.show()