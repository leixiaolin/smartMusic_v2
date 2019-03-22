#coding=utf-8

import tensorflow as tf
import numpy as np
import pdb
import cv2
import os
import glob
import slim.nets.alexnet as alaxnet

from create_tf_record import *
import tensorflow.contrib.slim as slim

'''
predict an image
'''
def  predict(models_path,image_path,labels_filename,labels_nums, data_format):
    #[batch_size, resize_height, resize_width, depths] = data_format
    tf.reset_default_graph()
    #labels = np.loadtxt(labels_filename, str, delimiter='\t')
    input_images = tf.placeholder(dtype=tf.float32, shape=[None, resize_height, resize_width, depths], name='input')

    with slim.arg_scope(alaxnet.alexnet_v2_arg_scope()):
        out, end_points = alaxnet.alexnet_v2(inputs=input_images, num_classes=labels_nums, dropout_keep_prob=1.0, is_training=False)

    # 将输出结果进行softmax分布,再求最大概率所属类别
    score = tf.nn.softmax(out,name='pre')
    class_id = tf.argmax(score, 1)

    #
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, models_path)



    im=read_image(image_path,resize_height,resize_width,normalization=True)
    im=im[np.newaxis,:]
    pre_score,pre_label = sess.run([score,class_id], feed_dict={input_images:im})

    sess.close()

    return pre_label[0]

if __name__ == '__main__':

    class_nums=2
    labels_filename='./data/label.txt'
    models_path='./models/alex/model.ckpt-10000'
    #models_path = './models/alex/best_models_2000_0.9635.ckpt'
    batch_size = 1
    resize_height = 224  # 指定存储图片高度
    resize_width = 224  # 指定存储图片宽度
    depths=3
    data_format=[batch_size,resize_height,resize_width,depths]
    '''
    分别批量测试
    '''
    image_dir = './data/test/test/yes/' # 选择要测试的目录
    label = 1  # yes是1 no是0
    #image_path = image_dir + '130.png'
    image_list = os.listdir(image_dir)
    error_list = []
    score = 0
    for image_name in image_list:
        image_path = image_dir + image_name
        pre_label = predict(models_path,image_path, labels_filename, class_nums, data_format)
        if pre_label == label:
            score += 1
        else:
            error_list.append(image_name)
    acc = score / len(image_list)
    print('accuracy is {}'.format(acc))
    for i in error_list:
        print(i)
