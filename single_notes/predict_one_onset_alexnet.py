#coding=utf-8

# 让代码仅仅运行在CPU下,一定要放在import tensorflow或keras等之前，否则不起作用。去掉则会在gpu在运行
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  #意为使用cpu

# 设置路径
# 获取指定路径下的文件
dirs = os.path.split(os.path.realpath(__file__))[0]
print(dirs)

import slim.nets.alexnet as alaxnet

from single_notes.create_tf_record import *
import tensorflow.contrib.slim as slim

'''
1、将训练样本放到/data/train里面的yes/no目录中；
2、运行changeFileName，规整化文件名称；
3、运行moveFiles，将样本分为训练集和验证集；
4、运行create_labels_files，生成标签文件；
5、运行create_tf_record，生成tf数据；
6、运行alexnet_train_val，开始训练网络；
7、运行predict_one_onset_alexnet，进行预测
predict an image
'''
def  predict(models_path,image_path,labels_filename,labels_nums, data_format):
    #[batch_size, resize_height, resize_width, depths] = data_format
    resize_height = 224  # 指定存储图片高度
    resize_width = 224  # 指定存储图片宽度
    depths = 3
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
    # sess.list_devices()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, models_path)



    im=read_image(image_path,resize_height,resize_width,normalization=True)
    im=im[np.newaxis,:]
    pre_score,pre_label = sess.run([score,class_id], feed_dict={input_images:im})

    sess.close()

    return pre_label[0]

def get_starts_by_alexnet(filename, rhythm_code, savepath = dirs + '/data/test/'):
    # init_data(filename, rhythm_code, savepath)  # 切分潜在的节拍点，并且保存切分的结果

    class_nums = 2
    labels_filename = dirs + '/data/label.txt'
    models_path = dirs + '/models/alex/model.ckpt-10000'
    # models_path = './models/alex/best_models_2000_0.9635.ckpt'
    batch_size = 1
    resize_height = 224  # 指定存储图片高度
    resize_width = 224  # 指定存储图片宽度
    depths = 3
    data_format = [batch_size, resize_height, resize_width, depths]
    '''
    yes和no分开批量测试
    '''
    image_dir = savepath  # 选择要测试的目录
    label = 1  # yes是1 no是0
    # image_path = image_dir + '130.png'
    image_list = os.listdir(image_dir)
    error_list = []
    correct_list = []

    for image_name in image_list:
        image_path = image_dir + image_name
        image_path = image_path.replace("\\",os.altsep).replace("/",os.altsep)
        # print(image_path)
        pre_label = predict(models_path, image_path, labels_filename, class_nums, data_format)
        if pre_label == label:
            correct_list.append(int(image_name.split('.jpg')[0]))
        else:
            error_list.append(image_name)
    correct_list.sort()

    onset_frames = correct_list
    # print("onset_frames is {}, size {}".format(onset_frames,len(onset_frames)))

    onset_frames_by_overage = get_starts_by_overage(onset_frames)

    return onset_frames,onset_frames_by_overage


def get_starts_by_overage(onset_frames):
    select_starts = []
    tmp = onset_frames.copy()
    tmp.append(0)
    tmp.sort()
    tmp_diff = np.diff(tmp)

    last = 0
    for i in range(0,len(tmp_diff)):
        if onset_frames[i] - last >= 5:
            if i == 0:
                if tmp_diff[i] >= 5 and tmp_diff[i + 1] >= 5:
                    select_starts.append(onset_frames[i])
                    last = onset_frames[i]
            elif i < len(tmp_diff)-1:
                if tmp_diff[i-1] > tmp_diff[i] and tmp_diff[i+1] >= tmp_diff[i]:
                    select_starts.append(onset_frames[i])
                    last = onset_frames[i]
                elif tmp_diff[i] >=5 and tmp_diff[i+1] >= 5:
                    select_starts.append(onset_frames[i])
                    last = onset_frames[i]
            else:
                if tmp_diff[i] > 10:
                    select_starts.append(onset_frames[i])
                    last = onset_frames[i]
    return select_starts
    # select_starts.append(onset_frames[0])
    #
    # for i in range(1,len(onset_frames)):
    #     if onset_frames[i] - onset_frames[i-1] <= 6:
    #         select_starts[-1] = int((select_starts[-1] + onset_frames[i])/2)
    #     else:
    #         select_starts.append(onset_frames[i])
    # return select_starts