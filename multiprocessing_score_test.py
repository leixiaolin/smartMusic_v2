# -*- coding:utf-8 -*-

from multiprocessing_score_util import parallel_score_without_pipe
import time

if __name__ == '__main__':
    # parallel_score()
    var = (4, 8, 12, 20, 16)
    t = time.time()
    params_list = [(4,t),(8,t),(12,t),(20,t),(16,t)]
    filename = 'F:/项目/花城音乐项目/样式数据/20.04.08MP3/2段词-标准1648.wav'
    standard_kc = '喜爱春天的人儿是心地纯洁的人像紫罗兰花儿一样是我知心朋友'
    standard_kc_time = [0,1,2,3,3.5,4,5,6,8,9,10,11,11.5,12,16,17,18,19,20,21,22,23,24,25,26,26.5,27,28,32]
    standard_notations = '3,3,2,1,1,7-,6-,6-,6-,4,4,3,2,1,2,4,3,4,4,3,2,2,4,3,3,1,6-,6-,7-,3,2,1,7-,1,6-'
    standard_notation_time = [0,1,1.5,2,3,3.5,4,5,6,8,9,9.5,10,10.5,11,11.5,12,16,17,17.5,18,19,19.5,20,21,21.5,22,23,24,25,26,26.5,27,27.5,28,32]
    params = (filename, standard_kc, standard_kc_time, standard_notations, standard_notation_time)
    params_list = [params for i in range(500)]
    print(params_list)
    parallel_score_without_pipe(params_list)
    t = time.time() - t
    print("speed time is {}".format(t))