# -*- coding:utf-8 -*-

from parselmouth_util import score_all
import multiprocessing
import time


def score_for_each(params):
    (file_path, standard_kc, standard_kc_time, standard_notations, standard_notation_time) = params
    print("{},{},{},{},{}".format(file_path, standard_kc, standard_kc_time, standard_notations, standard_notation_time))
    total_score, pitch_total_score, notation_duration_total_score, kc_duration_total_score, kc_express_total_score, fluency_total_score, pitch_score_detail, notation_duration_score_detail, kc_rhythm_sscore_detail, kc_express_sscore_detail, fluency_sscore_detail = score_all(
        file_path, standard_kc, standard_kc_time, standard_notations, standard_notation_time)
    return total_score, pitch_total_score, notation_duration_total_score, kc_duration_total_score, kc_express_total_score, fluency_total_score

def parallel_score_without_pipe(params_list):
    # var = (4, 8, 12, 20, 16)
    p = multiprocessing.Pool()
    t0 = time.time()
    res = p.map_async(score_for_each, params_list).get()
    p.close()
    p.join()

    t = time.time() - t0
    # print('factorial of %s@%.2fs: %s' % (params_list, t, res))
    print(res)
