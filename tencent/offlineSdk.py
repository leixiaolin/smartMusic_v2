# -*- coding:utf-8 -*-

import requests
import hashlib
import time
import hmac
import base64
import urllib
from tencent.Config import Config

def task_process(audio_url):
    request_data = dict()
    request_data['channel_num'] = Config.config.CHANNEL_NUM
    request_data['secretid'] = Config.config.SECRETID
    request_data['engine_model_type'] = Config.config.ENGINE_MODEL_TYPE
    request_data['timestamp'] = int(time.time())
    request_data['expired'] = int(time.time()) + 3600
    request_data['nonce'] = 6666
    request_data['projectid'] = Config.config.PROJECT_ID
    request_data['callback_url'] = Config.config.CALLBACK_URL
    request_data['res_text_format'] = Config.config.RES_TEXT_FORMAT
    request_data['res_type'] = Config.config.RES_TYPE
    request_data['source_type'] = Config.config.SOURCE_TYPE
    request_data['sub_service_type'] = Config.config.SUB_SERVICE_TYPE

    request_data['url'] = urllib.parse.quote(audio_url)
    authorization = generate_sign(request_data)
    task_req_url = generate_request(request_data)
    header = {
        "Content-Type": "application/json",
        "Authorization": str(authorization)
    }
    r = requests.post(task_req_url, headers=header, data=request_data)
    # print(r.text)
    return r.text


def generate_sign(request_data):
    sign_str = "POST" + Config.config.SIGN_URL + str(Config.config.APPID) + "?"
    sort_dict = sorted(request_data.keys())
    for key in sort_dict:
        sign_str = sign_str + key + "=" + urllib.parse.unquote(str(request_data[key])) + '&'
    sign_str = sign_str[:-1]
    authorization = base64.b64encode(hmac.new(Config.config.SECRET_KEY.encode('utf-8'), sign_str.encode('utf-8'), hashlib.sha1).digest())
    return authorization.decode('utf-8')


def generate_request(request_data):
    result_url = Config.config.REQUEST_URL + str(Config.config.APPID) + "?"
    for key in request_data:
        result_url = result_url + key + "=" + str(request_data[key]) + '&'
    result_url = result_url[:-1]
    return result_url

if __name__ == '__main__':
    # 语音 URL，公网可下载。当 source_type值为 0时须填写该字段，为 1时不填；长度大于 0，小于 2048
    audio_url = "https://xuhai2-1255824371.cos.ap-chengdu.myqcloud.com/test.wav"
    task_process(audio_url)
