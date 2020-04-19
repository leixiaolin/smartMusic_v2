import json
import time
import requests
import os
import tempfile
import subprocess
import base64
import logging
logger = logging.getLogger(__name__)


def mp3_2_wav(_path = None, _byte = None):
    ''' MP3转WAV
    _path和_byte必须存在一个, 优先级_path > _byte
    :param _path:
    :param _byte:
    :return: wav的字节流
    '''
    try:
        if _path is None and _byte is None: return
        temp = None
        if _path is None: # 字节流存入临时文件
            temp = tempfile.NamedTemporaryFile(mode="w+b", delete=False)
            temp.write(_byte)
            temp.seek(0)
            _path = temp.name
        if _path is None: return
        # 根据要求进行格式转换,-t 60 最大保存60秒, 采样率 16K, 默认单声道
        logger.info('mp3 ==> wav ========================')
        target_file = tempfile.NamedTemporaryFile(mode="w+b", delete=False, suffix='.wav')
        _perfix = r'ffmpeg'
        command = [_perfix, '-y', '-t', '60', '-i', _path, '-ar', '16K', target_file.name]
        return_code = subprocess.call(command)
        logger.info('mp3 ==> wav ==={}====================='.format(return_code))
        if return_code == 0:
            target_file.seek(0)
            _byte = target_file.read()
            target_file.close()
            os.remove(target_file.name)
            if temp is not None:
                temp.close()
                os.remove(temp.name)
            return _byte
    except Exception as e:
        logger.error('mp3_2_wav error [{}]'.format(e))

def XUNFEI_ASR(_path):
    ''' 讯飞语音转文字

    :param _path:
    :return:
    '''
    _byte = mp3_2_wav(_path)
    base64_audio = base64.b64encode(_byte)
    import urllib.parse
    body = urllib.parse.urlencode({'audio': base64_audio})
    url = 'http://api.xfyun.cn/v1/service/v1/iat'
    APP_ID = '5d5ba58f'
    API_KEY = '2b05d51544e2f7896d4c539ccd2c8454'
    param = {"engine_type": "sms16k", "aue": "raw"}
    import hashlib
    x_param = base64.b64encode(json.dumps(param).replace(' ', '').encode('utf-8'))
    x_time = int(int(round(time.time() * 1000)) / 1000)
    _str = API_KEY + str(x_time) + x_param.decode('utf-8')
    x_checksum = hashlib.md5(_str.encode('utf-8')).hexdigest()
    x_header = {'Content-Type': 'application/x-www-form-urlencoded; charset=utf-8',
                'X-Appid': APP_ID,
                'X-CurTime': str(x_time),
                'X-Param': x_param,
                'X-CheckSum': x_checksum}
    res = requests.post(url, body, headers=x_header)
    res = res.content.decode('utf-8')
    print(res)
    return res


if __name__ == '__main__':
    filename = "F:/项目/花城音乐项目/样式数据/6.24MP3/旋律/两只老虎20190624-2939.wav"
    XUNFEI_ASR(filename)


























