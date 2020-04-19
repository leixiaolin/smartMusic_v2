import urllib.request
import time
import urllib
import json
import hashlib
import base64
from urllib import parse

def main():
    # f = open("F:/项目/花城音乐项目/样式数据/20.04.01MP3/人声打分.wav", 'rb')
    f = open("F:/项目/花城音乐项目/样式数据/6.24MP3/旋律/两只老虎20190624-2939.wav", 'rb')  # rb表示二进制格式只读打开文件
    file_content = f.read()
    base64_audio = base64.b64encode(file_content)
    body = parse.urlencode({'audio': base64_audio})

    url = 'http://api.xfyun.cn/v1/service/v1/iat'
    api_key = '2b05d51544e2f7896d4c539ccd2c8454'
    param = {"engine_type": "sms16k", "aue": "raw"}

    x_appid = '5d5ba58f'
    json_str = json.dumps(param).replace(' ', '')
    print('json_str:{}'.format(json_str))
    x_param = base64.b64encode(bytes(json_str, 'ascii'))
    x_time = int(int(round(time.time() * 1000)) / 1000)
    x_checksum_str = api_key + str(x_time) + str(x_param)[2:-1]
    print('x_checksum_str:[{}]'.format(x_checksum_str))
    x_checksum = hashlib.md5(x_checksum_str.encode(encoding='ascii')).hexdigest()
    print('x_checksum:{}'.format(x_checksum))
    x_header = {'X-Appid': x_appid,
                'X-CurTime': x_time,
                'X-Param': x_param,
                'X-CheckSum': x_checksum}

    start_time = time.time()
    req = urllib.request.Request(url, bytes(body, 'ascii'), x_header)
    result = urllib.request.urlopen(req)
    result = result.read()
    print("used time: {}s".format(round(time.time() - start_time, 2)))
    print('result:'+str(result.decode(encoding='UTF8')))
    return

if __name__ == '__main__':
    main()
