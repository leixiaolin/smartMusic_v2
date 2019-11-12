# -*- coding:utf-8 -*-
import tencent.offlineSdk as offlineSdk
import tencent.Config as Config

# 说明：请先将Config.py中的配置项按需改成自己的值，然后再开始使用。

# 音频文件路径。每调用一次task_process方法，可发出一份请求。
# 语音 URL，公网可下载。当 source_type值为 0时须填写该字段，为 1时不填；长度大于 0，小于 2048
audio_url = "https://xuhai2-1255824371.cos.ap-chengdu.myqcloud.com/test.wav"
# 调用语音识别函数获得识别结果
# result = offlineSdk.task_process(audio_url)
# print (result)


# ------------------------------------------------------------------------------------
# 若需中途调整参数值，可直接修改，然后继续发请求即可。比如：
# Config.config.CALLBACK_URL = ""
Config.config.ENGINE_MODEL_TYPE = "8k_0"
Config.config.SOURCE_TYPE = 1
# ......
audio_url = "https://ruskin-1256085166.cos.ap-guangzhou.myqcloud.com/test.wav"
audio_url = "http://qiniu.sgjkzx.com/1212.wav"
audio_url = "http://qiniu.sgjkzx.com/3333.wav"
result = offlineSdk.task_process(audio_url)
print (result)