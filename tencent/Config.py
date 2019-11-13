# -*- coding:utf-8 -*-
'''
Created on 2019-4-28
@author: iantang
'''

class Config:
    '全局变量配置信息，请按需求改成自己的配置'
    
    # ------------- Required,必须填写 ---------------
    # AppId, secretId, secretKey获取方法可参考截图： 
    # https://cloud.tencent.com/document/product/441/6203
    # 具体路径：点控制台右上角您的账号-->选：访问管理-->点左边菜单的：访问秘钥-->API秘钥管理
    SECRET_KEY = 'ZsvlLemeGm224rPghS7I2pMddNsOxU6T'
    SECRETID = 'AKIDFH4h5FHCLQHuwrexNYTFk00RbW3QhR5U'
    APPID = '1251471683'
    # 我们会将识别结果通过Post方式发送至这个URL。用户需要先搭建好自己的用于接收post数据的服务。
    CALLBACK_URL = "http://www.gzzaihua.com/asr"
         
    # ------------- optional，根据自身需求配置值 ---------------
    # 识别引擎 8k_0 or 16k_0 or 8k_6
    ENGINE_MODEL_TYPE = '16k_0'
    # 1 or 2. 语音声道数。在电话 8k通用模型下支持 1和 2，其他模型仅支持 1声道
    CHANNEL_NUM = 1
    # 识别结果文本编码方式 0:UTF-8, 1:GB2312, 2:GBK, 3:BIG5
    RES_TEXT_FORMAT = 0
    # 语音数据来源。0：语音 URL；1：语音数据（post body）
    SOURCE_TYPE = 0
    
    # ------------- optional，采用默认值即可 ---------------
    # 腾讯云项目 ID, 填0。也可改成用户的：控制台-账号中心-项目管理中的配置。
    PROJECT_ID = 0
    # 子服务类型。0：离线语音识别。
    SUB_SERVICE_TYPE = 0
    # 结果返回方式。0：同步返回；1：异步返回。目前只支持异步返回
    RES_TYPE = 1
    # 腾讯服务器的URL，通常无需修改。
    REQUEST_URL = "https://aai.qcloud.com/asr/v1/"
    # 注册签名时用的URL，通常无需修改。
    SIGN_URL = "aai.qcloud.com/asr/v1/"
    
    # ------------- 下面是初始化和验证方法，可跳过 ---------------
    def __init__(self):
        print ("")

    def verifyProperties(self):
        if len(str(self.SECRET_KEY)) == 0:
            print('SECRET_KEY can not empty')
            return
        if len(str(self.SECRETID)) == 0:
            print('SECRETID can not empty')
            return
        if len(str(self.APPID)) == 0:
            print('APPID can not empty')
            return
        if len(str(self.CALLBACK_URL)) == 0:
            print('CALLBACK_URL can not empty')
            return
        
        if len(str(self.ENGINE_MODEL_TYPE)) == 0 or (
            str(self.ENGINE_MODEL_TYPE) != '8k_0' and str(self.ENGINE_MODEL_TYPE) != '16k_0' and str(self.ENGINE_MODEL_TYPE) != '8k_6'):
            print('ENGINE_MODEL_TYPE is not right')
            return
        if len(str(self.CHANNEL_NUM)) == 0 or (str(self.CHANNEL_NUM) != '0' and str(self.CHANNEL_NUM) != '1'):
            print('CHANNEL_NUM is not right')
            return
        if len(str(self.RES_TEXT_FORMAT)) == 0 or (str(self.RES_TEXT_FORMAT) != '0' and str(self.RES_TEXT_FORMAT) != '1' and str(
                self.RES_TEXT_FORMAT) != '2' and str(self.RES_TEXT_FORMAT) != '3'):
            print('RES_TEXT_FORMAT is not right')
            return
        if len(str(self.SOURCE_TYPE)) == 0 or (str(self.SOURCE_TYPE) != '0' and str(self.SOURCE_TYPE) != '1'):
            print('SOURCE_TYPE is not right')
            return
        
        if len(str(self.PROJECT_ID)) == 0:
            print('self.PROJECT_ID can not empty')
            return
        if len(str(self.SUB_SERVICE_TYPE)) == 0 or (str(self.SUB_SERVICE_TYPE) != '0' and str(self.SUB_SERVICE_TYPE) != '1'):
            print('SUB_SERVICE_TYPE is not right')
            return
        if len(str(self.RES_TYPE)) == 0 or (str(self.RES_TYPE) != '0' and str(self.RES_TYPE) != '1'):
            print('RES_TYPE is not right')
            return
        
config = Config()
config.verifyProperties()
