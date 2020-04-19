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







































