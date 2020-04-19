from pydub import AudioSegment
import os

def slice_file(base,save_path,file, start,end):
    audio = AudioSegment.from_wav(os.path.join(base, file))
    audio[start * 1000:end * 1000].export(save_path + file, format="wav")

if __name__ == '__main__':
    base = "F:/项目/花城音乐项目/样式数据/20.03.26MP3/wav/"
    base = "F:/项目/花城音乐项目/样式数据/20.04.01MP3/"
    base = "F:/项目/花城音乐项目/样式数据/20.04.08MP3/"
    save_path = "F:/项目/花城音乐项目/样式数据/20.03.26MP3/wav/slice/"
    save_path = "F:/项目/花城音乐项目/样式数据/20.04.08MP3/slice/"
    filename = '人声打分.wav'
    filename = '乐器演奏打分.wav'
    filename = '2段词-标准.wav'
    start =16
    end =40
    slice_file(base, save_path, filename, start, end)