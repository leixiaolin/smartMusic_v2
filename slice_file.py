from pydub import AudioSegment
import os

def slice_file(base,save_path,file, start,end):
    audio = AudioSegment.from_mp3(os.path.join(base, file))
    audio[start * 1000:end * 1000].export(save_path + file, format="wav")

if __name__ == '__main__':
    base = "F:/项目/花城音乐项目/样式数据/20.03.26MP3/wav/"
    save_path = "F:/项目/花城音乐项目/样式数据/20.03.26MP3/wav/slice/"
    filename = 'LMING1.wav'
    start = 56
    end = 87
    slice_file(base, save_path, filename, start, end)