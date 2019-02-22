import  librosa

y,sr = librosa.load('F:/项目/花城音乐项目/样式数据/ALL/旋律/1.31MP3/旋律1.100分.wav')
# 通过改变采样率来改变音速，相当于播放速度X2
sr_shift = int(sr*0.8)
librosa.output.write_wav('F:/shift1.wav',y,sr_shift)

# 通过移动音调变声，14是上移14个半步，如果是-14，则是下移14个半步
b = librosa.effects.pitch_shift(y,sr,n_steps=-4)
librosa.output.write_wav('F:/shift2.wav',b,sr)