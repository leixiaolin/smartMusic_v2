import librosa
import matplotlib.pyplot as plt
import numpy as np


#filename = 'F:/项目/花城音乐项目/样式数据/ALL/节奏/节奏八（6）(95).wav'
#filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏2林(20).wav'
filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏1.2(100).wav'
filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏1-01（75）.wav'
filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏6林(70).wav'
filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏8_40434（30）.wav'
filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏5_40240（30）.wav'
filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏2（四）(90).wav'

#filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏7_40405（25）.wav'
y, sr = librosa.load(filename)


# And compute the spectrogram magnitude and phase
# stft 短时傅立叶变换
S_full, phase = librosa.magphase(librosa.stft(y))
a = librosa.stft(y)
length = len(a)

S_filter = librosa.decompose.nn_filter(S_full,
                                       aggregate=np.median,
                                       metric='cosine',
                                       width=int(librosa.time_to_frames(1, sr=sr)))

# The output of the filter shouldn't be greater than the input
# if we assume signals are additive.  Taking the pointwise minimium
# with the input spectrum forces this.
S_filter = np.minimum(S_full, S_filter)

# We can also use a margin to reduce bleed between the vocals and instrumentation masks.
# Note: the margins need not be equal for foreground and background separation
margin_i, margin_v = 10, 20
power = 8

mask_i = librosa.util.softmask(S_filter,
                               margin_i * (S_full - S_filter),
                               power=power)

mask_v = librosa.util.softmask(S_full - S_filter,
                               margin_v * S_filter,
                               power=power)

# Once we have the masks, simply multiply them with the input spectrum
# to separate the components

S_foreground = mask_v * S_full
S_background = mask_i * S_full

b = librosa.istft(S_foreground)

librosa.output.write_wav("f:/stft.wav", b, sr)

# 以下是显示频谱图
fig = plt.figure()
s1 = fig.add_subplot(3, 1, 1)
s2 = fig.add_subplot(3, 1, 2)
s3 = fig.add_subplot(3, 1, 3)

s1.plot(y)
s2.plot(a)
s3.plot(b)

plt.show()