import parselmouth

import numpy as np
import matplotlib.pyplot as plt
import librosa

FREQS = [
    ('B0',30.87), ('C1',32.7), ('C#1',34.65),
    ('D1',36.71), ('D#1',38.89), ('E1',41.2),
    ('F1',43.65), ('F#1',46.35), ('G1',49),
    ('G#1',51.91), ('A1',55), ('A#1',58.27),
    ('B1',61.74), ('C2',65.41), ('C#2',69.3),
    ('D2',73.42), ('D#2',77.78), ('E2',82.41),
    ('F2',87.31), ('F#2',92.50), ('G2',98.00),
    ('G#2',103.83), ('A2',110.00), ('A#2',116.54),
    ('B2',123.54), ('C3',130.81), ('C#3',138.59),
    ('D3',146.83), ('D#3',155.56), ('E3',164.81),
    ('F3',174.61), ('F#3',185.00), ('G3',196.00),
    ('G#3',207.65), ('A3',220.00), ('A#3',233.08),
    ('B3',246.94), ('C4',261.63), ('C#4',277.18),
    ('D4',293.66), ('D#4',311.13), ('E4',329.63),
    ('F4',349.23), ('F#4',369.99), ('G4',392.00),
    ('G#4',415.30), ('A4' ,440.00), ('A#4',466.16),
    ('B4',493.88), ('C5',523.25), ('C#5',554.37),
    ('D5',587.33), ('D#5',622.25), ('E5',659.26),
    ('F5',698.46), ('F#5',739.99), ('G5',783.99),
    ('G#5',830.61), ('A5',880,00), ('A#5',932.33),
    ('B5',987.77), ('C6',1046.50), ('C#6',1108.73),
    ('D6',1174.66), ('D#6',1244.51), ('E6',1318.51),
    ('F6',1396.91), ('F#6',1479.98), ('G6',1567.98),
    ('G#6',1661.22), ('A6',1760.00), ('A#6',1864.66),
    ('B6',1975.53), ('C7',2093), ('C#7',2217.46),
    ('D7',2349.32), ('D#7',2489.02), ('E7',2637.03),
    ('F7',2793.83), ('F#7',2959.96), ('G7',3135.44),
    ('G#7',3322.44), ('A7',3520), ('A#7',3729.31),
    ('B7',3951.07)
]

def get_numbered_musical_notation(str):
    if str.find("C") >= 0:
        result = "1"
    elif str.find("D") >= 0:
        result = "2"
    elif str.find("E") >= 0:
        result = "3"
    elif str.find("F") >= 0:
        result = "4"
    elif str.find("G") >= 0:
        result = "5"
    elif str.find("A") >= 0:
        result = "6"
    elif str.find("B") >= 0:
        result = "7"

    if str.find("#") >= 0:
        return result + "#"
    else:
        return result

def draw_pitch(pitch,draw_type=1,filename='',notation='',grain_size=0):
    # Extract selected pitch contour, and
    # replace unvoiced samples by NaN to not plot
    p_min = 100
    p_max = 300
    pitch_values = pitch.selected_array['frequency']
    select_pitch_values = [p for p in pitch_values if p != 0]
    pitch_values_max = np.max(select_pitch_values)
    pitch_values_mean = np.mean(select_pitch_values)
    pitch_values = pitch_values + 60   #平移操作
    p_min = np.min(pitch_values) - 30 if np.min(pitch_values) - 30 > 80 else 80
    p_min = int(p_min)
    p_max = np.max(pitch_values) + 30
    p_max = int(p_max)

    #防止个别偏离现象
    if pitch_values_max - pitch_values_mean > 100:
        p_min = int(pitch_values_mean * 0.5)
        p_max = int(pitch_values_mean * 1.5)
    pitch_values[pitch_values==0] = np.nan
    if draw_type == 1:
        plt.plot(pitch.xs(), pitch_values, 'o', markersize=5, color='w')
        plt.plot(pitch.xs(), pitch_values, 'o', markersize=2)
    else:
        if grain_size == 1:
            freqs = FREQS
        else:
            freqs =  [tup for tup in FREQS if tup[0].find('#') < 0]
        freqs_points = [tup[1] for tup in freqs]
        # freqs_points = [tup[1] for tup in FREQS if tup[0].find('#') < 0]
        pitch_values_candidate = []
        for p in pitch_values:
            gaps = [np.abs(f - p) for f in freqs_points]
            gap_min = np.min(gaps)
            if np.isnan(gap_min):
                pitch_values_candidate.append(np.nan)
            else:
                p = gaps.index(gap_min)
                pitch_values_candidate.append(freqs_points[p])
        plt.plot(pitch.xs(), pitch_values, 'o', markersize=2)
        plt.plot(pitch.xs(), pitch_values_candidate, 'o', markersize=2)
    plt.grid(False)
    plt.title(filename, fontsize=16)
    # plt.ylim(0, pitch.ceiling)
    pitch_all = [p for p in freqs_points if p > p_min and p < p_max]
    plt.hlines(pitch_all, 0, len(pitch_values), color = '0.2', linewidth=1, linestyle=":")
    plt.ylim(p_min, p_max)
    plt.ylabel("fundamental frequency [Hz]")
    plt.xlabel(notation)
    pitch_name = [tup[0] for tup in freqs if tup[1] > p_min and tup[1] < p_max]
    for i,p in enumerate(pitch_all):
        numbered_musical_notation = get_numbered_musical_notation(pitch_name[i])
        plt.text(0.1, p, pitch_name[i] + " - " + numbered_musical_notation,size='8')

    # plt.xlim([snd.xmin, snd.xmax])
    return plt

def draw_spectrogram(spectrogram, dynamic_range=70):
    X, Y = spectrogram.x_grid(), spectrogram.y_grid()
    sg_db = 10 * np.log10(spectrogram.values)
    plt.pcolormesh(X, Y, sg_db, vmin=sg_db.max() - dynamic_range, cmap='afmhot')
    plt.ylim([spectrogram.ymin, spectrogram.ymax])
    plt.xlabel("time [s]")
    plt.ylabel("frequency [Hz]")

def draw_intensity(intensity):
    plt.plot(intensity.xs(), intensity.values.T, linewidth=3, color='w')
    plt.plot(intensity.xs(), intensity.values.T, linewidth=1)
    plt.grid(False)
    plt.ylim(0)
    plt.ylabel("intensity [dB]")

def get_mean_pitch(start_frame,end_frame,sr,pitch):
    #将起始帧和结束帧换算成时间点
    onset_times = librosa.frames_to_time([start_frame,end_frame], sr=sr)

    if onset_times[0] > pitch.duration or onset_times[1] > pitch.duration:
        print("the parameter is wrong")
        return None

    #获得总帧数
    frames_total = int(np.floor((pitch.duration - pitch.t1) / pitch.dt) - 1)

    #librosa时间点换算成parselmouth的帧所在位置
    ps_frame = int(onset_times[0] * frames_total / pitch.duration) + 1
    pe_frame = int(onset_times[1] * frames_total / pitch.duration) + 1
    pe_frame = pe_frame if pe_frame < frames_total -1 else frames_total -1 # 防止越界

    pitch_values = pitch.selected_array['frequency']
    pitch_tmp = pitch_values[ps_frame:pe_frame]
    mean_pitch = np.median(pitch_tmp)
    return  mean_pitch

def get_pitch_by_parselmouth(filename):
    snd = parselmouth.Sound(filename)
    pitch = snd.to_pitch()
    return pitch

def draw_intensity(intensity):
    # plt.plot(intensity.xs(), intensity.values.T, linewidth=3, color='w')
    intensity_values_t = intensity.values.T - 50
    plt.plot(intensity.xs(), intensity_values_t, linewidth=1, color='r')
    plt.grid(False)
    plt.ylim(0)
    plt.ylabel("intensity [dB]")