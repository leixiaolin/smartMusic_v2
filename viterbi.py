# -*- coding: utf-8 -*-
"""
================
Viterbi decoding
================

This notebook demonstrates how to use Viterbi decoding to impose temporal
smoothing on frame-wise state predictions.

Our working example will be the problem of silence/non-silence detection.
"""

# Code source: Brian McFee
# License: ISC

##################
# Standard imports
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import librosa

import librosa.display
from create_base import *

#############################################
# Load an example signal
#filename = 'F:/项目/花城音乐项目/样式数据/ALL/节奏/节奏八（6）(95).wav'
#filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏2林(20).wav'
filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏1.2(100).wav'
#filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏1-01（75）.wav'
#filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏2（二）(100).wav'
#filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏九（3）（100）.wav'

filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏八（9）（95）.wav'
#filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏1_40441（96）.wav'
#filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏1_40441（96）.wav'
#filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏1语(90).wav'
#filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏1周(95).wav'



def get_viterbi_state(y,silence_threshold):
    #filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/视唱1-02（90）.wav'
    #y, sr = librosa.load(filename)


    # And compute the spectrogram magnitude and phase
    S_full, phase = librosa.magphase(librosa.stft(y))


    ###########################################################
    # As you can see, there are periods of silence and
    # non-silence throughout this recording.
    #

    # As a first step, we can plot the root-mean-square (RMS) curve
    rms = librosa.feature.rmse(y=y)[0]
    #print("max rms is : {}".format(np.max(rms)))
    silence_threshold = np.max(rms)*silence_threshold

    times = librosa.frames_to_time(np.arange(len(rms)))



    # The red line at 0.02 indicates a reasonable threshold for silence detection.
    # However, the RMS curve occasionally dips below the threshold momentarily,
    # and we would prefer the detector to not count these brief dips as silence.
    # This is where the Viterbi algorithm comes in handy!

    #####################################################
    # As a first step, we will convert the raw RMS score
    # into a likelihood (probability) by logistic mapping
    #
    #   :math:`P[V=1 | x] = \frac{\exp(x - \tau)}{1 + \exp(x - \tau)}`
    #
    # where :math:`x` denotes the RMS value and :math:`\tau=0.02` is our threshold.
    # The variable :math:`V` indicates whether the signal is non-silent (1) or silent (0).
    #
    # We'll normalize the RMS by its standard deviation to expand the
    # range of the probability vector

    r_normalized = (rms - silence_threshold) / np.std(rms)
    p = np.exp(r_normalized) / (1 + np.exp(r_normalized))

    # We can plot the probability curve over time:

    #######################################################################
    # which looks much like the first plot, but with the decision threshold
    # shifted to 0.5.  A simple silence detector would classify each frame
    # independently of its neighbors, which would result in the following plot:


    ###############################################
    # We can do better using the Viterbi algorithm.
    # We'll use state 0 to indicate silent, and 1 to indicate non-silent.
    # We'll assume that a silent frame is equally likely to be followed
    # by silence or non-silence, but that non-silence is slightly
    # more likely to be followed by non-silence.
    # This is accomplished by building a self-loop transition matrix,
    # where `transition[i, j]` is the probability of moving from state
    # `i` to state `j` in the next frame.

    transition = librosa.sequence.transition_loop(2, [0.5, 0.6])
    #print(transition)

    #####################################################################
    # Our `p` variable only indicates the probability of non-silence,
    # so we need to also compute the probability of silence as its complement.

    full_p = np.vstack([1 - p, p])
    #print(full_p)

    ####################################
    # Now, we're ready to decode!
    # We'll use `viterbi_discriminative` here, since the inputs are
    # state likelihoods conditional on data (in our case, data is rms).

    states = librosa.sequence.viterbi_discriminative(full_p, transition)

    return times,states
def check_need_vocal_separation(y,silence_threshold):
    times, states = get_viterbi_state(y, silence_threshold)
    states_diff = np.diff(states)
    result = [i for i, x in enumerate(states_diff) if x == 1]
    size = len(result)
    print("len result is {}".format(len(result)))
    if size <= 3:
        return True
    else:
        return False


if __name__ == '__main__':
    y, sr = librosa.load(filename)
    librosa.display.waveplot(y, sr=sr)
    #onsets_frames, onsets_frames_strength = get_onsets_by_all(y, sr)
    onsets_frames = librosa.onset.onset_detect(y, sr)
    onstm = librosa.frames_to_time(onsets_frames, sr=sr)
    print("onstm is {}".format(onstm))

    plt.vlines(onstm, -1 * np.max(y), np.max(y), color='b', linestyle='solid')
    plt.tight_layout()

    silence_threshold = 0.2
    y, sr = librosa.load(filename)
    times, states = get_viterbi_state(y,silence_threshold)
    states_diff = np.diff(states)
    result = [i for i,x in enumerate(states_diff) if x == 1]
    print("len result is {}".format(len(result)))
    need_vocal_separation = check_need_vocal_separation(y,silence_threshold)
    print("need_vocal_separation is : {}".format(need_vocal_separation))
    onstm = librosa.frames_to_time(result, sr=sr)
    plt.vlines(onstm, -1 * np.max(y), np.max(y), color='r', linestyle='solid')
    #plt.figure(figsize=(12, 6))
    #plt.step(times, p>=0.5, label='Frame-wise')
    plt.step(times, states*np.max(y),0,np.max(y), linestyle='--', color='orange')
    plt.xlabel('Time')
    plt.axis('tight')
    plt.ylim([0, np.max(y)*1.2])
    plt.legend()
    plt.show()


#########################################################################
# Note how the Viterbi output has fewer state changes than the frame-wise
# predictor, and it is less sensitive to momentary dips in energy.
# This is controlled directly by the transition matrix.
# A higher self-transition probability means that the decoder is less
# likely to change states.
