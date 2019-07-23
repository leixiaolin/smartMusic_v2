from myDtw import *

onset_frames = [96, 117, 145, 195, 214, 240, 289, 314, 336, 372,386,422]
base_frames =  [96, 117, 137, 178, 198, 219, 259, 280, 300, 331, 341, 422]
def get_dtw(onset_frames,base_frames):
    euclidean_norm = lambda x, y: np.abs(x - y)
    d, cost_matrix, acc_cost_matrix, path = dtw(onset_frames,base_frames, dist=euclidean_norm)
    #dis = d * np.sum(acc_cost_matrix.shape)
    dis = d
    return dis

dis = get_dtw(np.diff(onset_frames),np.diff(base_frames))
print(dis)