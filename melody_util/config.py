contextlen = 15  # +- frames
duration = 2 * contextlen + 1
window_len = duration

def get_config():
    return contextlen,duration,window_len