import sys
sys.path.append('../code')
import tci
import utils

file_path = "../data/segments-librispeech-1k/clip_0000_WERE.wav"

wf, sr = tci.load_audio(file_path)
print(wf[:,1:10])
utils.print_stats(wf, sample_rate=sr)
utils.plot_waveform(wf, sr)
utils.plot_specgram(wf, sr)


