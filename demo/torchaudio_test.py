import sys
sys.path.append('../code')
import tci
import utils
import numpy as np
import torch.nn.functional as F

audio_path = "sample"

wf, sr = tci.load_audio(audio_path)
wf = list(zip(wf,sr))[0][0]
print(wf.size())
p2d = (0,0,10,10)
wf = F.pad(wf, p2d)
print(wf[100:-100])

