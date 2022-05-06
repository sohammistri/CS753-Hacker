import torch
import torch.nn.functional as F
import torchaudio
import numpy as np
import os

SEGMENT_DURS = [20, 40, 60, 80, 100, 120, 140, 160, 200, 240, 280, 340, 400, 480, 580, 700, 840, 1000, 1200, 1440, 1720, 2060, 2480]

def load_audio(audio_dir):
    waveforms, sample_rates = zip(*[torchaudio.load(os.path.join(audio_dir,file), channels_first=False) for file in os.listdir(audio_dir)])
    waveforms = [torch.as_tensor(x, dtype=float) for x in waveforms]
    waveforms = torch.stack(waveforms, dim=0)
    return waveforms, sample_rates


def extract_central_segment(waveform, sample_rate, seg_dur, overlap_dur=20):
    seg_len = np.round(sample_rate*seg_dur/1000)
    overlap_len = np.round(sample_rate*overlap_dur/1000)
    p2d = (0,0,np.floor(overlap_len/2),np.ceil(overlap_len/2))
    waveform = F.pad(waveform, p2d)
    extra = len(waveform)-seg_len-overlap_len
    if extra>0:
        central_seg = waveform[np.floor(extra/2):-np.ceil(extra/2)]
    else:
        central_seg = waveform
    return central_seg


def crossfade(segments, sample_rates, overlap_dur=20):
    pieces = []
    for i in range(len(segments)):
        overlap_len = np.round(sample_rates[i]*overlap_dur/1000)
        window = torch.hann_window(2*overlap_len + 3)[overlap_len+2:-1].reshape([-1]+[1]*(segments[i].ndim-1))

        if i > 0:
            pieces.append(segments[i-1][len(segments[i-1])-overlap_len:]*window + segments[i][:overlap_len]*window.flip(0))
        if i == 0:
            pieces.append(segments[i][overlap_len//2:len(segments[i])-overlap_len])
        elif i == len(segments)-1:
            pieces.append(segments[i][overlap_len:len(segments[i])-overlap_len//2])
        else:
            pieces.append(segments[i][overlap_len:len(segments[i])-overlap_len])
    
    return torch.cat(pieces, dim=0)



def model_output(layer, seg_dur, seq1, seq2, isr, out_sample_rate=50, block_size=48.0, context_size=8.0, device="cpu" ):
    assert len(seq1)==len(seq2)

    seglen_in = np.round(seg_dur * isr/1000)
    seglen_out = np.round(seg_dur * out_sample_rate/1000)
    nbatch = np.floor(block_size / seg_dur)
    ncontx = np.ceil(context_size / seg_dur)
    ntargt = nbatch - ncontx*2
    num_batch = np.ceil(len(seq1)/ ntargt / seglen_in)
    
    response1 = []
    response2 = []
    sequence1 = torch.cat((
        seq1[-ncontx*seglen_in:], seq1, seq1[:ncontx*seglen_in]), dim=0
    )
    sequence2 = torch.cat((
        seq2[-ncontx*seglen_in:], seq2, seq2[:ncontx*seglen_in]), dim=0
    )
    for k in range(num_batch):
        # batch input
        seq_batch1 = sequence1[(k*ntargt)*seglen_in:(k*ntargt+nbatch)*seglen_in].float().to(device)

        # run model inference
        res_batch1 = layer(seq_batch1)[ncontx*seglen_out:-ncontx*seglen_out]
        response1.append(res_batch1)

        seq_batch2 = sequence2[(k*ntargt)*seglen_in:(k*ntargt+nbatch)*seglen_in].float().to(device)

        # run model inference
        res_batch2 = layer(seq_batch2)[ncontx*seglen_out:-ncontx*seglen_out]
        response2.append(res_batch2)
    
    return (torch.cat(response1, dim=0), torch.cat(response2, dim=0))


def rearrange_seq(response, seg_dur, seed, out_sample_rate=50, margin=1.0):
    for sequence in response:
        target_dur = seg_dur
        # reshape sequence into separate segments
        seglen_t = np.round(target_dur * out_sample_rate/1000)
        seglen = np.round(seg_dur * out_sample_rate/1000)
        segments = sequence.reshape((len(sequence)//seglen, seglen, sequence.shape[-1]))

        # extract extra margins around shared segments in case of noncausality, etc.
        nmargn = np.ceil(margin / seg_dur)
        segments = torch.cat([torch.roll(segments, k, 0) for k in range(nmargn, -nmargn-1, -1)], dim=1)
        segments = segments[:, round(nmargn*seg_dur*out_sample_rate-margin*out_sample_rate):round(segments.shape[1]-nmargn*seg_dur*out_sample_rate+margin*out_sample_rate)]

        np.random.seed(seed)
        segments = segments[np.argsort(np.random.permutation(len(segments)))]

    return segments



def main(audio_folder, layer_num, seed1=1, seed2=2):
    waveforms, in_sample_rates = load_audio(audio_folder)
    if len(set(in_sample_rates)==1):
        isr = in_sample_rates[0]

    for dur in SEGMENT_DURS:
        central_seg_list = [extract_central_segment(x, isr, dur) for x in waveforms]
        central_seg_list = torch.stack(central_seg_list, dim=0)
        np.random.seed(seed1)
        central_seg_list = central_seg_list[np.random.permutation(len(central_seg_list))]
        final_segments_1 = crossfade(central_seg_list, in_sample_rates)
        np.random.seed(seed2)
        central_seg_list = central_seg_list[np.random.permutation(len(central_seg_list))]
        final_segments_2 = crossfade(central_seg_list, in_sample_rates)
        model_response1, model_response2 = model_output(layer, dur, final_segments_1, final_segments_2, isr)
        SAR1 = rearrange_seq(model_response1, dur, seed=seed1)
        SAR2 = rearrange_seq(model_response2, dur, seed=seed2)




