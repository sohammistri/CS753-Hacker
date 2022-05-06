from typing import Iterable
import torch
import torch.nn.functional as F
import torchaudio
import numpy as np
import os
import math
from scipy.interpolate import interp1d

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


@torch.no_grad()
def corr(a, b, axis=None):
    """Compute Pearson's correlation along specified axis."""
    a_mean = a.mean(axis=axis, keepdims=True)
    b_mean = b.mean(axis=axis, keepdims=True)
    a, b = (a - a_mean), (b - b_mean)
    
    a_sum2 = (a ** 2).sum(axis=axis, keepdims=True)
    b_sum2 = (b ** 2).sum(axis=axis, keepdims=True)
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        a, b = (a / np.sqrt(a_sum2)), (b / np.sqrt(b_sum2))
    elif isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        a, b = (a / torch.sqrt(a_sum2)), (b / torch.sqrt(b_sum2))
    else:
        raise TypeError(f'Incompatible types: {type(a)} and {type(b)}')
    
    rvals = (a * b).sum(axis=axis)
    rvals[~torch.isfinite(rvals)] = 0

    return rvals


@torch.no_grad()
def batch_corr(seq_A, seq_B, batch_size=None, device='cpu'):
    """
    Compute correlation of `seq_A` and `seq_B` along the first axis, performed in batches if `batch_size` is
    set. Use batch processing if the large number of segments is causing memory issues.
    seq_A: input sequence A.
    seq_B: input sequence B.
    """
    if batch_size:
        return torch.cat([
            corr(
                seq_A[:, k*batch_size:(k+1)*batch_size].to(device),
                seq_B[:, k*batch_size:(k+1)*batch_size].to(device),
                axis=0
            ).cpu() for k in range(math.ceil(seq_A.shape[1] / batch_size))
        ], axis=0).numpy()
    else:
        return corr(
            seq_A.to(device),
            seq_B.to(device),
            axis=0
        ).cpu().numpy()


@torch.no_grad()
def cross_context_corrs(sequence_pair, batch_size=None, device='cpu'):
    return [
            batch_corr(
                seq_A,
                seq_B,
                batch_size=batch_size,
                device=device
            ) for seq_A, seq_B in sequence_pair
        ]

@torch.no_grad()
def estimate_integration_window(cc_corrs, segment_durs, threshold=0.75):
    """
    Find when correlations (`cc_corrs`) cross the specified `threshold`.
    corrs: correlation matrix for all segment durations, with shape [segments x channels]
    """
    if not isinstance(segment_durs, Iterable):
        cc_corrs = [cc_corrs]
        segment_durs = [segment_durs]
    
    cc_corrs = np.stack([np.nanmax(c, axis=0) for c in cc_corrs], axis=0)

    seglens = np.log(segment_durs)
    x_intrp = np.linspace(seglens.min(), seglens.max(), 1000)
    
    integration_windows = np.zeros(cc_corrs.shape[1])
    for j in range(cc_corrs.shape[1]):
        y_intrp = interp1d(
            seglens,
            np.convolve(np.pad(cc_corrs[:, j], [(1, 1)], 'edge'), [0.15, 0.7, 0.15], 'valid')
        )(x_intrp)
        
        passthresh = np.where(y_intrp >= threshold)[0]
        integration_windows[j] = round(np.exp(x_intrp[passthresh[0]]), 3) if len(passthresh) > 0 else np.nan
    
    return integration_windows

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
        
        model = DeepSpeech().to("cpu").eval()
        model.load_state_dict(torch.load('resources/deepspeech2-pretrained.ckpt')['state_dict'])
        model_response1, model_response2 = model_output(0, dur, final_segments_1, final_segments_2, isr)
        SAR1 = rearrange_seq(model_response1, dur, seed=seed1)
        SAR2 = rearrange_seq(model_response2, dur, seed=seed2)

        cross_context_corr = cross_context_corrs((SAR1, SAR2), batch_size=100)
        int_window = estimate_integration_window(cross_context_corr, dur, threshold=0.75)

        print(dur, int_window)




