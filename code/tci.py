import torch
import torch.nn.functional as F
import torchaudio
import numpy as np
import os

SEGMENT_DURS = [20, 40, 60, 80, 100, 120, 140, 160, 200, 240, 280, 340, 400, 480, 580, 700, 840, 1000, 1200, 1440, 1720, 2060, 2480]
np.random.seed(42)

def load_audio(file_path):
    waveform, sample_rate = torchaudio.load(file_path)
    waveform = torch.as_tensor(waveform, dtype=float)
    return waveform, sample_rate


def extract_central_segment(waveform, sample_rate, seg_dur, overlap_dur=20):
    seg_len = np.round(sample_rate*seg_dur/1000)
    overlap_len = np.round(sample_rate*overlap_dur/1000)
    p2d = (0,0,np.floor(overlap_len/2),np.ceil(overlap_len/2))
    waveform = F.pad(waveform, p2d)
    wf_len = waveform.size(dim=1)
    extra = wf_len-seg_len-overlap_len
    central_seg = wf_len[:,np.floor(extra/2):-np.ceil(extra/2)]
    return central_seg


def crossfade(segments, sample_rate, overlap_dur=20):
    overlap_len = np.round(sample_rate*overlap_dur)
    segemnt_dim = segments[0].size(dim=1)
    window = torch.hann_window(2*overlap_len + 3)[overlap_len+2:-1].reshape((1, segemnt_dim))
    
    pieces = []
    for i in range(len(segments)):
        if i > 0:
            pieces.append(segments[i-1][len(segments[i-1])-overlap_len:]*window + segments[i][:overlap_len]*window.flip(0))
        if i == 0:
            pieces.append(segments[i][overlap_len//2:len(segments[i])-overlap_len])
        elif i == len(segments)-1:
            pieces.append(segments[i][overlap_len:len(segments[i])-overlap_len//2])
        else:
            pieces.append(segments[i][overlap_len:len(segments[i])-overlap_len])
    
    return torch.cat(pieces, dim=0)



def model_output(layer, seg_dur, seq1, seq2, in_sample_rate, out_sample_rate=50, block_size=48.0, context_size=8.0, device="cpu" ):
    num_samples = seq1.size(dim=0)
    assert num_samples==seq2.size(dim=0)

    for i in range(num_samples):
        sequence1 = seq1[i,:]
        sequence2 = seq2[i,:]
        assert len(sequence1)==len(sequence2)
        seq_dim = len(sequence1)

        seglen_in = np.round(seg_dur * in_sample_rate)
        seglen_out = np.round(seg_dur * out_sample_rate)
        nbatch = np.floor(block_size / seg_dur)
        ncontx = np.ceil(context_size / seg_dur)
        ntargt = nbatch - ncontx*2
        num_batch = np.ceil(seq_dim/ ntargt / seglen_in)
        
        response1 = []
        response2 = []
        sequence1 = torch.cat((
            sequence1[-ncontx*seglen_in:], sequence1, sequence1[:ncontx*seglen_in]), dim=0
        )
        sequence2 = torch.cat((
            sequence2[-ncontx*seglen_in:], sequence2, sequence2[:ncontx*seglen_in]), dim=0
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


def restore_dur(response, seg_dur, out_sample_rate=50, margin=1.0):
    for sequence in response:
        target_dur = seg_dur
        # reshape sequence into separate segments
        seglen_t = np.round(target_dur * out_sample_rate)
        seglen = round(seg_dur * out_sample_rate)
        segments = sequence.reshape((len(sequence)//seglen, seglen, sequence.shape[-1]))

        # extract extra margins around shared segments in case of noncausality, etc.
        nmargn = np.ceil(margin / seg_dur)
        segments = torch.cat([torch.roll(segments, k, 0) for k in range(nmargn, -nmargn-1, -1)], dim=1)
        segments = segments[:, round(nmargn*seg_dur*out_sample_rate-margin*out_sample_rate):round(segments.shape[1]-nmargn*seg_dur*out_sample_rate+margin*out_sample_rate)]

        # if natural-random comparison, extract relevant part of natural segment response
        if seed < 0:
            discard = segments.shape[1] - seglen_t - 2*round(margin * out_sample_rate)
            if discard > 0:
                if segment_alignment == 'center':
                    segments = segments[:, math.floor(discard/2):-math.ceil(discard/2)]
                elif segment_alignment == 'start':
                    segments = segments[:, :-discard]
                elif segment_alignment == 'end':
                    segments = segments[:, discard:]
                else:
                    raise NotImplementedError()
                    # x = x[:, ...]
            elif discard < 0:
                raise RuntimeError('All segment time-series must be at least as long as the target segment duration.')

        if seed > 0:
            np.random.seed(seed)
            segments = segments[np.argsort(np.random.permutation(len(segments)))]

    return segments



def main(audio_folder, layer_num):
    waveform_list = []
    for f in os.listdir(audio_folder):
        waveform, in_sample_rate = load_audio(os.path.join(audio_folder, f))
        waveform_list.append(waveform)

    for dur in SEGMENT_DURS:
        central_seg_list = [extract_central_segment(x, in_sample_rate, dur) for x in waveform_list]
        central_seg_list = central_seg_list[np.random.permutation(len(central_seg_list))]
        final_segments_1 = crossfade(central_seg_list, in_sample_rate)
        central_seg_list = central_seg_list[np.random.permutation(len(central_seg_list))]
        final_segments_2 = crossfade(central_seg_list, in_sample_rate)
        model_response1, model_response2 = model_output(layer, dur, final_segments_1, final_segments_2, in_sample_rate)
        SAR1 = restore_dur(model_response1, dur)
        SAR2 = restore_dur(model_response2, dur)




