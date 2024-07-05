import numpy as np
import numpy.fft as npf
import torch
import torch.fft as tft
from video_utils import *
from batch_index import *
from filters import PyramidFilters
from scipy.signal import firwin
import time

from matplotlib import pyplot as plt

FORCE_CPU = False
BATCHSIZE = 140
CSP_BANDS = 7
CSP_ORIENTS = 7
MAG_FACTOR = 25
TEMPORAL_FILTERING = True
LO_CUTOFF_HZ = 0.3
HI_CUTOFF_HZ = 2
FRAME_OVERLAP = 5
VIDEOPATH = "vids/baby.avi"
OUTPUTFILE = "mag_baby_overlap.avi"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if FORCE_CPU:
    device = torch.device('cpu')
print(f"magnification.py running on: {device}.")

video = Video(VIDEOPATH)
filterbank = PyramidFilters(*video.dim, n_bands=CSP_BANDS, n_orients=CSP_ORIENTS)

if TEMPORAL_FILTERING:
    norm_lo = LO_CUTOFF_HZ / (video.fps * 2)
    norm_hi = HI_CUTOFF_HZ / (video.fps * 2)
    fir_bandpass = firwin(BATCHSIZE, [norm_lo, norm_hi], pass_zero=False)
    fir_bandpass = npf.fft(npf.ifftshift(fir_bandpass))
    fir_bandpass = torch.from_numpy(fir_bandpass).to(device)

frame0 = video.array[0,:,:,:]
video.array = video.array[1:,:,:,:]
magnified = np.zeros_like(video.array)

frame0 = rgb2yiqImage(frame0)[:,:,0]
f0FFT = tft.fftshift(tft.fft2(torch.from_numpy(frame0)))
f0phases = []
for sub in filterbank.subband_filters:
    sub = torch.from_numpy(sub)
    f0phase = torch.angle(tft.ifft2(tft.ifftshift(f0FFT * sub))).unsqueeze(0)
    f0phases.append(f0phase)

start_time = time.perf_counter()
batchIDXS = batch_organizer(BATCHSIZE, video.array.shape[0], FRAME_OVERLAP)


phasetrack = np.array([])


for iB, bIDX in enumerate(batchIDXS):
    print(f"processing batch {iB+1} of {len(batchIDXS)}.")
    vidbatch = video.array[bIDX.start:bIDX.end, :, :, :]
    numframes = vidbatch.shape[0]
    currentResult = torch.zeros(vidbatch.shape, dtype=torch.float32).to(device)

    yiq = torch.from_numpy(rgb2yiqVideo(vidbatch, forceCPU=FORCE_CPU))
    processBatch, iq_ch = separate_ychannel(yiq)
    currentResult[:, :, :, 1:] += iq_ch.to(device)

    processBatch = processBatch.to(torch.complex64).to(device)
    processBatch = tft.fftshift(tft.fft2(processBatch, dim=(1,2)))

    # high and lowpass not processed
    # highpass = torch.from_numpy(filterbank.highpass).unsqueeze(0).to(device)
    # highpass = tft.ifft2(tft.ifftshift(processBatch * highpass, dim=(1,2)))
    # currentResult[:, :, :, 0] += highpass.real

    lowpass = torch.from_numpy(filterbank.lowpass).unsqueeze(0).to(device)
    lowpass = tft.ifft2(tft.ifftshift(processBatch * lowpass, dim=(1,2)))
    currentResult[:, :, :, 0] += lowpass.real

    del vidbatch, yiq, iq_ch, lowpass
    torch.cuda.empty_cache()

    CACHE_AVAILABLE = False
    cached_phase = torch.zeros((len(filterbank.subband_filters), -bIDX.saveCache, 
                                processBatch.shape[-2], processBatch.shape[-1]))
    for iF, sub in enumerate(filterbank.subband_filters):
        print(f'filter {iF+1} / {len(filterbank.subband_filters)}')

        if CACHE_AVAILABLE:
            processBatch = processBatch[bIDX.loadCache:,:,:]

        subband = torch.from_numpy(sub).unsqueeze(0).to(device)
        subband = tft.ifft2(tft.ifftshift(processBatch * subband, dim=(1,2)))
        phase = torch.angle(subband)
        refrencePhase = f0phases[iF].to(device)
        f0phases[iF] = phase[bIDX.vEnd,:,:].unsqueeze(0).cpu()
        phase = ((phase - refrencePhase + np.pi) % (2 * np.pi)) - np.pi
        
        if CACHE_AVAILABLE:
            phase = torch.cat((cached_phase[iF,:,:,:], phase), dim=0)
        cached_phase[iF,:,:,:] = phase[bIDX.saveCache:,:,:]

        if TEMPORAL_FILTERING:
            phase = tft.fft(phase, dim=(0))
            phase *= fir_bandpass[:, None, None]
            phase = tft.ifft(phase, dim=(0)).real

        phase *= MAG_FACTOR
        subband *= torch.exp(1j * phase)
        currentResult[:, :, :, 0] += subband.real
        del subband, phase, refrencePhase
        torch.cuda.empty_cache()

    CACHE_AVAILABLE = True
    currentResult = currentResult[bIDX.vStart:bIDX.vEnd, :, :]
    currentResult = rgb2yiqVideo(currentResult.cpu().numpy(), backward=True, forceCPU=FORCE_CPU)
    nStart = bIDX.start + bIDX.vStart
    nEnd = bIDX.end + bIDX.vEnd
    magnified[nStart:nEnd:,:,:] = currentResult

    del currentResult
    torch.cuda.empty_cache()

end_time = time.perf_counter()
print(f"Execution time magnification part: {float(end_time - start_time):.4f} seconds")
create_video(magnified, OUTPUTFILE, video.fps)