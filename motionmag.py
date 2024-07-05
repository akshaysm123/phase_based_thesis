import numpy as np
import numpy.fft as npf
import torch
import torch.fft as tft
from video_utils import *
from filters import PyramidFilters
from scipy.signal import firwin
import time

BATCHSIZE = 150
MAG_FACTOR = 35
NUM_BANDS = 10
NUM_ORIENTS = 10
FORCE_CPU = False
TEMPORAL_FILTERING = True
LO_CUTOFF_HZ = 0.3
HI_CUTOFF_HZ = 2
VIDEOPATH = "vids/baby.avi"
OUTPUTFILE = "mag.avi"


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if FORCE_CPU:
    device = torch.device('cpu')
print(f"magnification.py running on: {device}.")

# load video & create filterbank
video = Video(VIDEOPATH)
#video.array = video.array[:200,:,:,:]
filterbank = PyramidFilters(*video.dim, n_bands=NUM_BANDS, n_orients=NUM_ORIENTS)

# FIR filter
if TEMPORAL_FILTERING:
    norm_lo = LO_CUTOFF_HZ / (video.fps * 2)
    norm_hi = HI_CUTOFF_HZ / (video.fps * 2)
    fir_bandpass = firwin(BATCHSIZE, [norm_lo, norm_hi], pass_zero=False)
    fir_bandpass = npf.fft(npf.ifftshift(fir_bandpass))
    fir_bandpass = torch.from_numpy(fir_bandpass).to(device)

# isolate reference frame
frame0 = video.array[0,:,:,:]
video.array = video.array[1:,:,:,:]
magnified = np.zeros_like(video.array)

# phase from pyramid decomposition of reference frame
frame0 = rgb2yiqImage(frame0)[:,:,0]
f0FFT = tft.fftshift(tft.fft2(torch.from_numpy(frame0)))
f0phases = []
for sub in filterbank.subband_filters:
    sub = torch.from_numpy(sub)
    f0phase = torch.angle(tft.ifft2(tft.ifftshift(f0FFT * sub))).unsqueeze(0)
    f0phases.append(f0phase)


start_time = time.perf_counter()
phasetrack = np.array([])

# process frames in batches
nframes = video.array.shape[0]
nbatches = nframes // BATCHSIZE + int(nframes % BATCHSIZE != 0)
for b in range(nbatches):
    print(f"processing batch {b+1} of {nbatches}.")
    vidbatch = video.array[b*BATCHSIZE:(b+1)*BATCHSIZE, :, :, :]
    numframes = vidbatch.shape[0]
    currentResult = torch.zeros(vidbatch.shape, dtype=torch.float32).to(device)

    yiq = torch.from_numpy(rgb2yiqVideo(vidbatch, forceCPU=FORCE_CPU))
    processBatch, iq_ch = separate_ychannel(yiq)
    currentResult[:, :, :, 1:] += iq_ch.to(device)

    # current batch to GPU
    numvals = np.product(processBatch.shape)
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

    for i, sub in enumerate(filterbank.subband_filters):
        #print(f'filter {i+1} / {len(filterbank.subband_filters)}')
        subband = torch.from_numpy(sub).unsqueeze(0).to(device)
        subband = tft.ifft2(tft.ifftshift(processBatch * subband, dim=(1,2)))
        phase = torch.angle(subband)
        refrencePhase = f0phases[i].to(device)
        f0phases[i] = phase[-1,:,:].unsqueeze(0).cpu()
        phase = ((phase - refrencePhase + np.pi) % (2 * np.pi)) - np.pi

        if TEMPORAL_FILTERING:
            if BATCHSIZE != numframes:
                fir_bandpass = firwin(numframes, [norm_lo, norm_hi], pass_zero=False)
                fir_bandpass = npf.fft(npf.ifftshift(fir_bandpass))
                fir_bandpass = torch.from_numpy(fir_bandpass).to(device)

            phase = tft.fft(phase, dim=(0))
            phase *= fir_bandpass[:, None, None]
            phase = tft.ifft(phase, dim=(0)).real

        phase *= MAG_FACTOR
        subband *= torch.exp(1j * phase)
        currentResult[:, :, :, 0] += subband.real
        del subband, phase, refrencePhase
        torch.cuda.empty_cache()

    currentResult = rgb2yiqVideo(currentResult.cpu().numpy(), backward=True, forceCPU=FORCE_CPU)
    magnified[b*BATCHSIZE:(b+1)*BATCHSIZE,:,:,:] = currentResult
    del currentResult
    torch.cuda.empty_cache()

end_time = time.perf_counter()
print(f"Execution time magnification part: {float(end_time - start_time):.4f} seconds")

create_video(magnified, OUTPUTFILE, video.fps)

