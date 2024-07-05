import numpy as np
import torch
import cv2

class PyramidFilters:

    def __init__(self, height, width, n_bands=5, n_orients=5):
        self.height, self.width = height, width
        self.n_bands = n_bands
        self.n_orients = n_orients

        self.angle, self.radius = self.make_polar_grid()
        self.highpass, self.bandpasses, self.lowpass = self.make_bandpass_filters()
        self.angular_filters = self.make_angular_filters()
        self.subband_filters = self.make_subband_filters()

    # creates the polar mesh grid
    def make_polar_grid(self):
        height, width = self.height, self.width
        long = 1 + ((max(height, width) - min(height, width)) / min(height, width))

        if height > width:
            h_lin = np.linspace(-long, long, height) 
            w_lin = np.linspace(-1, 1, width)
        elif height == width:
            h_lin = np.linspace(-1, 1, height)
            w_lin = np.linspace(-1, 1, width)
        else:
            h_lin = np.linspace(-1, 1, height)
            w_lin = np.linspace(-long, long, width)

        xmesh, ymesh = np.meshgrid(w_lin, h_lin)
        angle = np.arctan2(ymesh, xmesh)
        radius = np.sqrt(xmesh**2 + ymesh**2)

        angle = angle.astype(np.float32)
        radius = radius.astype(np.float32)

        return angle, radius
    

    @staticmethod
    def tanh(rate, zero, x):
        t = np.tanh(rate*(x - zero))
        t = np.clip(t, 0, 1)
        return t
    
    def make_bandpass_filters(self):
        # squared such that the bands get smaller towards the center
        bounds = np.linspace(1, 0.1, self.n_bands+1)**2
        dropoffs = np.linspace(5, 20, self.n_bands+1)

        highpass = []
        lowpass = []
        for b, d in zip(bounds, dropoffs):
            highpass.append(self.tanh(d, b, self.radius))
            lowpass.append(np.ones_like(self.radius) - self.tanh(d, b, self.radius))

        bandpasses = []
        for lo in range(len(lowpass)-1):
            bandpasses.append(lowpass[lo] - lowpass[lo+1])
        
        return highpass[0], bandpasses, lowpass[-1]
    

    @staticmethod
    def sigmoid(rate, midpoint, x):
        return 1 / (1 + np.exp(-rate * (x - midpoint)))
    
    def make_angular_filters(self):

        # shifts the angular plain such that 0pi rad is at 9 o'clock
        shifted_angle = ((((self.angle + np.pi) / (2*np.pi) - 1/4) % 1) - 0.5) * (2*np.pi)

        midpoints = np.linspace(-np.pi/2, np.pi/2, self.n_orients+1)
        angfilters = []
        for i in range(len(midpoints)-1):
            filt = (self.sigmoid(15, midpoints[i], shifted_angle, ) - \
                    self.sigmoid(15, midpoints[i+1], shifted_angle, ))
            angfilters.append(filt*2)

        return angfilters
    

    def make_subband_filters(self):
        subbands = []
        for r in self.bandpasses:
            for a in self.angular_filters:
                subbands.append(a*r)    
        return subbands
    

    def get_filter_bank(self):
        highpass = torch.from_numpy(self.highpass)
        lowpass = torch.from_numpy(self.lowpass)
        subbands = list(map(lambda x: torch.from_numpy(x), self.subband_filters))
        return highpass, subbands, lowpass