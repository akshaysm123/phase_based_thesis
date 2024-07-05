import numpy as np
import numpy.fft as npf
import cv2

class Pyramid:
    def __init__(self, image_path, highpass_cutoff=0.9, lowpass_cutoff=0.1, n_bands=4, 
                 n_orients=5, radial_tanh_rate=10, angular_sigm_rate=35):
        self.image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        self.highpass_cutoff = highpass_cutoff
        self.lowpass_cutoff = lowpass_cutoff
        self.n_bands = n_bands
        self.n_orients = n_orients
        self.radial_tanh_rate = radial_tanh_rate
        self.angular_sigm_rate = angular_sigm_rate

        self._angle, self._radius = self._polar_grid()
        self.highpass_filter, self._radial_filters, self.lowpass_filter = self._make_radial_filters()
        self._angular_filters = self._make_angular_filters()
        self.subband_filters = self._make_subband_filters()

        self._fft_image = npf.fft2(self.image)
        self._fft_image_shifted = npf.fftshift(self._fft_image)
        self.fft_highpass, self.fft_subbands, self.fft_lowpass = self._fft_filters()

    def _polar_grid(self):
        h,w = self.image.shape
        h_lin = np.linspace(-1, 1, h)
        w_lin = np.linspace(-1, 1, w)
        x_mesh, y_mesh = np.meshgrid(w_lin, h_lin)
        angle = np.arctan2(y_mesh, x_mesh)
        radius = np.sqrt(x_mesh**2 + y_mesh**2)
        radius[h//2][w//2] += 1e-6
        return angle, radius
    
    @staticmethod
    def _sigmoid(x, midpoint, rate):
        return 1 / (1 + np.exp(-rate * (x - midpoint)))
    
    @staticmethod
    def _cut_tanh(x, cutoff, rate):
        tanh = np.tanh(rate * (x-cutoff))
        mask = tanh > 0
        return tanh*mask
    
    def _make_radial_filters(self):
        high = self.highpass_cutoff * min(np.min(self._radius [0,:]), np.min(self._radius [:,0]))
        midpoints = np.linspace(1, 0, self.n_bands+1)
        midpoints = np.exp(midpoints)
        midpoints -= np.min(midpoints)
        midpoints /= np.max(midpoints)
        midpoints *= (high-self.lowpass_cutoff)
        midpoints += self.lowpass_cutoff

        hi_pass, lo_pass = [], []
        for m in midpoints:
            filt = self._cut_tanh(self._radius, m, self.radial_tanh_rate)
            hi_pass.append(filt)
            lo_pass.append(1 - filt)
        
        rad_filters = []
        for i in range(len(midpoints)-1):
            band = np.ones(self._radius.shape) - hi_pass[i] - lo_pass[i+1]
            rad_filters.append(band)

        return hi_pass[0], rad_filters, lo_pass[-1]
    
    def _make_angular_filters(self):
        norm_angle = ((self._angle + np.pi)/ (2*np.pi) - 1/8) % 1
        midpoints = np.linspace(1/4, 3/4, self.n_orients+1)

        angfilters = []
        for i in range(len(midpoints)-1):
            filt = (self._sigmoid(norm_angle, midpoints[i], self.angular_sigm_rate) - \
                    self._sigmoid(norm_angle, midpoints[i+1], self.angular_sigm_rate))
            angfilters.append(filt)

        return angfilters
    
    def _make_subband_filters(self):
        subbands = []
        for r in self._radial_filters:
            for a in self._angular_filters:
                subbands.append(a*r)    
        return subbands
    
    def _fft_filters(self):
        hipass = self.highpass_filter * self._fft_image_shifted
        lopass = self.lowpass_filter * self._fft_image_shifted
        subbands = []
        for f in self.subband_filters:
            subbands.append(2 * f * self._fft_image_shifted)
        return hipass, subbands, lopass
        
    def collapse_fft(self):
        return self.fft_highpass + sum(self.fft_subbands) + self.fft_lowpass
    
    