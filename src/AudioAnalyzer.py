"""
AudioAnalyzer is a DSP analyzer for audio data

Python 3 is used.

Copyright (c) 2020 Edgar Lubicz

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""
__version__ = "0.1.0"
__author__ = "Edgar Lubicz"

import numpy as np
from numpy.fft import fft
from scipy import signal

class AudioAnalyzer(object):
    """
    TODO:
    """

    def __init__(self, fs=44100.0, blocksize=1024, npeaks=3, overlap=0.0, 
        peakthresh=-120.0, avgtime=0.0, smooth=0.0, ingain=0.0, window='none'):
        """
        Simple initializer
        :param fs: sampling rate in Hz
        :param blocksize: chunk length in samples
        :param npeaks: number of peaks to be extracted
        :param overlap: overlap factor between 0.0 and 0.5
        :param peakthresh: threshold for peaks in dB
        :param avgtime: average time release time in ms
        :param smooth: smoothing factor for octave band filtering (0.0 = OFF)
        :param ingain: input gain in dB
        :param window: FFT window. For possible window functions check self.wins
        """
        self.MIN_MAG_DB = -120.0
        self.MIN_MAG_LIN = 10.0 ** (self.MIN_MAG_DB * 0.05)

        # Windows
        # Overview below extracted from
        # https://cdn.rohde-schwarz.com/pws/dl_downloads/dl_application/application_notes/1ef77/1EF77_3e_Real-time_Spectrum_Analysis.pdf
        # ------------------------------------------------------------------
        # WindowSpectral | LeakageAmplitude | AccuracyFrequency | Resolution
        # ------------------------------------------------------------------
        # Blackman       | Best             | Good              | Fair
        # Flattop        | Good             | Best              | Poor
        # Gaussian       | Fair             | Good              | Fair
        # Rectangle      | Poor             | Poor              | Best
        # Hanning        | Good             | Fair              | Good
        # Hamming        | Fair             | Fair              | Good
        # Kaiser         | Good             | Good              | Fair
        # ------------------------------------------------------------------
        self.wins = {
            'hanning':np.hanning,
            'hamming':np.hamming,
            'bartlett':np.bartlett,
            'blackman':np.blackman,
            'kaiser':np.kaiser,
            'flattop':signal.flattop,
            'gaussian':signal.gaussian, # std fixed to 10
            'none':np.ones,
        }
        self.set_properties(fs,blocksize,npeaks,overlap,
            peakthresh,avgtime,smooth,ingain,window)
        self.yf = np.zeros(int(self.blocksize / 2 - 1))
        self.yf_prev = np.zeros(len(self.yf))
        self.y_prev = np.zeros(self.blocksize)

    def set_properties(self, fs=None, blocksize=None, npeaks=None, overlap=None, 
            peakthresh=None, avgtime=None, smooth=None, ingain=None, window=None):
        """
        Updates properties of the audio analyzer. The ones that are 
        not None at least.

        :param fs: same as in __init__
        :param blocksize: same as in __init__
        :param npeaks: same as in __init__
        :param overlap: same as in __init__
        :param peakthresh: same as in __init__
        :param avgtime: same as in __init__
        :param smooth: same as in __init__
        :param ingain: same as in __init__
        :param window: same as in __init__
        """
        should_update_smooth = False
        should_update_window = False
        if ingain is not None and ingain < 10.0:
            self.ingain = 10.0 ** (ingain * 0.05)
        if fs is not None and (fs == 44100 or fs == 48000):
            self.fs = fs
            should_update_smooth = True
        if blocksize is not None and blocksize >= 4 and blocksize <= 32768:
            self.blocksize = int(2**np.round(np.log2(blocksize)))
            self.yf_prev = np.zeros(int(self.blocksize / 2 - 1))
            should_update_window = True
        if npeaks is not None and npeaks > 0 and npeaks < 50:
            self.npeaks = int(npeaks)
        if overlap is not None and overlap >= 0.0 and overlap <= 100.0:
            self.overlap = overlap
        if peakthresh is not None and peakthresh >= self.MIN_MAG_DB:
            self.peak_thesh = peakthresh
        if avgtime is not None and avgtime >= 0.0:
            # https://dsp.stackexchange.com/questions/28308/exponential-weighted-moving-average-time-constant/
            if avgtime > 0.0:
                wc = (self.blocksize / self.fs) / (avgtime * 0.001)
                self.alpha = 1.0 - np.exp(-wc)
            else:
                self.alpha = 1.0
        if smooth is not None and smooth >= 0.0 and smooth <= 2.0:
            self.smooth = smooth
            should_update_smooth = True
        if window is not None:
            self.update_window(window)

        # update dependent parameters, if any
        if should_update_smooth:
            self.update_smooth_filter()
        if should_update_window:
            self.update_window()

    def update_smooth_filter(self):
        """
        Calculates indices for the octave band smoothing filter.
        This is based on the current smoothing factor, which indicates
        the fraction of octave to be used.
        """
        self.smooth_inds = []
        if self.smooth > 0.0:
            smooth_factors = [2 ** -(self.smooth/2.0), 2 ** (self.smooth/2.0)]
            max_ind = (self.blocksize / 2 - 1)
            for k in range(int(max_ind)):
                a = np.round(k * smooth_factors[0])          
                b = np.min([np.round(k * smooth_factors[1]), max_ind])
                self.smooth_inds += [(int(a),int(b))]
                #print(str(k) + ':' + str((a,b)))

    def update_window(self, window=None):
        """
        Updates thw window based on new window type, if not None, 
        or just refreshed current one

        :param window: same as in __init__
        """
        if window is not None:
            if window not in self.wins.keys():
                window = 'none'
            self.window_type = window
            # special treatment for kaiser, as it needs extra argument
            if window == 'kaiser':
                self.window = self.wins[window](self.blocksize,14)
            # also for gaussian, because of standard deviation
            elif window == 'gaussian':
                self.window = self.wins[window](self.blocksize,10.0)
            else:
                self.window = self.wins[window](self.blocksize)
        else:
            # probably just a blocksize update
            self.window = self.wins[self.window_type](self.blocksize)

    def process(self, in_data):
        """
        TODO:
        :param in_data: input data as 1D NumPy array
        """
        # if the blocksize has changed, wait until it gets updated
        if len(in_data) == len(self.window):
            self.y = in_data * self.ingain
            # apply FFT window
            y = self.window * self.y
            # overlap
            self.y_last = y
            # compute FFT and get relevant indices
            yf = fft(y)[0:int(self.blocksize/2)-1]
            # get the normalized of the magnitude
            norm_fact = np.sum(self.window) / 2.0
            yf_abs = np.abs(yf) / norm_fact
            # clip it to self.MIN_MAG_LIN as a relevant minimum
            yf_abs = np.clip(yf_abs, a_min = self.MIN_MAG_LIN, a_max = None)
            # dB value
            yf_abs = 20.0 * np.log10(yf_abs)
            # apply octave band smoothing if necessary
            if len(self.smooth_inds) == len(yf_abs):
                yf_abs_filt = np.zeros(len(yf_abs))
                #sqr_yf_abs = np.square(yf_abs)
                # smooth with octave band filters
                for k in range(len(yf)):
                    a = self.smooth_inds[k][0]
                    b = self.smooth_inds[k][1]
                    #mean = np.mean(sqr_yf_abs[a:b+1])
                    mean = np.mean(yf_abs[a:b+1])
                    yf_abs_filt[k] = mean
                #self.yf = np.sqrt(yf_abs_filt)
                self.yf = yf_abs_filt
            else:
                self.yf = yf_abs
            # smooth in time
            self.yf = self.alpha * self.yf + (1.0 - self.alpha) * self.yf_prev
            self.yf_prev = self.yf
            # find peaks
            data_peaks = [(0,self.MIN_MAG_DB)] * self.npeaks
            diff = self.yf[1:] - self.yf[0:-1]
            for i in range(1, len(diff) - 1):
                # peak detected by change of derivative
                if diff[i-1] > 0 and diff[i] <= 0:
                    # quadratic interpolation. 
                    # see https://ccrma.stanford.edu/~jos/sasp/Quadratic_Interpolation_Spectral_Peaks.html
                    alpha = self.yf[i-1]
                    beta = self.yf[i]
                    gamma = self.yf[i+1]
                    peakx = 0.5 * (alpha - gamma) / (alpha - 2 * beta + gamma)
                    peaky = beta - 0.25 * (alpha - gamma) * peakx
                    # peak needs to be greater than threshold and greater than minimum peak so far
                    if peaky > self.peak_thesh and peaky > data_peaks[0][1]:
                        data_peaks[0] = ((i + peakx) * self.fs / self.blocksize, peaky)
                        data_peaks = sorted(data_peaks, key = lambda tup:tup[1])
            
            self.peaky = [xy[1] for xy in data_peaks]
            self.peakx = [xy[0] for xy in data_peaks]
