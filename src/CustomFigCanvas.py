"""
CustomFigCanvas is a helper class for plotting matplotlib in realtime.

Python 3 is used.

This is largely based on many StackOverflow links that I lost the account for.
The main approach is to process an instance of Figure() instead of using 
matplotlib.pyplot (aka plt). FuncAnimation is used for realtime and blit is turned
on, as basically only the real time data changes (one exception is the blocksize).
The animation is set to run every 40ms, which leads to 25 FPS. More than that our
eyes cannot handle. 

There are two subplots. One for the time domain and the other for the frequency 
spectrum.

"""
__version__ = "0.2.0"
__author__ = "Edgar Lubicz"

from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar)
import time
import numpy as np

class CustomFigCanvas(FigureCanvas, FuncAnimation):
    """
    Class for plotting matplot data in realtime. 2 subplots,
    one for time data and the bottom one for the frequenc spectrum
    """

    def __init__(self, blocksize=1024, npeaks=3, fs=44100.0, size=None):
        """
        Simple initializer
        """
        self.TIME_RANGE = 5.0
        self.blocksize = blocksize
        self.npeaks = npeaks
        self.fs = fs
        self.frame_count = 0
        self.start_time = time.time()
        self.should_redraw = False

        # create matplotlib figure and axes
        self.fig = Figure(figsize=size)
        self.ax1 = self.fig.add_subplot(211)
        self.ax2 = self.fig.add_subplot(212)
        self.fig.subplots_adjust(hspace=0.5)
        #self.fig.canvas.mpl_connect('button_press_event', self.onClick)
        
        npointsx = int(self.blocksize * np.ceil(self.TIME_RANGE / (self.blocksize / self.fs)))
        self.x = np.linspace(0, self.TIME_RANGE, npointsx)
        self.xf = np.linspace(0, self.fs / 2, int(self.blocksize/2)-1)
        self.y = np.zeros(len(self.x))
        self.yf = np.zeros(len(self.xf))
        # format waveform axis
        self.ax1.set_title('Time Domain')
        self.ax1.set_xlabel('samples')
        self.ax1.set_ylabel('volume')
        self.ax1.set_ylim(-1.5, 1.5)
        self.ax1.set_xlim(0, self.TIME_RANGE)
        self.ax1.grid(which='both')
        # format spectrum axis
        self.ax2.set_title('Frequency Domain')
        self.ax2.set_xlabel('Hz')
        self.ax2.set_ylabel('dB')
        self.ax2.set_ylim(-90, 5)
        self.ax2.set_xlim(20.0, self.fs / 2.0)
        self.ax2.set_xscale('log')
        self.ax2.grid(which='both')

        # create a line object with random data
        self.line, = self.ax1.plot(self.x, self.y, '-', lw=2)
        # create semilogx line for spectrum
        self.line_fft, = self.ax2.semilogx(self.xf, self.yf, '-', lw=2)
        
        # markers for peaks of FFT
        self.peaky = [-120] * self.npeaks
        self.peakx = [0] * self.npeaks
        self.line_peaksfft, = self.ax2.semilogx(self.peakx, self.peaky, 'ro', lw=2)

        # marker "shadow" lines
        # horizontal
        self.line_hshadows = self.ax2.hlines(y = self.peaky,
            xmin = [self.ax2.get_xlim()[0]] * self.npeaks, 
            xmax = self.peakx, 
            color='red',
            #linestyle="dotted")
            linestyle=(0, (1, 3)))
        # vertical
        self.line_vshadows = self.ax2.vlines(x = self.peakx,
            ymin = [self.ax2.get_ylim()[0]] * self.npeaks, 
            ymax = self.peaky,
            color='red',
            #linestyle="dotted")
            linestyle=(0, (1, 3)))
        
        self.plot_peaks = True
        self.plot_shadows = True

        FigureCanvas.__init__(self, self.fig)
        FuncAnimation.__init__(self, self.fig, self.plot_callback, init_func=self.reset_plot, interval=40, blit=True)

    def set_plot_properties(self, blocksize=None,fs=None, peaks=None, shadows = None):
        """
        Sets the properties for the plots.
        :param blocksize: length of the chunk in samples
        :param fs: sampling rate in Hz
        """
        refresh_plot = False
        if blocksize is not None:
            self.blocksize = blocksize
            refresh_plot = True
        if fs is not None:
            self.fs = fs
            refresh_plot = True
        if peaks is not None:
            self.plot_peaks = peaks
        if shadows is not None:
            self.plot_shadows = shadows

        if refresh_plot:
            print("Refreshing plot with:")
            print("  Blocksize: " + str(self.blocksize))
            print("  FS: " + str(self.fs))
            # x variables for plotting
            npointsx = int(self.blocksize * np.ceil(self.TIME_RANGE / (self.blocksize / self.fs)))
            self.x = np.linspace(0, self.TIME_RANGE, npointsx)
            self.xf = np.linspace(0, self.fs / 2, int(self.blocksize/2)-1)
            self.y = np.zeros(len(self.x))
            self.yf = np.zeros(len(self.xf))
            self.ax2.set_xlim(20.0, self.fs / 2.0)
            # force draw canvas
            self.should_redraw = True
            

    def set_plot_data(self, y=None,yf=None,peakx=None,peaky=None):
        """
        Sets the data of any plotting curve.
        :param y: time data in linear
        :param yf: frequency magnitude data 
        :param peakx: x axis values for the peaks
        :param peaky: y axis values for the peaks
        """
        if y is not None:
            self.y = np.append(self.y[len(y):], y)
        if yf is not None:
            self.yf = yf
        if peakx is not None:
            self.peakx = peakx
        if peaky is not None:
            self.peaky = peaky

    def plot_callback(self, i):
        """
        Animation callback function. Here we update all of the plotted
        curves according to the latest set data
        """
        ret = [self.line, self.line_fft]

        if self.should_redraw:
            self.fig.canvas.draw()
            self.should_redraw = False
        if len(self.x) == len(self.y):
            self.line.set_xdata(self.x)
            self.line.set_ydata(self.y)
        if len(self.xf) == len(self.yf) == int(self.blocksize/2)-1:
            self.line_fft.set_xdata(self.xf)
            self.line_fft.set_ydata(self.yf)
        # peaks
        if self.plot_peaks:
            self.line_peaksfft.set_ydata(self.peaky)
            self.line_peaksfft.set_xdata(self.peakx)
            ret += [self.line_peaksfft]
        # shadows
        if self.plot_shadows:
            xmin = self.ax2.get_xlim()[0]
            ymin = self.ax2.get_ylim()[0]
            # vertical shadows
            segs = ([[x, ymin], [x, y]] for x, y in zip(self.peakx, self.peaky))
            self.line_vshadows.set_segments(segs)
            # horizontal shadows
            segs = ([[xmin, y], [x, y]] for x, y in zip(self.peakx, self.peaky))
            self.line_hshadows.set_segments(segs)
            ret += [self.line_hshadows, self.line_vshadows]
        
        self.frame_count += 1
          
        return ret

    def reset_plot(self):
        """
        Resets animation
        """
        print('Plot animation re-started')
        self.frame_count = 0
        self.start_time = time.time()

        ret = [self.line, self.line_fft]
        if self.plot_shadows:
            ret += [self.line_hshadows, self.line_vshadows]
        if self.plot_peaks:
            ret += [self.line_peaksfft]

        return ret
