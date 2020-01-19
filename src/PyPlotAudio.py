"""
PyPlotAudio is a graphical audio analyzer based on QT5, soundfile, numpy and scipy.

Python 3 is used.

Audio files, test signals and sound device streams can be read and analyzed in real time.
The user is able to tweak a few parameters and observe their influence in the spectrum 
straight away.

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
__version__ = "0.2.0"
__author__ = "Edgar Lubicz"

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QFileDialog, QMessageBox
import numpy as np
from scipy import signal
import pyaudio
import soundfile as sf
from  pydub import AudioSegment
import sys, time, os
import gui
from AudioAnalyzer import AudioAnalyzer
from CustomFigCanvas import CustomFigCanvas
import json

class PyPlotAudio(QtWidgets.QMainWindow, gui.Ui_MainWindow):
    """
    Class for graphically plotting audio in time and frequency domain.
    User may teak a few parameters and observe its influence on frequency
    spectrum
    """

    def __init__(self, parent=None):
        """
        Simple initializer
        """

        # constructors
        super(PyPlotAudio, self).__init__(parent)
        self.setupUi(self)
        
        # Menu bar
        self.actionExit.triggered.connect(self.close)
        self.actionAbout.triggered.connect(self.about)

        # making tuning elements invisible
        self.labelTuning.setVisible(False) , self.comboTuning.setVisible(False)
        self.labelString1.setVisible(False), self.sliderStringUp1.setVisible(False), self.sliderStringDown1.setVisible(False)
        self.labelString2.setVisible(False), self.sliderStringUp2.setVisible(False), self.sliderStringDown2.setVisible(False)
        self.labelString3.setVisible(False), self.sliderStringUp3.setVisible(False), self.sliderStringDown3.setVisible(False)
        self.labelString4.setVisible(False), self.sliderStringUp4.setVisible(False), self.sliderStringDown4.setVisible(False)
        self.labelString5.setVisible(False), self.sliderStringUp5.setVisible(False), self.sliderStringDown5.setVisible(False)
        self.labelString6.setVisible(False), self.sliderStringUp6.setVisible(False), self.sliderStringDown6.setVisible(False)
        self.labelString7.setVisible(False), self.sliderStringUp7.setVisible(False), self.sliderStringDown7.setVisible(False)
        
        # set up status bar
        self.labelStatusPeak = []
        for i in range(3):
            self.labelStatusPeak.insert(0, QtWidgets.QLabel())
            self.labelStatusPeak[0].setAlignment(QtCore.Qt.AlignLeft)
            self.labelStatusPeak[0].setObjectName("labelStatusPeak" + str(i))
            self.labelStatusPeak[0].setText("-Hz=-dB")
            self.statusBar.addWidget(self.labelStatusPeak[0])
            width = 145
            geometry = self.labelStatusPeak[0].geometry()
            geometry.setWidth(width)
            self.labelStatusPeak[0].setGeometry(geometry)
            self.labelStatusPeak[0].setMinimumWidth(width)
            self.labelStatusPeak[0].setMaximumWidth(width)

        self.labelStatusFps = QtWidgets.QLabel()
        self.labelStatusFps.setAlignment(QtCore.Qt.AlignCenter)
        self.labelStatusFps.setObjectName("labelStatusFps")
        self.labelStatusFps.setText("FPS=-")
        self.statusBar.addPermanentWidget(self.labelStatusFps)
        self.labelStatusCpu = QtWidgets.QLabel()
        self.labelStatusCpu.setAlignment(QtCore.Qt.AlignCenter)
        self.labelStatusCpu.setObjectName("labelStatusCpu")
        self.labelStatusCpu.setText("CPU(avg)=-")
        self.statusBar.addPermanentWidget(self.labelStatusCpu)
        self.labelStatusCpuPeak = QtWidgets.QLabel()
        self.labelStatusCpuPeak.setAlignment(QtCore.Qt.AlignCenter)
        self.labelStatusCpuPeak.setObjectName("labelStatusCpuPeak")
        self.labelStatusCpuPeak.setText("CPU(peak)=-")
        self.statusBar.addPermanentWidget(self.labelStatusCpuPeak)
        
        # values at start up
        self.DEFAULT_VALUES = {
            'ingain':0.0,
            'blocksize':1024,
            'fs':48000.0,
            'avgtime':0.0,
            'overlap':0.0,
            'npeaks':3,
            'peakthresh':-60,
            'smooth':'off',
            'audioch':0,
            'window':'rectangular',
            'runmode':'wav',
            'testparam' : 500,
            'peaks' : True,
            'shadows' : True
        }

        self.SETTINGS_PATH = ".settings"
        self.settings = {}
        if not os.path.isfile(self.SETTINGS_PATH):
            with open(self.SETTINGS_PATH, 'w') as fp:
                json.dump(self.DEFAULT_VALUES, fp)
                self.settings = self.DEFAULT_VALUES
        else:
            with open(self.SETTINGS_PATH, 'r') as fp:
                self.settings = json.load(fp)
                for key, val in self.DEFAULT_VALUES.items():
                    if key not in self.settings:
                        self.settings[key] = val

        # generic members
        self.runmode = 'none'
        self.test_data = []
        self.testparam = self.settings['testparam']
        self.alpha = 0.99 # time smoothing for stats
        self.cpu = 0.0
        self.cpupeak = 0.0
        self.cpu_prev = 0.0
        self.toggle_playpause = { 'play':'pause', 'pause':'play'}

        # audio stuff
        self.aa = AudioAnalyzer()
        self.pa = pyaudio.PyAudio()
        self.stream = None
        self.is_playing = False
        self.current_ch = 0
        self.last_cpu_update = time.time()
        self.last_wav_update = time.time()
        self.wf_info = None
        self.wf = None
        self.nchannels = 1
        self.converted_mp3 = []

        # wave slider
        self.wav_slider_moving = False
        self.sliderWavPlayer.sliderReleased.connect(self.slider_wav_onleave)
        self.sliderWavPlayer.sliderPressed.connect(self.slider_wav_onpress)

        # fill up audio IF combobox
        for i in range(self.pa.get_device_count()):
            info = self.pa.get_device_info_by_index(i)
            if(info['maxInputChannels'] > 0 and info['maxOutputChannels'] > 0):
                self.comboAudioIF.addItem(str(i) + ':' + info['name'])
        
        # matplotlib handling
        self.last_nframes = 0
        self.canvas = CustomFigCanvas(size=(50, 50)) # TODO: un-hardcode this
        self.layoutContentMpl.addWidget(self.canvas, alignment=QtCore.Qt.AlignCenter)
        self.canvas.draw()

        # connect checkboxes
        self.checkboxPeaks.stateChanged.connect(self.checkbox_peaks_changed)
        self.checkboxShadows.stateChanged.connect(self.checkbox_shadows_changed)

        # connect dial signals
        self.dialOverlap.valueChanged.connect(self.dial_overlap_changed)
        self.dialAverageTime.valueChanged.connect(self.dial_avg_changed)
        self.dialNumberPeaks.valueChanged.connect(self.dial_npeaks_changed)
        self.dialPeakThresh.valueChanged.connect(self.dial_peakthresh_changed)
        self.dialInputGain.valueChanged.connect(self.dial_ingain_changed)
        self.dialTestParam.valueChanged.connect(self.dial_testparam_changed)
        
        # connect combobox change
        self.comboBlockSize.currentIndexChanged.connect(self.combo_blocksize_changed)
        self.comboWindow.currentIndexChanged.connect(self.combo_window_changed)
        self.comboFS.currentIndexChanged.connect(self.combo_fs_changed)
        self.comboSmoothing.currentIndexChanged.connect(self.combo_smooth_changed)
        self.comboAudioIF.currentIndexChanged.connect(self.update_audioif)
        self.comboAudioCh.currentIndexChanged.connect(self.update_audioch)
        self.comboRunMode.currentIndexChanged.connect(self.update_runmode)
        
        # connect buttons
        self.btnOpenTestFile.clicked.connect(self.btn_open_test_file)
        self.btnPlayPause.clicked.connect(self.btn_play_pause)

        # set default values
        self.combo_setindex_by_value(self.comboBlockSize, self.settings['blocksize'])
        self.combo_setindex_by_value(self.comboWindow, self.settings['window'])
        self.combo_setindex_by_value(self.comboFS, int(self.settings['fs']))
        self.combo_setindex_by_value(self.comboSmoothing, self.settings['smooth'])
        self.combo_setindex_by_value(self.comboAudioCh, self.settings['audioch'])
        self.combo_setindex_by_value(self.comboRunMode, self.settings['runmode'])
        self.dialOverlap.setValue(self.settings['overlap'])
        self.dialAverageTime.setValue(self.settings['avgtime'])
        self.dialNumberPeaks.setValue(self.settings['npeaks'])
        self.dialPeakThresh.setValue(self.settings['peakthresh'])
        self.dialInputGain.setValue(self.settings['ingain'])
        self.dialTestParam.setValue(self.settings['testparam'])
        self.checkboxPeaks.setChecked(self.settings['peaks'])
        self.checkboxShadows.setChecked(self.settings['shadows'])

        # select default device
        didi = self.pa.get_default_input_device_info()
        self.combo_setindex_by_value(self.comboAudioIF, str(didi['index']) + ':' + didi['name'])

    def about(self):
        """
        Open about message box
        """
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        about_text = \
        "PyPlotAudio is a graphical audio analyzer based on QT5, soundfile, numpy and scipy." \
        "\n\nAudio files, test signals and sound device streams can be read and analyzed in real time." \
        "The user is able to tweak a few parameters and observe their influence in the spectrum " \
        "straight away." \
        "\n\nCopyright (c) 2020 Edgar Lubicz" \
        
        msg.setText(about_text)
        msg.setWindowTitle("PyPlotAudio Version " + __version__)
        #msg.setDetailedText("Version " + __version__)
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec()

    def combo_setindex_by_value(self, combo, value):
        """
        Helper function for seting the index of a QComboBox
        by an value that is contained in one of its
        items

        :param combo: A QComboBox object.
        :param value: An integer value
        """
        # first try exact match
        found = False
        for i in range(combo.count()):
            if str(value) == combo.itemText(i):
                combo.setCurrentIndex(i)
                found = True
                break
        # if not found, then try substring 
        if not found:
            for i in range(combo.count()):
                if str(value) in combo.itemText(i).lower():
                    combo.setCurrentIndex(i)
                    break

    def slider_wav_onpress(self):
        """
        Callback when the GUI wave player slider
        is pressed.
        """
        self.wav_slider_moving = True

    def slider_wav_onleave(self):
        """
        Callback when the GUI wave player slider
        is released.
        """
        self.wav_slider_moving = False
        if self.wf and not self.wf.closed:
            # update wave player position based on new slider value
            slidermin = float(self.sliderWavPlayer.minimum())
            slidermax = float(self.sliderWavPlayer.maximum())
            sliderdelta = (slidermax - slidermin)
            frame = self.wf_info.frames * (self.sliderWavPlayer.value() - slidermin) / sliderdelta
            self.wf.seek(int(frame))
            # immediately update time label
            self.update_player_timelabel(int(frame))
        else:
            # no relevant WAV yet, go back to leftmost value
            self.sliderWavPlayer.setValue(self.sliderWavPlayer.minimum())

    def checkbox_peaks_changed(self):
        """
        Callback for when the peaks checkbox is ticked
        """
        self.canvas.set_plot_properties(peaks=self.checkboxPeaks.isChecked())
        self.settings["peaks"] = self.checkboxPeaks.isChecked()
    
    def checkbox_shadows_changed(self):
        """
        Callback for when the shadows checkbox is ticked
        """
        self.canvas.set_plot_properties(shadows=self.checkboxShadows.isChecked())
        self.settings["shadows"] = self.checkboxShadows.isChecked()

    def dial_overlap_changed(self):
        """
        Callback when the GUI dial for overlap
        is turned.
        """
        self.overlap = self.dialOverlap.value()
        self.labelOverlapValue.setText(str(self.dialOverlap.value()))
        self.aa.set_properties(overlap=self.overlap)
        self.update_test_data()
        self.settings["overlap"] = self.overlap
    
    def dial_avg_changed(self):
        """
        Callback when the GUI dial for time average
        is turned.
        """
        self.avgtime = self.dialAverageTime.value()
        self.labelAverageTimeValue.setText(str(self.dialAverageTime.value()))
        self.aa.set_properties(avgtime=self.avgtime)
        self.update_test_data()
        self.settings["avgtime"] = self.avgtime
    
    def dial_npeaks_changed(self):
        """
        Callback when the GUI dial for number of peaks
        is turned.
        """
        self.npeaks = self.dialNumberPeaks.value()
        self.labelNumberPeaksValue.setText(str(self.dialNumberPeaks.value()))
        self.aa.set_properties(npeaks=self.npeaks)
        self.update_test_data()
        self.settings["npeaks"] = self.npeaks
    
    def dial_peakthresh_changed(self):
        """
        Callback when the GUI dial for peak threshold
        is turned.
        """
        self.peakthresh = self.dialPeakThresh.value()
        self.labelPeakThreshValue.setText(str(self.dialPeakThresh.value()))
        self.aa.set_properties(peakthresh=self.peakthresh)
        self.update_test_data()
        self.settings["peakthresh"] = self.peakthresh

    def dial_ingain_changed(self):
        """
        Callback when the GUI dial for input gain
        is turned.
        """
        self.ingain = self.dialInputGain.value()
        self.labelInputGainValue.setText(str(self.dialInputGain.value()))
        self.aa.set_properties(ingain=self.ingain)
        self.update_test_data()
        self.settings["ingain"] = self.ingain

    def dial_testparam_changed(self):
        """
        Callback when the GUI dial for test parameter
        is turned. This is only relevant when we're in 
        one of the test modes
        """
        self.testparam = self.dialTestParam.value()
        self.labelTestParamValue.setText(str(self.dialTestParam.value()))
        self.update_test_data()
        self.settings["testparam"] = self.testparam

    def combo_blocksize_changed(self):
        """
        Callback when the GUI combo box for blocksize
        changes index.
        """
        self.blocksize = int(self.comboBlockSize.currentText())
        self.aa.set_properties(blocksize=self.blocksize)
        self.canvas.set_plot_properties(blocksize=self.blocksize)
        self.update_test_data()
        self.update_stream()
        self.settings["blocksize"] = self.blocksize

    def combo_window_changed(self):
        """
        Callback when the GUI combo box for FFT window
        changes index.
        """
        self.window = self.comboWindow.currentText().lower()
        self.aa.set_properties(window=self.window)
        self.update_test_data()
        self.settings["window"] = self.window

    def combo_fs_changed(self):
        """
        Callback when the GUI combo box for sampling rate
        changes index.
        """
        self.fs = int(self.comboFS.currentText().split(" ")[0])
        self.aa.set_properties(fs=self.fs)
        self.canvas.set_plot_properties(fs=self.fs)
        self.update_test_data()
        self.update_stream()
        self.settings["fs"] = self.fs

    def combo_smooth_changed(self):
        """
        Callback when the GUI combo box for octave band smoothing
        changes index.
        """
        smooth = self.comboSmoothing.currentText().lower()
        if smooth == 'off':
            self.smoothing = 0
        else:
            # calculate the real value of the fraction
            fraction = smooth.split(" ")[0].split("/")
            self.smoothing = float(fraction[0])
            for den in fraction[1:]:
                self.smoothing /= float(den)
        self.aa.set_properties(smooth=self.smoothing)
        self.update_test_data()
        self.settings["smooth"] = smooth

    def stream_callback(self, in_data, frame_count, time_info, flag):
        """
        Callback for PyAudio.
        This is the main processing function for the audio analysis.
        """
        start_time = time.time()
        finished = False
        if self.runmode == 'wav' and self.wf is not None:
            curr_frame = self.wf.tell()
            if curr_frame < self.wf_info.frames:
                out_flag = pyaudio.paContinue
                data = self.wf.read(self.blocksize)
                out_data = np.array(data, dtype=np.float32)
                # blend to one channel
                data = data.mean(1) 
                if len(data) < self.blocksize: 
                    data = np.append(data, np.zeros(self.blocksize - len(data)))
                    finished = True
                if (start_time - self.last_wav_update) > 1.0:
                    self.last_wav_update = start_time
                    self.update_player_timelabel(curr_frame)
                    if not self.wav_slider_moving:
                        slidermin = float(self.sliderWavPlayer.minimum())
                        slidermax = float(self.sliderWavPlayer.maximum())
                        sliderdelta = (slidermax - slidermin)
                        sliderpos = int(slidermin + sliderdelta * curr_frame / self.wf_info.frames)
                        self.sliderWavPlayer.setValue(sliderpos)
            else:
                self.sliderWavPlayer.setValue(self.sliderWavPlayer.minimum())
                self.update_player_timelabel(0)
                self.wf.seek(0)
                finished = True
        elif self.runmode == 'stream' and in_data is not None:
            out_flag = pyaudio.paContinue
            out_data = in_data
            # get data for the specified channel
            data = in_data[self.current_ch::self.current_ch+1]
            # format if to numpy
            data = np.fromstring(data, 'Float32')
        else:
            finished = True
        
        if finished:
            print("Player finished!")
            out_data = np.zeros([self.blocksize, self.nchannels])
            data = np.zeros(self.blocksize)
            self.is_playing = False
            out_flag = pyaudio.paComplete
            self.update_play_pause(force='play')
                
        # process data
        self.aa.process(data)
        # update plot
        self.canvas.set_plot_data(y=self.aa.y,yf= self.aa.yf, peakx=self.aa.peakx, peaky=self.aa.peaky)
        
        stop_time = time.time()
        cpu_load = 100.0 * (stop_time - start_time) / (self.blocksize / self.fs)
        self.cpu = self.cpu * self.alpha + cpu_load * (1.0 - self.alpha)
        self.cpupeak = max(self.cpupeak, self.cpu)

        # Only update stats every 0.5s
        delta = stop_time - self.last_cpu_update
        if delta > 0.5:
            curr_nframes = self.canvas.frame_count
            fps = (curr_nframes - self.last_nframes) / delta
            self.last_nframes = curr_nframes 
            self.last_cpu_update = stop_time
            self.labelStatusCpu.setText("CPU(avg)=" + "{0:.1f}".format(self.cpu) + "%")
            self.labelStatusCpuPeak.setText("CPU(peak)=" + "{0:.1f}".format(self.cpupeak) + "%")
            self.labelStatusFps.setText("FPS=" + "{0:.1f}".format(fps))
            self.cpupeak = 0.0
            for ind, lsp in enumerate(self.labelStatusPeak):
                if ind < self.npeaks:
                    lsp.setText("{0:.1f}".format(self.aa.peakx[ind]) + "Hz=" + 
                        "{0:.1f}".format(self.aa.peaky[ind]) + "dB")
                else:
                    lsp.setText("-Hz=-dB")

        return (out_data, out_flag)

    def update_player_timelabel(self, frame):
        """
        Helper function for updating the GUI label for WAV player timestamp
        :param frame: current frame index of wave file
        """
        total_secs = self.wf_info.frames // self.fs
        total_mins = total_secs // 60
        total_secs_str = "{:02d}".format(total_secs % 60)
        total_mins_str = "{:02d}".format(total_mins)
        curr_secs = frame // self.fs
        curr_mins = curr_secs // 60
        curr_secs_str = "{:02d}".format(curr_secs % 60)
        curr_mins_str = "{:02d}".format(curr_mins)
        self.labelWavPlayer.setText(curr_mins_str + ':' + curr_secs_str + '/' + total_mins_str + ':' + total_secs_str)

    def update_stream(self, force = None):
        """
        Helper function for updating the audio stream 
        :param force: Forces either 'pause' or 'play'. If None, then refreshes
                    the stream based on the is_playing flag
        """
        should_play = False
        if force is not None: 
            if force != 'play':
                should_play = False
            else:
                should_play = True
        elif self.is_playing: #if it was playing, keep on playing
            should_play = True
        
        # regardless of play or pause, we need to stop the current stream first, if any
        if self.stream:
            print("Stopping stream")
            self.stream.stop_stream()
            self.stream.close()

        if should_play:
            input = (self.runmode == 'stream')
            output = (self.runmode == 'wav')
            if self.runmode == 'wav':
                self.nchannels = self.wf_info.channels
            else:
                self.nchannels = 1
            print("Starting stream with:")
            print("  Number of channels: " + str(self.nchannels))
            print("  Has Input: " + str(input))
            print("  Has Output: " + str(output))
            print("  Blocksize: " + str(self.blocksize))
            print("  FS: " + str(self.fs))
            # stream object
            self.stream = self.pa.open(
                format=pyaudio.paFloat32,
                channels=self.nchannels,
                rate=int(self.fs),
                input=input,
                output=output,
                input_device_index=self.audioif_ind,
                output_device_index=self.audioif_ind,
                frames_per_buffer=self.blocksize,
                stream_callback=self.stream_callback
            )
            self.stream.start_stream()
            self.is_playing = True
        else:
            self.is_playing = False

    def update_play_pause(self, force=None):
        """
        Helper function for updating play/pause button state
        :param force: Forces either 'pause' or 'play'. If None, then toggles
        """
        if force:
            next_state = force
        else:
            # toggle
            curr_state = self.btnPlayPause.text().lower()
            next_state = self.toggle_playpause[curr_state]

        palette = self.btnPlayPause.palette()
        role = self.btnPlayPause.backgroundRole()
        if next_state == 'pause':
            self.btnPlayPause.setText('Pause')
            palette.setColor(role, QtGui.QColor('red'))
            self.btnPlayPause.setPalette(palette)
            self.comboAudioIF.setEnabled(False)
            self.comboAudioCh.setEnabled(False)
        else:
            self.btnPlayPause.setText('Play')
            palette.setColor(role, QtGui.QColor('green'))
            self.btnPlayPause.setPalette(palette)
            if self.runmode == 'stream' or self.runmode == 'wav':
                self.comboAudioIF.setEnabled(True)
                self.comboAudioCh.setEnabled(True)

    def btn_play_pause(self):
        """
        Callback when the GUI button for play/pause
        is pressed.
        """
        if 'stream' == self.runmode or ('wav' == self.runmode and self.wf is not None):
            try:
                curr_mode = self.btnPlayPause.text().lower()
                self.update_stream(force=curr_mode)
                # toggle GUI state of play/pause button
                self.update_play_pause()
                # force update stream
            except Exception as e:
                print("Failed to open stream ")
                print(str(e))

    def btn_open_test_file(self):
        """
        Callback when the GUI button for open a test WAV file
        is pressed.
        """
        dlg = QFileDialog()
        dlg.setFileMode(QFileDialog.AnyFile)
        dlg.setNameFilters(["WAV files (*.wav)", 
            "MP3 (*.mp3)", 
            #"MP4 (*.mp4)", 
            "FLAC (*.flac)", 
            "OGG (*.ogg)",
            "Other Audio Formats (*.AIFF *.AU *.RAW)"])
        if dlg.exec_():
            filename = dlg.selectedFiles()[0]
            fname, ext = os.path.splitext(filename)
            if ext.lower() == ".mp3":
                try:
                    print("Converting mp3 to wav")
                    mp3 = filename
                    filename = ""
                    new_wav = fname + ".wav"
                    if not os.path.isfile(new_wav):
                        if new_wav not in self.converted_mp3:
                            sound = AudioSegment.from_mp3(mp3)
                            sound.export(new_wav, format="wav")
                            self.converted_mp3 += [new_wav]
                    filename = new_wav
                except:
                    msg = QMessageBox()
                    msg.setIcon(QMessageBox.Information)
                    msg.setText("Could not convert mp3. Try installing ffmpeg")
                    msg.setWindowTitle("Warning")
                    #msg.setDetailedText("The details are as follows:")
                    msg.setStandardButtons(QMessageBox.Ok)
                    msg.exec()
            if len(filename) > 0:
                self.txtTestFile.setText(filename)
                if self.wf is not None:
                    self.wf.close()
                self.wf = sf.SoundFile(filename)
                self.wf_info = sf.info(filename)
                self.fs = sf.info(filename).samplerate
                self.combo_setindex_by_value(self.comboFS, self.fs)
                self.aa.set_properties(fs=self.fs)
                self.canvas.set_plot_properties(fs=self.fs)
                self.sliderWavPlayer.setValue(self.sliderWavPlayer.minimum())
                self.update_player_timelabel(0)

    def update_test_data(self):
        """
        We update the audio data depending on which test mode we are in.
        If we are either streaming from audio interface of WAv player, this
        is not applicable. There are 3 test modes:
        - multisine: fixed
        - white noise: fixed
        - square: frequency variable via test parameter dial
        - sawtooth: : frequency variable via test parameter dial
        """
        if self.runmode == 'test':
            test_data = []
            if self.test_mode == 'multisine':
                freqs = [100, 300,  500, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 7000, 12000, 15000]
                test_data = np.zeros(self.blocksize)
                #A = 1.0 / len(freqs)
                A = 1.0
                for f in freqs:
                    t = np.linspace(0, 2 * np.pi * f * (self.blocksize / self.fs), self.blocksize)
                    test_data += A * np.sin(t)
            elif self.test_mode == 'whitenoise':
                test_data = 2 * (np.random.rand(self.blocksize) - 0.5) # between -1 and 1
            elif self.test_mode == 'square':
                f = self.testparam
                t = np.linspace(0, 2 * np.pi * f * (self.blocksize / self.fs), self.blocksize)
                test_data = signal.square(t)
            elif self.test_mode == 'sawtooth':
                f = self.testparam
                t = np.linspace(0, 2 * np.pi * f * (self.blocksize / self.fs), self.blocksize)
                test_data = signal.sawtooth(t)
            self.aa.process(test_data)
            self.canvas.set_plot_data(y=self.aa.y, yf=self.aa.yf, peakx=self.aa.peakx, peaky=self.aa.peaky)

    def update_runmode(self):
        """
        We update the run mode, depending on the current selection of the QComboBox
        comboRunMode. There are 3 run modes:
        - stream: streaming input audio from audio interface and analysing it
        - wav: streaming output audio from WAV file and analyzing it
        - test: Analyze simple test data
        """
        rm = self.comboRunMode.currentText().lower()
        if 'stream' in rm:
            self.runmode = 'stream'
            self.comboAudioIF.setEnabled(True)
            self.comboAudioCh.setEnabled(True)
            self.btnOpenTestFile.setEnabled(False)
            self.update_stream(force='pause')
            self.update_play_pause('play')
            self.btnPlayPause.setEnabled(True)
            self.comboAudioIF.setEnabled(True)
            self.comboAudioCh.setEnabled(True)
            self.comboFS.setEnabled(True)
            self.dialTestParam.setEnabled(False)
        elif 'test' in rm:
            self.runmode = 'test'
            self.comboAudioIF.setEnabled(False)
            self.comboAudioCh.setEnabled(False)
            self.btnOpenTestFile.setEnabled(False)
            self.update_stream(force='pause')
            self.update_play_pause('play')
            self.btnPlayPause.setEnabled(False)
            self.comboAudioIF.setEnabled(False)
            self.comboAudioCh.setEnabled(False)
            self.comboFS.setEnabled(True)
            if 'multisine' in rm:
                self.test_mode = 'multisine'
                self.dialTestParam.setEnabled(False)
            elif 'white noise' in rm:
                self.test_mode = 'whitenoise'
                self.dialTestParam.setEnabled(False)
            elif 'square' in rm:
                self.test_mode = 'square'
                currtext = self.labelTestParam.text().lower()
                if currtext != 'freq':
                    self.labelTestParam.setText('Freq')
                    self.dialTestParam.setMinimum(20)
                    self.dialTestParam.setMaximum(20000)
                    self.dialTestParam.setValue(500)
                self.dialTestParam.setEnabled(True)
            elif 'sawtooth' in rm:
                self.test_mode = 'sawtooth'
                currtext = self.labelTestParam.text().lower()
                if currtext != 'freq':
                    self.labelTestParam.setText('Freq')
                    self.dialTestParam.setMinimum(20)
                    self.dialTestParam.setMaximum(20000)
                    self.dialTestParam.setValue(500)
                self.dialTestParam.setEnabled(True)
        elif 'wav' in rm:
            self.runmode = 'wav'
            self.comboAudioIF.setEnabled(True)
            self.comboAudioCh.setEnabled(True)
            self.btnOpenTestFile.setEnabled(True)
            self.update_stream(force='pause')
            self.update_play_pause('play')
            self.btnPlayPause.setEnabled(True)
            self.comboAudioIF.setEnabled(True)
            self.comboAudioCh.setEnabled(True)
            self.comboFS.setEnabled(False)
            self.dialTestParam.setEnabled(False)
        self.update_test_data()
        self.settings["runmode"] = rm

    def update_audioif(self):
        """
        Helper function for updating audio interface parameters based on
        user selection.
        """
        audioif_name = self.comboAudioIF.currentText().split(':')[1]
        self.audioif_ind = int(self.comboAudioIF.currentText().split(':')[0])
        info = self.pa.get_device_info_by_index(self.audioif_ind)
        print("New Audio IF selected")
        print("  Name: " + audioif_name)
        print("  Sample Rate: " + str(info['defaultSampleRate']))
        # number of channels (input or output) dictated by the run mode
        if self.runmode=='wav':
            numch = info['maxOutputChannels']
            print("  Num Ch (output): " + str(numch))
        else:
            numch = info['maxInputChannels']
            print("  Num Ch (input): " + str(numch))

        # update audio channels combo box
        prev_isplaying = self.is_playing
        self.is_playing = False
        self.comboAudioCh.clear()
        for i in range(numch):
            self.comboAudioCh.addItem(str(i))
        self.is_playing = prev_isplaying
        # force index 0, assuming it always exists
        self.comboAudioCh.setCurrentIndex(0)

    def update_audioch(self):
        """
        Helper function for updating audio interface current channel based on
        user selection.
        """
        try:
            ch = int(self.comboAudioCh.currentText())
            print("Current channel = " + str(ch))
            if ch > -1:
                #self.current_ch = ch
                # TODO: fix this
                self.current_ch = 0
                self.settings["audioch"] = self.current_ch
        except:
            print("Failed to assign audio channel: " + self.comboAudioCh.currentText())
    
    def closeEvent(self, event):
        print ("Closing window")
        for convmp3 in self.converted_mp3:
            try:
                os.remove(convmp3) 
            except:
                print("Failed to remove " + convmp3)
        self.canvas.close_event()
        if self.stream:
            print("Stopping stream")
            self.stream.stop_stream()
            self.stream.close()
        self.pa.terminate()
        # save settings
        with open(self.SETTINGS_PATH, 'w') as fp:
            json.dump(self.settings, fp)
        event.accept()

def main():
    app = QApplication(sys.argv)
    form = PyPlotAudio()
    form.show()
    app.exec_()

if __name__ == '__main__':
    main()
    