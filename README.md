# PyPlotAudio

PyPlotAudio is a graphical audio analyzer based on QT5, soundfile, numpy and scipy.

Python 3(.6.9) is used. The dependencies are listed in the requirements.txt and can be installed with pip.

WAV files, test signals and sound device streams can be read and analyzed in real time.
The user is able to tweak a few parameters and observe their influence in the spectrum 
straight away.

# GUI

The GUI is done on PyQt5 Designer. For converting into the respective python file:

`pyuic5 analyzer.ui -o gui.py`

# TODO:
- Overlap
- Strech plot when resizing window
- Better soundcard handling, as now only channel 0 is used
