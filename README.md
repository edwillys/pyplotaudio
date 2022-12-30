# PyPlotAudio

PyPlotAudio is a graphical audio analyzer based on QT5, soundfile, numpy and scipy.

It was developped using Python 3(.6.9), however, it should work on later versions, at least up to 3.9.x. PyQt5 doesn't work with python 3.10 and up, as of today (December 2022).

The dependencies are listed in the requirements.txt and can be installed with pip.

WAV files, test signals and sound device streams can be read and analyzed in real time.
The user is able to tweak a few parameters and observe their influence in the spectrum 
straight away.

## GUI

The GUI is done on PyQt5 Designer. For converting into the respective python file:

`pyuic5 analyzer.ui -o gui.py`

## Running
You might want to create a virtual environment, activate it and run it from there
```bash
python -m venv ven
venv/Scripts/activate
pip install -r requirements.txt
python src/AudioAnalyzer.py
```

## TODO:
- Overlap
- Strech plot when resizing window
- Better soundcard handling, as now only channel 0 is used
