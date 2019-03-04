# subgen
A subtitle generator for audios and videos

## Installation

1. Install SpeechRecognition

```
pip install SpeechRecognition
```


2. Install Pocketsphinx

Try these commands:

```
python -m pip install --upgrade pip setuptools wheel
pip install --upgrade pocketsphinx --user
```

If failed (e.g. "cannot find swig.exe"), go to the [Unofficial Windows Binaries for Python Extension Packages](https://www.lfd.uci.edu/~gohlke/pythonlibs/#pocketsphinx). Download the `pocketsphinx-*.whl`, and then go to the download directory, execute:

```
pip install pocketsphinx-*.whl
```

3. Run test example

The "english.wav" can be from [here](https://github.com/Uberi/speech_recognition/blob/master/examples/english.wav).

```
import speech_recognition as sr
from os import path

# obtain path to "english.wav" in the same folder as this script
AUDIO_FILE = path.join(path.dirname(path.realpath(__file__)), "english.wav")

# use the audio file as the audio source
r = sr.Recognizer()
with sr.AudioFile(AUDIO_FILE) as source:
    audio = r.record(source)  # read the entire audio file

# recognize speech using Sphinx
try:
    print("Sphinx thinks you said: " + r.recognize_sphinx(audio))
except sr.UnknownValueError:
    print("Sphinx could not understand audio")
except sr.RequestError as e:
    print("Sphinx error; {0}".format(e))
```
These codes should output

```
Sphinx thinks you said: one two three
```