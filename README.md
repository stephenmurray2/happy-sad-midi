# Happy Sad Midi

Make music... with your face! This python code will allow you play chords on an external midi device using facial expressions detected via webcam. Notable details:

- For computer vision, this uses an (unofficial) pre-built version of OpenCV: [opencv-python](https://pypi.org/project/opencv-python/)
- Uses a convolutional neural network framework from [TensorFlow](https://www.tensorflow.org/)
- Uses the facial expression classifer: [FER](https://pypi.org/project/fer/)
- MIDI events are programmatically generated with [python-rtmidi](https://pypi.org/project/python-rtmidi/)

## Installation

After cloning the repo locally, creating a virtual environment (recommended), and entering the root directory of this repo, install the dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Ensure you have a functioning webcam, and that your computer is hooked up to some external midi device. Simply run:

```bash
python main.py
```

## License

[MIT](https://choosealicense.com/licenses/mit/)
