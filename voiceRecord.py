from pyaudio import PyAudio,paInt16
from datetime import datetime
import wave, os

TIME = 6
NUM_SAMPLES = 2000  
framerate = 16000  
channels = 1  
sampwidth = 2 

def save_wave_file(filename, data):
    '''save the date to the wav file'''
    framerate = 16000
    channels = 1
    sampwidth = 2
    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(sampwidth)
    wf.setframerate(framerate)
    wf.writeframes(b"".join(data))
    wf.close()

def record_wave(filename='', prefix='wav/'):
    '''open the input of wave
    '''
    pa = PyAudio()
    stream = pa.open(format = paInt16, channels = 1,
                    rate = framerate, input = True, 
                    frames_per_buffer = NUM_SAMPLES)
    save_buffer = []
    count = 0
    while count < TIME * 8:
        string_audio_data = stream.read(NUM_SAMPLES)
        save_buffer.append(string_audio_data)
        count += 1
        print('recording...')

    if filename == '':
        filename = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
    filename += ".wav"
    save_wave_file(os.path.join(prefix, filename), save_buffer)
    save_buffer = []
    return filename