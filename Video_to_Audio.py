from pytube import YouTube
from pydub import AudioSegment

# Youtube Video Link
VIDEO_URL = "https://youtube.com/watch?v=hWLf6JFbZoo"

try:
    # object creation using YouTube
    # which was imported in the beginning
    yt = YouTube(VIDEO_URL)
except:
    print("Connection Error")


# To query the streams that contain only the audio track and download as "filename"
audio_stream = yt.streams.filter(only_audio=True).first().download("Audios", filename='a1.mp4')


# CONVERT mp4 TO wav FORMAT
"""
WAV files aren't compressed when encoded. 
That means all of the original audio elements stay in the file. 
Audio editors describe WAV files as “lossless” because you don't 
lose any part of your audio. 
As a result, WAV files objectively have better quality and 
provide more true and accurate audio clips.
"""

"""
# shell command
import subprocess
command = "ffmpeg -i C:/Users/jatin/PycharmProjects/NLTK_Summariser/a1.mp4 -ab 160k -ac 2 -ar 44100 -vn C:/Users/jatin/PycharmProjects/NLTK_Summariser/a1.wav"
subprocess.call(command, shell=True)
"""
load_audio = AudioSegment.from_file('Audios/a1.mp4', format="mp4")
load_audio.export("wav_format/a1.wav", format="wav")
