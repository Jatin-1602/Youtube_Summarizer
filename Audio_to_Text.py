# Reference : https://github.com/snakers4/silero-vad
# Silero VAD supports 8000 Hz and 16000 Hz sampling rates.
# Model was trained on 30 ms. Longer chunks are supported directly, others may work as well.
# Used for getting chunks of large audio
"""
Reason for selecting silero-vad :
a. Normally dividing audio based on timestamps, it may be possible that at cutting point,
   person is speaking some word and word gets cut in half way.
b. silero-vad overcomes the above drawback and divides audio in such a way that every word
   is complete in each chunk
"""


SAMPLING_RATE = 16000

import torch
torch.set_num_threads(1)
# from pprint import pprint

# Reference : https://github.com/snakers4/silero-vad/blob/master/silero-vad.ipynb
# Load "silero-vad" model
model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_vad',
                              force_reload=True,
                              onnx=False)

(get_speech_timestamps,
 save_audio,
 read_audio,
 VADIterator,
 collect_chunks) = utils


# get speech timestamps from full audio file
# Reference : https://github.com/snakers4/silero-vad/blob/master/utils_vad.py
def collect_chunks(tss: dict,
                   wav: torch.Tensor):
    chunks = []
    chunks.append(wav[tss['start']: tss['end']])
    return torch.cat(chunks)

# Input Audio File (.wav format)
wav = read_audio('wav_format/a1.wav', sampling_rate=SAMPLING_RATE)

speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=SAMPLING_RATE)
pprint(speech_timestamps)

# Saving Chunks of Audio
for i in range(len(speech_timestamps)):
    save_audio(f'wav_format/a1_chunks/a1_chunk{i}.wav',
               collect_chunks(speech_timestamps[i], wav), sampling_rate=SAMPLING_RATE)




# -------------------------------------------------------------------------------------
# vad_iterator = VADIterator(model)
# wav = read_audio(f'wav_format/a1.wav', sampling_rate=SAMPLING_RATE)
#
# window_size_samples = 1536 # number of samples in a single audio chunk
# for i in range(0, len(wav), window_size_samples):
#     chunk = wav[i: i+ window_size_samples]
#     if len(chunk) < window_size_samples:
#       break
#     speech_dict = vad_iterator(chunk, return_seconds=True)
#     if speech_dict:
#         print(speech_dict, end=' ')
# vad_iterator.reset_states() # reset model states after each audio