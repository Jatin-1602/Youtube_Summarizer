"""
    @misc {peter_szemraj_2022,
    author       = { {Peter Szemraj} },
    title        = { long-t5-tglobal-base-16384-book-summary (Revision 4b12bce) },
    year         = 2022,
    url          = { https://huggingface.co/pszemraj/long-t5-tglobal-base-16384-book-summary },
    doi          = { 10.57967/hf/0100 },
    publisher    = { Hugging Face }
}
"""

import os
import torch
import soundfile as sf

from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
# SIZE : 1.26GB
# model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")

# SAVING THE MODEL AND PROCESSOR
# model.save_pretrained('./saved_model/')
# processor.save_pretrained("./saved_model/")

# LOAD THE MODEL AND PROCESSOR
model = Wav2Vec2ForCTC.from_pretrained('./saved_model/')
processor = Wav2Vec2Processor.from_pretrained('./saved_model/')

audio_path = []
path = "wav_format/a1_chunks"
for i in range(len(os.listdir(path))):
    audio_path.append(path + "/only_speech" + str(i) + ".wav")


# AUDIO TO TEXT
full_video_text = ""
transcriptions = []
path = "wav_format/a1_chunks"
for audio in os.listdir(path):
    data, samplerate = sf.read(path + "/" + audio)

    # tokenize
    input_values = processor(data,
                             sampling_rate=16000,
                             return_tensors="pt",
                             padding="longest").input_values

    # retrieve logits
    logits = model(input_values).logits

    # take argmax and decode
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)

    transcriptions.append(transcription[0])
    full_video_text += " " + transcription[0]

print("\n\n----- FULL TEXT -----\n\n")
print(full_video_text)




# SUMMARIZING TEXT
from transformers import pipeline

# summarizer = pipeline(
#     "summarization",
#     "pszemraj/long-t5-tglobal-base-16384-book-summary",
# )

# SAVE AND LOAD THE MODEL FOR LATER USE
# torch.save(summarizer, "summarizer_model/long-t5-tglobal-base-16384-book-summary")
summarizer = torch.load("summarizer_model/long-t5-tglobal-base-16384-book-summary.pt")

# GETTING RESULT FROM THE MODEL
result = summarizer(full_video_text)
summarized_text = result[0]['summary_text']

print("----- SUMMARY -----")
print(summarized_text)


# TRANSLATING TEXT

from transformers import FSMTForConditionalGeneration, FSMTTokenizer

translator_model_name = "facebook/wmt19-en-ru"
tokenizer = FSMTTokenizer.from_pretrained("./saved_model/translation_models/" + translator_model_name + "/")
model = FSMTForConditionalGeneration.from_pretrained("./saved_model/translation_models/" + translator_model_name + "/")

input = summarized_text

input_ids = tokenizer.encode(input, return_tensors="pt", max_length=200, truncation=True)

outputs = model.generate(input_ids, max_new_tokens=200)

decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(decoded)
