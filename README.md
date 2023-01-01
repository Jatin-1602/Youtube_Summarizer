# Youtube_Summarizer

## Description
Created a `NLTK summarizer model` that works as follows : <br>
* input of YOUTUBE VIDEO LINK
* convert video to audio
* generate text from audio
* Summarize the generated text
* Translation of summarized text

## Approach
Used the HUGGING FACE TRANSFORMERS <br>
* [`silero-vad`](https://github.com/snakers4/silero-vad) : For generating Audio Chunks
* [`Wav2Vec2Processor`](https://huggingface.co/transformers/v4.10.1/model_doc/wav2vec2.html#wav2vec2processor) : Pre-Trained Tokenizer
* [`Wav2Vec2ForCTC`](https://huggingface.co/transformers/v4.10.1/model_doc/wav2vec2.html#wav2vec2forctc) : Speech Recognition Model
* [`long-t5-tglobal-base-16384-book-summary`](https://huggingface.co/pszemraj/long-t5-tglobal-base-16384-book-summary) : Summarizer Model
* [`facebook/wmt19-en-ru`](https://huggingface.co/facebook/wmt19-en-ru) : Translation Models
