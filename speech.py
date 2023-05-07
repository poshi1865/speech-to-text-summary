import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from collections import defaultdict


import speech_recognition as sr
from pydub import AudioSegment
from pydub.utils import make_chunks

def summarize_text(text):
        
    # tokenize the text into sentences and words
    sentences = sent_tokenize(text)
    words = word_tokenize(text)

    # remove stop words from the words list
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if not word in stop_words]

    # compute word frequency distribution
    word_frequencies = FreqDist(filtered_words)

    # calculate sentence scores based on word frequency
    sentence_scores = defaultdict(int)
    for sentence in sentences:
        for word in word_tokenize(sentence.lower()):
            if word in word_frequencies:
                sentence_scores[sentence] += word_frequencies[word]

    # get top 3 sentences with highest scores
    summary_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:3]

    # combine the summary sentences into a summary paragraph
    summary = ' '.join(summary_sentences)

    # print the summary
    print(summary)


# create a recognizer object
r = sr.Recognizer()

# load the audio file and convert to WAV format
filename = "marketplace_full.mp3"
audio = AudioSegment.from_file(filename, format="mp3")
audio = audio.set_channels(1)  # convert stereo to mono
audio = audio.set_frame_rate(16000)  # set the frame rate to 16 kHz
chunks = make_chunks(audio, 10000)  # split the audio into chunks of 10 seconds
audio_chunks = [chunk.get_array_of_samples() for chunk in chunks]

# recognize speech using Google Speech Recognition
try:
    text = ""
    for chunk in audio_chunks:
        audio_data = sr.AudioData(bytes(chunk), 16000, 2)
        text += r.recognize_google(audio_data, language='en-US', show_all=False)

    print(f"You said:\n {text}.")
    
    print("SUMMARIZED TEXT:")
    summarize_text(text)

except sr.UnknownValueError:
    print("Google Speech Recognition could not understand your speech.")
    
except sr.RequestError as e:
    print(f"Could not request results from Google Speech Recognition service; {e}")