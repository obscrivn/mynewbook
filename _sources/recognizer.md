## Speech Recognition
<small>Source. Jurafsky. Chapter 26</small>

Q1. Name current speech recognition systems

> Google assistant, Apple’s Siri, Amazon’s Alexa



### Challenges

Q2. Why speech recognition is difficult?

<small>Adapted from Peter Ball Slides. ASR. 2022. [Link](http://www.inf.ed.ac.uk/teaching/courses/asr/2021-22/asr01-intro.pdf) </small>

#### Linguistic Challenges

- **Speaker** Tuned for a particular speaker, or
speaker-independent? Adaptation to speaker characteristics

- **Environment** Noise, competing speakers, channel conditions (microphone, phone line, room acoustics)

- **Style** Continuously spoken or isolated? Planned monologue or spontaneous conversation?

- **Vocabulary** Machine-directed commands, scientific language, colloquial expressions

- **Accent/dialect** Recognise the speech of all speakers who speak a particular language

- **Other paralinguistics** Emotional state, social class, . . .

- **Language spoken** Estimated 7,000 languages, most with limited training resources; code-switching; language change


#### Machine Learning Challenges

- **Classification problem**: very high dimensional output space

- **Sequence-to-sequence problem**: very long input sequence 

- **Data**:

    - often noisy, with many “nuisance” factors of variation in the data
   - limited quantities of training data available (in terms of words) compared to text-based NLP

   - Manual speech transcription is very expensive

- **Complexity**: Hierachical and compositional nature of speech productionand comprehension makes it difficult to handle with a single model



### Speech Recognizer Architecture
![](_static/speech1.png)

<small>Source: https://medium.com/@haejin2909/having-problems-with-voice-recognition-youre-not-wrong-c26f9eec8d4</small>

![](_static/speech3.png)

> a recorded utterance is a sequence of feature vectors

![](_static/speech4.png)

#### Automatic Speech Recognition ASR

> The task of speech recognition (or speech-to-text) is to map acoustic waveforms to sequences of graphemes.

![](_static/speech5.png)

![](_static/speech6.png)



- The input to a speech recognizer is a series of **acoustic waves**. 
    - waves are sampled, quantized, and converted to a spectral representation like the log mel spectrum

- Two common paradigms for speech recognition: 
    - the encoder-decoder with attention model
    - models based on the CTC loss function 
    
```{note} Attention based models have higher accuracies, but models based on CTC more easily adapt to streaming: outputting graphemes online instead of waiting until the acoustic input is complete.
```
- ASR is evaluated using the Word Error Rate; the edit distance between the hypothesis and the gold transcription

#### Text-to-Speech TTS

Q3. What are different ways we can say **1750**?

![](_static/speech2.png) 



### Practice
TTS LINK [https://colab.research.google.com/drive/1p08TLT1DxPA9Navfl12fmgZd3_3Xz_Mn?usp=sharing](https://colab.research.google.com/drive/1p08TLT1DxPA9Navfl12fmgZd3_3Xz_Mn?usp=sharing)

ASR LINK [https://colab.research.google.com/drive/1Nb8dPMvQZVV97OROcU192gsJE1fRfFUv?usp=sharing](https://colab.research.google.com/drive/1Nb8dPMvQZVV97OROcU192gsJE1fRfFUv?usp=sharing)


