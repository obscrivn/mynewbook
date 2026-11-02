# Chapter 11 Speech: From Sound to Text

Every model so far has started from text. This week starts one step earlier, with the sound of someone talking. You will see how speech becomes a list of numbers a computer can use, how modern speech recognizers turn those numbers into text, and how to judge whether a transcript is good enough to trust.

- [Reading 11.1: Speech as Data, From Sound to Signal](speech_signals.md)
- [Reading 11.2: Speech Recognition, From Signal to Text](speech_recognition.md)
- [Coding Practice 11.3: Working with Speech in Python](week11_speech_practice.ipynb)

Read 11.1 first: it explains waveforms, spectrograms, and the sound units that 11.2 builds on. Focus on reading figures and on the ideas (sampling, spectrograms, alignment, and word error rate) rather than on the math. Equations, MFCCs, and extra wave physics appear only in optional collapsible notes.

Then work through the coding practice in Google Colab. It is a guided, ungraded follow-along: you turn text into speech, look at it as a waveform and a spectrogram, prepare it for a speech recognizer, transcribe it with two models, and measure the errors. An optional last part lets you try the same steps with your own voice. The page shows every output, so you can also read it without running the code.
