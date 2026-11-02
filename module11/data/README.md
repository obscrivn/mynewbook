# Week 11 coding practice data

Used by `module11/week11_speech_practice.ipynb` (11.3 Coding Practice: Working with Speech in Python).

| File | Content | Format |
| --- | --- | --- |
| `welcome.mp3` | Synthetic speech of the sentence "Welcome to the big hotel California", generated with the gTTS library (Google Translate text-to-speech) | MP3, 24,000 Hz, mono, 2.8 seconds |

The notebook first tries to create this file itself with gTTS. If Google's service cannot be reached, it downloads this copy instead:
`https://raw.githubusercontent.com/obscrivn/mynewbook/master/module11/data/welcome.mp3`

**Student recordings are never stored in this repository.** The optional "your own voice" part of the notebook works only with files uploaded to the student's own Colab session.
