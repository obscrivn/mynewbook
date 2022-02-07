## Machine Translation

<small>Source. Jurafsky. Chapter 26</small>

<small>Source: [Human Auditory Range](http://www.cochlea.org/en/hear/human-auditory-range)</small>


> Machine Translation (MT) is the task of automatically converting one natural language into another, preserving the meaning of the input text, and producing fluent text in the output language ([NLP Stanford Grooup](https://nlp.stanford.edu/projects/mt.shtmls))

### History

- Machine Translation (MT) was one of the first applications envisaged for computers

- First demonstrated by IBM in 1954 with a basic word-for-word translation system

> Warren Weaver (1949): I have a text in front of me which is written in Russian but I am going to pretend that it is really written in English and that it has been coded
in some strange symbols. All I need to do is strip off the code in order
to retrieve the information contained in the text.
<small>Adapted from Stephen Clark Slides. Machine Translation. 2022. [Link](https://www.cl.cam.ac.uk/teaching/0809/NLP/slides-steve.pdf) </small>

### Chalengess

Q1. Why Machine Tranlation is hard?

- Word order
- Word sense
- Pronouns
- Tense
- Idioms

#### Word Orders

![](_static/tr1.png)


#### Word Sense Ambiguity
![](_static/tr2.png)

#### Correference
![](_static/tr9.png)

#### Pronouns
![](_static/tr3.png)

Q2. Do you know any other pro-drop language? How some languages encode information when the personal pronoun is omitted?


![](_static/tr7.png)

Q3. Use Google translate for **I saw the movie and it is good**. Does your translation is correct for **it**? Copy the translation  and translate it back to English.

#### Different Tenses
![](_static/tr4.png)

#### Idioms

Q4. Use Google translate and find a translation in aanotherr language for ```spill the beans```

![](_static/tr5.png)

### Approaches
![](_static/tr6.png)

#### Rule-Based Translation

> Classical machine translation methods often involve rules for converting text in the source language to the target language. The rules are often developed by linguists and may operate at the lexical, syntactic, or semantic level. This focus on rules gives the name to this area of study: Rule-based Machine Translation, or RBMT.
![](_static/tr11.png)

- Foreign input is segmented in phrases
- Each phrase is translated into English
- Phrases are reordered

```{note}
The key limitations of the classical machine translation approaches are both the expertise required to develop the rules, and the vast number of rules and exceptions required.
```



#### Statistical Machine Translation
> Statistical machine translation, or SMT for short, is the use of statistical models that learn to translate text from a source language to a target language gives a large corpus of examples.

- Find most probable English sentence given a foreign language sentence
- Automatically align words and phrases within sentence pairs in a parallel corpus
- Probabilities are determined automatically by training a statistical model using the parallel corpus

![](_static/tr10.png)

#### Neural Machine Translation
> Neural machine translation, or NMT for short, is the use of neural network models to learn a statistical model for machine translation.


![](_static/tr12.png)

### Challengess

![](_static/tr13.png)



