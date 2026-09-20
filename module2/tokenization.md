# 2.1 From Text to Tokens

> A tokenizer is not a universal preprocessing tool. Its vocabulary and segmentation choices affect efficiency, usable context, and potentially model performance across languages.

## What Is a Token?
**What counts as a word?** 
Whitespace seems obvious until you encounter punctuation, contractions, compounds, URLs, emoji, hashtags, clitics, or languages without whitespace word boundaries.

**What counts as a token?**
Depending on the tokenizer, a token may represent:

- a whole word
- part of a word
- punctuation
- a character
- or, in some modern systems, a byte

**Tokenization**: The process of splitting text into smaller units (tokens).	All NLP models rely on tokens as their “language,” so knowing how they are formed helps with model usage, efficiency, and fairness.

```{important}
Tokens do not strictly align with orthographic words. They can represent a flexible mix of whole words, subword fragments, single characters, or individual bytes
```

## Word Tokenization

The simplest approach to tokenization is to treat words as the basic units of text.

```{note}

>The students studied NLP.

a simple tokenizer might produce:

["The", "students", "studied", "NLP", "."]
```
However, whitespace alone is not enough to identify words reliably. Traditional NLP systems often use rule-based tokenizers or regular expressions to identify these boundaries.

**This is one reason regular expressions remain useful in NLP even when working with modern language models.**


## Morphological Processing

Tokenization identifies units in text. Morphological processing helps us understand how those units are related.

> **Morphology** is the study of how words are formed from smaller meaningful components.

Consider these related word forms:

`walk`  
`walks`  
`walked`  
`walking`

A computer initially sees these as different strings, even though they share the same underlying meaning.

Two common NLP techniques used to reduce this variation are **stemming** and **lemmatization**.

### Stemming

**Stemming** removes or modifies word endings using relatively simple rules.

Examples:

`connected → connect`  
`connecting → connect`  
`connection → connect`

A stem does not always correspond to a valid dictionary word. The goal is usually to reduce related word forms to a common representation.

### Lemmatization

**Lemmatization** maps an inflected word to its dictionary base form, called its **lemma**.

Examples:

`cars → car`  
`running → run`  
`was → be`

Lemmatization usually requires more linguistic information than stemming because the correct base form may depend on the word’s grammatical role.

```{note} Stem vs. Lemma

A **stem** is a computationally reduced form.

A **lemma** is a linguistically valid base form.
```

## Part-of-Speech Tagging

> Words do more than carry meaning. They also play different **grammatical roles** in a sentence.

**Part-of-speech (POS) tagging** is the process of assigning a grammatical category to each token in a sequence.

Common part-of-speech categories include:

- nouns
- verbs
- adjectives
- adverbs
- pronouns
- determiners
- prepositions

```{note}
`book`

In the sentence:

`I read a book.`

`book` functions as a **noun**.

But in:

`Book the flight.`

`book` functions as a **verb**.

The word itself has not changed. Its grammatical role depends on the surrounding context.
```


```{important} 

Tokenization and POS tagging solve different problems.

**Tokenization** asks:

> Where are the units in the text?

**POS tagging** asks:

> What grammatical role does each unit play in this context?
```
### The Penn Treebank Tagset

One influential annotation scheme for English is the Penn Treebank Part-of-Speech Tagset.

It uses short labels to represent grammatical categories.


| Tag   | Meaning          | Example   |
| ----- | ---------------- | --------- |
| `NN`  | singular noun    | `book`    |
| `NNS` | plural noun      | `books`   |
| `VB`  | base-form verb   | `book`    |
| `VBD` | past-tense verb  | `booked`  |
| `JJ`  | adjective        | `useful`  |
| `RB`  | adverb           | `quickly` |
| `DT`  | determiner       | `the`     |
| `PRP` | personal pronoun | `she`     |


### Tagging as a Sequence-Labeling Problem

POS tagging is not simply a lookup task. The correct label for a token often depends on the words that appear before and after it.

> I can/MD swim vs I opened a can/NN

n the first sentence, can functions as a **modal verb**.

In the second sentence, can functions as a **noun**.

Each token is assigned a grammatical label, but the surrounding sequence helps determine which label is appropriate.

So far, we have focused mostly on **word-oriented tokenization and linguistic processing**.

In the next section, we move to **modern tokenization** and compare different ways of dividing text into computational units, including:

- word-based tokenization
- character-based tokenization
- subword tokenization
- Byte Pair Encoding (BPE)
- WordPiece
- Unigram and SentencePiece

We will also examine how tokenizers map text to **token IDs**, why different models tokenize the same text differently, and how tokenization affects **context windows, efficiency, and model behavior**.