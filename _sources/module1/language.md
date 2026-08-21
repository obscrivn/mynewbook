# 1.1 From Language To Data

## What is Natural Language?

**Natural language** is the language people use to communicate with one another, such as English, Spanish, Arabic, Mandarin, or Hindi. Unlike programming languages, natural languages were not designed according to a **fixed set of computational rules**. They evolved through human communication, culture, and social interaction.

This makes language remarkably **expressive**, but also remarkably **difficult** for computers to process.

Try out these challenges:
**Challenge 1 - Coreferences**

![](_static/challenge1.png)

Question: Who is Mr.Robin?

**Challenge 2 - Ambiguity**
```{figure} ../_static/challenge2.png
:name: challenge2
NLP challenge example 1.
```

![](../_static/challenge2.png)

![](../_static/challenge2b.png)

**Challenge 3 - Informal Communication**

Do you know these abreviations?

- ROFL
- ICYMI
- TL;DR
- NVM
- TGIF
- TBH

*Answers*
ROFL: Rolling on floor laughing.
ICYMI: In case you missed it.
TL;DR: Too long, didn't read.
LMK: Let me know.
NVM: Nevermind.
TGIF: Thank goodness it's Friday.
TBH: To be honest.

## What is NLP?

<object data="../_static/demo.pdf" width="950" height="650" type='application/pdf'/></object>

**Natural Language Processing**, or NLP, is the area of artificial intelligence concerned with computational methods for **analyzing**, **representing**, **understanding**, and **generating** human language. NLP includes tasks such as text classification, information extraction, search, question answering, translation, summarization, sentiment analysis, speech recognition, and text generation.

### NLP vs LLM: What’s the difference?

Modern NLP increasingly includes large language models (LLMs), but NLP is broader than LLMs.

```{note}
**Big idea:** Modern generative AI did not make traditional NLP obsolete. LLMs still operate on text representations, tokens, learned language patterns, similarity, classification-like decisions, information retrieval, and other concepts developed throughout NLP.
```

- LLMs are a powerful subset of NLP models characterized by their massive size, extensive training data, and ability to perform a wide range of language tasks with minimal task-specific training. Models like the Llama, GPT, or Claude series are examples of LLMs that have revolutionized what’s possible in NLP.
Source: https://huggingface.co/docs/course/chapter1/1 

### NLP Applications

<object data="_static/nlp-applications.pdf" width="950" height="650" type='application/pdf'/></object>

- Sentiment
- Name entities
- Translation
- NLP Inference
- Semantic Role Labeling
- Relation Extraction
- Semantic Parsing
- Machine Comprehension
- Textual Entailment
- Coreference Resolution
- Searching
- Question Answering
- Speaker Identification
- and more...


## Terminology

**Natural Language Processing (NLP)**
The broad field concerned with computational methods for working with human language. NLP includes tasks such as classification, information extraction, search, translation, summarization, speech recognition, and text generation.

**Natural Language Understanding (NLU)**
A subset of NLP focused on interpreting language. NLU tasks may involve identifying intent, extracting entities, determining sentiment, resolving meaning, or answering questions about text.

**Natural Language Generation (NLG)**
A subset of NLP focused on producing human-readable language. Examples include generating summaries, responses, reports, translations, and conversational output.

**Language Model (LM)**
A computational model that learns patterns in language and assigns probabilities to sequences of words or tokens. Modern language models are commonly trained to predict tokens based on surrounding context.

**Large Language Model (LLM)**
A language model trained on very large datasets with a large number of learned parameters. Modern LLMs are typically based on transformer architectures and can perform many NLP tasks through prompting, fine-tuning, retrieval, or other forms of adaptation.

**Generative AI**
A broader category of artificial intelligence systems designed to generate new content, including text, images, audio, video, and code. LLMs are one important type of generative AI, but generative AI is not limited to language.

**Transformer**
A neural network architecture introduced in 2017 that uses an attention mechanism to model relationships between elements in a sequence. Transformers became the foundation of most modern large language models.

**Token**
A unit of text processed by a language model. A token may represent a word, part of a word, punctuation, or another text unit depending on the tokenizer.

**Embedding**
A numerical vector used to represent a word, token, sentence, document, or other object. Embeddings allow computational systems to compare and process language mathematically.