# Chapter 12 Multilingual NER and Machine Translation

This week puts transformers to work on two tasks that both have to look past English. Named entity recognition pulls out the people, places, and organizations in a text; machine translation rewrites a whole sentence in another language. Both depend on the same encoder and encoder-decoder ideas from Module 10, and both run into the same problem from a different angle: a name that is easy to handle in one language is not automatically easy to handle in another.

- [Reading 12.1: Multilingual Named Entity Recognition](multilingual_ner.md)
- [Reading 12.2: Machine Translation](machine_translation.md)
- [Coding Practice 12.3: Lost in Translation? Entities Across Languages](week12_multilingual_entities_practice.ipynb)

Read 12.1 first. It sets up named entities, how a transformer labels them token by token, and why doing this across languages is harder than it looks. Reading 12.2 picks up right where it leaves off: the same kind of ambiguous name that trips up a tagger can also trip up a translator, and the two readings share one running example to make that connection concrete.

Then work through the coding practice in Google Colab. It is a guided, ungraded activity with checkpoints: you run a pretrained multilingual NER model, find a sentence where it gets an entity wrong, run a pretrained translation model, and then connect the two by translating a sentence and checking whether its named entities survive the trip. A separate instructor solution notebook is available with all checkpoints completed.
