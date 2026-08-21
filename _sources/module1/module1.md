# Chapter 1. Natural Language

**NLP in the Age of Generative AI**

Natural language is one of the most familiar forms of data we encounter and one of the most difficult to model computationally.

Humans routinely interpret incomplete sentences, ambiguous words, indirect requests, sarcasm, metaphor, references to earlier conversations, and assumptions that are never explicitly stated. We perform much of this interpretation almost automatically.

**What is NLP?**
Natural Language Processing (NLP) studies computational methods for representing, analyzing, understanding, and generating human language. Modern NLP spans linguistic analysis, statistical learning, neural models, information retrieval, speech technology, and large language models. {cite:p}`jurafsky2026,eisenstein2019`

```{note}
**Big idea:** Modern generative AI did not make traditional NLP obsolete.

Large language models still rely on tokens, numerical representations, learned patterns, similarity, retrieval, classification, and context. What has changed is how many of these problems are modeled and combined within increasingly general-purpose systems.
```

**What is NLP, NLU, and NLG?**
```{figure} ../_static/NLP-NLU-NLG.png
:name: NLP-NLU-NLG
Three subfields of NLP.
```

Three related terms appear frequently in language technology:

- Natural Language Processing (NLP) is the broad field concerned with computational methods for working with human language.
- Natural Language Understanding (NLU) focuses on interpreting language, such as identifying intent, extracting information, determining sentiment, or resolving meaning from context.
- Natural Language Generation (NLG) focuses on producing language, such as generating responses, summaries, reports, or translations.

This chapter begins with a simple question:

> **What makes human language so difficult for computers to process?**

```{figure} ../_static/ch1_Natural_Language.png
:name: MultipleLevels
Multiple Levels of Language from Processing to Understanding
```

We will examine how language differs from ordinary structured data, why meaning depends on context and ambiguity, and how computational systems begin transforming language into representations they can work with.

**What you’ll learn**

By the end of this chapter, you should be able to:

- explain what natural language processing is
- distinguish NLP, NLU, NLG, and LLMs
- describe why context and ambiguity make language difficult to process
- inspect text as data using basic Python string operations
- explain why simple string matching is not enough for NLP

**Why this matters**

Modern language models can generate remarkably fluent text, but fluency does not eliminate the underlying problems of language processing.

Throughout this course, you will return to core questions such as:

* How should language be divided into computational units?
* How should text be represented numerically?
* How can we measure similarity between texts?
* How can a system classify, retrieve, or generate language?
* How much context is needed to interpret meaning?

Chapter 1 establishes that foundation before we begin turning language into data a computer can process.



