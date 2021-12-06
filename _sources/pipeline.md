## NLP Pipeline

<object data="_static/nlp-pipeline.pdf" width="950" height="650" type='application/pdf'/></object>

### Step 1. Text Cleaning

- Is text cleaning task-specific?

- Is text cleaning language-specific?

- How should we tokenize **doesn't**?


### Step 2. Tokenization

- Should we keep sentences or only tokenize words?

- What is the formula for Lexical Diversity?

- Why would we need to calculate *Lexical Diversity*?


### Terminology

**Token** -  a sequence of characters

**Occurrence** - a count of tokens

**Vocabulary** - a set of unique tokens (``set()``)

**Corpus** (plural corpora) - a collection of texts

``to be or not to be``

- How many tokens (non-unique)?

- Count tokens occurrencies

- How many unique tokens?

![](_static/diversity.png)


### Corpora

Read [Corpora in NLP](https://devopedia.org/text-corpus-for-nlp)

- What is Corpus?

- What are some generic training corpora for NLP?

- What are task-specific corpora?

- Which are some corpora for non-English languages?

- Derived from text corpus, which datasets are useful for NLP tasks?

#### Corpus Annotations (Sarkar p. 51)

- **POS tagging**: This is mainly used to annotate each word with a POS tag indicating the part of speech associated with it.

- **Word stems**: A stem for a word is a part of the word to which various affixes can be attached.

- **Word lemmas**: A lemma is the canonical or base form for a set of words and is also known as the head word.

- **Dependency grammar**: These include finding out the various relationships among the components in sentences and annotating the dependencies.

- **Constituency grammar**: These are used to add syntactic annotation to sentences based on their constituents, including phrases and clauses.

- **Semantic types and roles**: The various constituents of sentences, including words and phrases, are annotated with specific semantic types and roles often obtained from an ontology that indicates what they do. These include things like place, person, time, organization, agent, recipient, theme, etc.



