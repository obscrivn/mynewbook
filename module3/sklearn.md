# Coding Practice: Inspecting Text Feature Vectors

This formative notebook turns a small text corpus into numerical features with scikit-learn. You will inspect the vocabulary and document-term matrix, compare representations, and explain what each representation preserves or loses.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/obscrivn/mynewbook/blob/master/module3/week3_coding_practice.ipynb)

[Download the notebook](week3_coding_practice.ipynb) if you prefer to run it in a local Jupyter environment.

## Learning objectives

By the end of the activity, you should be able to:

- connect vocabulary terms to columns in a document-term matrix;
- interpret matrix shape and individual feature values;
- compare count, binary, n-gram, and TF-IDF features;
- validate a vectorizer workflow and identify an unsafe refit;
- explain why sparse lexical vectors motivate—but are not replaced by—dense embeddings.

## What you will do

The notebook follows a worked-example-to-independent-practice progression:

1. predict features before running a vectorizer;
2. build and label a count matrix;
3. compare count and binary features;
4. explore which local word order bigrams preserve;
5. compare counts with TF-IDF weights and inspect smoothing;
6. critique a feature-space mismatch;
7. choose and justify a representation for a new corpus.

The corpus is included directly in the notebook. No downloads, accounts, API keys, or installation cells are required in Google Colab.

## References

- [CountVectorizer documentation](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)
- [TfidfVectorizer documentation](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
