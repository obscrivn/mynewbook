# 4.2 Discussion Experiment: From Similarity Scores to Retrieval

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/obscrivn/mynewbook/blob/master/module4/week4_coding_practice.ipynb)

[Download the student notebook](week4_coding_practice.ipynb)

Week 03 showed how text becomes sparse lexical vectors and dense embeddings. This activity asks the next question: once texts are vectors, which documents should a system retrieve for a query, and how much should we trust the ranking?

## Learning objectives

By the end of the activity, you will be able to:

- compare Jaccard overlap, TF-IDF cosine similarity, and dense-vector cosine similarity;
- rank candidate texts against a query;
- explain why representation choice changes retrieval results;
- test how a small query change affects rank order;
- identify a result that is similar but not useful for the task;
- explain how a retrieval error could affect a downstream AI system.

## Activity workflow

The notebook follows this sequence:

**query -> representation -> similarity scores -> ranking -> interpretation**

You will first predict which texts a human would choose. You will then build lexical and dense-vector rankings, compare them with your prediction, change the query, and evaluate a deliberately challenging retrieval result.

The five core candidates and reflection questions match the Week 04 discussion. Keep notes at each **Discussion evidence checkpoint** in the notebook; those observations will help you support your initial post with results rather than definitions alone.

## Setup

The text collection is defined inside the notebook, so no dataset or account is required. A setup cell installs a pinned spaCy model the first time it runs in a clean Colab session. The remaining analysis uses NumPy, pandas, and scikit-learn.

Cosine similarity is the main vector metric in this activity. Jaccard provides a lexical baseline. Clustering appears only as a downstream application; it is not part of this practice.
