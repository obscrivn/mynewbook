# Bag-of-words Using scikit-learn

**scikit-learn** is a library in Python that provides many unsupervised and supervised learning algorithms.

## Bag-of-word word tokenization
> Dataset sample is represented as a string, which could be a sentence, a tweet (X), or book. 

To represent the sample:

- we split the string into a list of tokens (e.g., split by whitespace, and then lowercase the word)

- we build a vocabulary of all tokens (lowercased words) from the whole dataset. 
    - Usually a very large vocabulary
- we show how often each word in the vocabulary appears
     - We represent the string as a vector, where each entry is how often a given word in the vocabulary appears in the string

<small>Source: Jason Kuruzovich. 2020. Analytics Dojo. </small>
As each sample will only contain very few words, most entries will be zero, leading to a very high-dimensional but sparse representation.

```{note}
The method is called “bag-of-words” : the order of the words is lost entirely.
```

## CountVectorizer

Reference: [https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)



![](_static/cv.png)


```from sklearn.feature_extraction.text import CountVectorizer```

STEPS

```
# initialize
vectorizer = CountVectorizer()
# generate word counts
X = vectorizer.fit_transform(corpus)
```

Practice with Google Colab:
```
from sklearn.feature_extraction.text import CountVectorizer
corpus = [
 'This is the first document.',
 'This document is the second document.',
 'And this is the third one.',
 'Is this the first document?',
 ]
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)
vectorizer.get_feature_names_out()
array(['and', 'document', 'first', 'is', 'one', 'second', 'the', 'third',
       'this'], ...)
print(X.toarray())
[[0 1 1 1 0 0 1 0 1]
 [0 2 0 1 0 1 1 0 1]
 [1 0 0 1 1 0 1 1 1]
 [0 1 1 1 0 0 1 0 1]]
```

Q1. What are the feature names in row0: [0 1 1 1 0 0 1 0 1]?

### TfIdf Vectorizer

> A useful transformation that is often applied to the bag-of-word encoding is the so-called term-frequency inverse-document-frequency (tf-idf) scaling, which is a non-linear transformation of the word counts.


Reference:[https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)

Q2. How to set up TF (instead of Tf-Idf)?

![](_static/tf.png)

```from sklearn.feature_extraction.text import TfidfVectorizer```

STEPS for TF

```
vectorizer = TfidfVectorizer(use_idf=False)
X = vectorizer.fit_transform(corpus)
```

STEPS for TF-IDF
```
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)
```

Practice with Google Colab

```
from sklearn.feature_extraction.text import TfidfVectorizer
corpus = [
 'This is the first document.',
 'This document is the second document.',
 'And this is the third one.',
 'Is this the first document?',
 ]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)
vectorizer.get_feature_names_out()
array(['and', 'document', 'first', 'is', 'one', 'second', 'the', 'third',
       'this'], ...)
print(X.shape)
```

Smoothing

- TfidfTransformer is the smooth_idf: 

  - smooth_idf : boolean, default=True

- Smooth idf prevents zero divisions and adds one to document frequencies, as if an extra document was seen containing every term in the collection exactly once.

$$
smoothIDF(t) = log{_e}(\frac{1+ Total\; number\; of\; documents}{Number\; of\; documents\; with \;term \;t\; in \;it + 1}) + 1
$$

RECAP
> tf-idfs are a way to represent documents as feature vectors. the tf is the count of how often a particular word occurs in a given document. tf-idfs is a modification of the raw term frequencies (tf). 

https://towardsdatascience.com/tf-idf-explained-and-python-sklearn-implementation-b020c5e83275

Practice Google Colab

[https://colab.research.google.com/drive/1eo75oo80vSFdvfrooxNpWawCWGNAwYao?usp=sharing](https://colab.research.google.com/drive/1eo75oo80vSFdvfrooxNpWawCWGNAwYao?usp=sharing)


Smoothing - https://nbviewer.org/github/rasbt/pattern_classification/blob/master/machine_learning/scikit-learn/tfidf_scikit-learn.ipynb 