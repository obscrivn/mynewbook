## sklearn


### CountVectorizer

Reference: [https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)

Q2. What are the default parameters?

![](_static/cv.png)


```from sklearn.feature_extraction.text import CountVectorizer```

STEPS

```
vectorizer = CountVectorizer()
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

Q2. What are the feature names in row0: [0 1 1 1 0 0 1 0 1]?

### TfIdf Transformer

We can use Transformer for TF or Tf-Idf
Reference:[https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html)

Q2. What are the default parameters? How to set up TF?

![](_static/tf.png)