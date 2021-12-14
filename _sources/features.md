## Features

<object data="_static/features.pdf" width="950" height="650" type='application/pdf'/></object>

### Vector Space

Which example has 1) **Same Words / Different Meaning**, 2) **Different Word / Same Meaning** ?

![](_static/vectorspace1.png)

<small> source: https://towardsdatascience.com/vector-space-models-48b42a15d86d </small>

 > <span style="font-size:2em;"> “You shall know a word by the company it keeps” — J.R.Firth </span>



### Count Frequency

**Count frequency** - the frequency of tokens in a text document

<span style="color:red">it was the best of times it was the worst of times</span>

**Q1**. Write the dictionary (list of unique words).

**Q2**. Provide a Count Frequency for each unique token (in a table format):
| Token1   |      Token2      |  Token3 |...|
|:----------:|:-------------:|:------:|:------:|
| count |  count | count |count |

### Term Frequency

**Term frequency** (TF) is how often a token appears in a text document, divided by how many words there are.

<span style="color:red">it was the best of times it was the worst of times</span>

**Q3**.Provide a Term Frequency for each unique token (in a table format):

| Token1   |      Token2      |  Token3 |...|
|:----------:|:-------------:|:------:|:------:|
| tf |  tf | tf |tf|


### N-Grams

**N-grams** – breaking down text into chunks. A unigram is a one word chunk. A bigram is a two word chunk. A trigram is a three word chunk.

<span style="color:red">Tim Cook is the CEO of Apple</span>

**Q4**. Write unigrams, bigrams, and trigrams of the above sentence (in a table format)
| Unigrams   |      Bigrams      |  Trigrams |
|:----------:|:-------------:|:------:|
| token1 |  token1 token2 | token1 token2 token3 |
| ... |  ... | ... |



### TF-IDF

TF-IDF is composed of 2 algorithms:

- **Term Frequency** - how common the term is

 $$ TF(t) = \frac{Number \; of \; times \; term \; t \; appears\; in \; a \; document}{Total \; number \;of\; terms \; in \;the\; document}$$

- **Inverse Document Frequency** - how unique / rare the term is

$$
IDF(t) = log{_e}(\frac{Total\; number\; of\; documents}{Number\; of\; documents\; with \;term \;t\; in \;it})
$$

Example:
![](_static/tf-idf-example.png)

<small> source: https://en.wikipedia.org/wiki/Tf%E2%80%93idf </small>

What is tf (_this_, document1)?
- Count = 1
- Total terms = 5

$$
tf(this, d1) = \frac{1}{5} =0.2 
$$

What is tf (_this_, document2)?
- Count = 1
- Total terms = 7

$$
tf(this, d2) = \frac{1}{7} =0.14 
$$


What is idf (_this_, D)?
- Total Documents = 2
- Documents with _this_ = 2
- IDF is constant per corpus

$$
idf(this, D) = log(\frac{2}{2}) = 0 
$$

What is tf-idf (_this_,d1,D)?

$$
tfidf(this, d1, D) = 0.2 * 0  = 0 
$$

What is tf-idf (_this_,d2,D)?

$$
tfidf(this, d2, D) = 0.14 * 0  = 0 
$$

**Conclusion**: word _this_ is not informative because it occures in all documents

Q1. Find TF-IDF for _example_