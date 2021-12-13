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






