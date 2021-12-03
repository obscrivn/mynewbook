# Natural Language Processing w/ NLTK (Python3)

## Pre-flight
(assumes a working Anaconda installation)

1. Create an `Anaconda` environment for this workshop: `conda create -n scraping python=3.8`
2. Activate the `scraping` environment: `conda activate scraping`
3. Install `nltk` to our environment: `pip install nltk`
4. Instal `matplotlib` to our environment: `pip install matplotlib`
5. Run `jupyter notebook` in our environment: `jupyter notebook`
6. Open a new `Python 3` `Notebook`
7. In a `notebook` cell: `import nltk`
8. Download `nltk` book: `nltk.download()`, find and select `book` and hit `download`


## Getting a Text

1. Get a sample text! I like using [Project Gutenberg](https://www.gutenberg.org/) for this.
2. In the search field, search for an older book that you think might be in Project Gutenberg. For example, one could find Herman Melville's Moby Dick or Lewis Carroll's Alice's Adventures in Wonderland. Take a moment to find a text to work with or, if you want, you can just follow along with me to [Alice's Adventures in Wonderland](https://www.gutenberg.org/ebooks/11)
3. We are looking for a `.txt`, or Plain Text, file. Once you have found it click on it to open it in your browser. For example: the Plain Text version of Alice's Adventures in Wonderland is [here](https://www.gutenberg.org/files/11/11-0.txt)
4. Next we open our `Terminal` and use `curl` to download the `.txt` file from the url, saving the output to a text file on our CPU: `curl https://www.gutenberg.org/files/11/11-0.txt --output alice.txt`


## Preparing the Text for NLTK
1. In Python Import `nltk`: `import nltk`
2. Import your text file into Python:

    ```python
    filename = "alice.txt"
    f = open(str(filename), 'r')

    corpus = f.read()
    f.close()
    ```
3. Before moving forward let's double check to make sure the text file looks right: `print(corpus)`


## Tokenizing the Text
1. A text is composed of tokens: words, sentences, punctuation, etc.
    `nltk` provides a variety of pre-built functions for `tokenization`, but the two that you will see used most frequently are `word` and `sentence` tokenization.

    We can split our corpus into words with

    ```python
    from nltk.tokenize import sent_tokenize, word_tokenize

    print(word_tokenize(corpus))
    ```

    or sentences with

    ```python
    from nltk.tokenize import sent_tokenize, word_tokenize

    print(sent_tokenize(corpus))
    ```
2. Run both of the tokenization processes again and store them to variables for usage later:
    ```python
    from nltk.tokenize import sent_tokenize, word_tokenize

    alice_sents = sent_tokenize(corpus)
    alice_words = word_tokenize(corpus)
    ```
3. Check the contents of `alice_sents` and `alice_words` with `print`
    * `print(alice_sents)`
    * `print(alice_words)`


## Analyzing and Cleaning the Text

### Frequency Analysis

1. `nltk` has a built-in funciton for analyzing the frequency words occur in a corpus: `.FreqDist()`. Run it on your list of words and store it to a variable called `freq`: `freq = nltk.FreqDist(alice_words)`
2. check the output of `FreqDist()`: `print(freq)`
3. for a more specific example of the problem here: `freq.most_common(5)`
   the code above results in the following (assuming you are using Alice's Adventures in Wonderland as input): `[(',', 2574), ('the', 1686), ('“', 1118), ('”', 1114), ('.', 911)]`
4. We can modify our code a bit to try to condition, or clean, our data:

    ```python
    from nltk.corpus import stopwords
    sr= stopwords.words('english')
    clean_tokens = alice_words

    for token in alice_words:
        if token in stopwords.words('english'):
            msg = ' '.join(['removed', str(token), 'from corpus'])
            print(msg)
            clean_tokens.remove(token)

    freq = nltk.FreqDist(clean_tokens)

    for key,val in freq.items():
        print(str(key) + ':' + str(val))

    freq.plot(20, cumulative=False)
    ```

Even when we filter out `stopwords`, though, we still have a ton of `symbols` left over to clean out. Perhaps another approach would be better!


### Parts of Speech Analysis

1. Run the parts of speech tagger in `nltk` and store the output to a variable called `alice_pos`: `alice_pos = nltk.pos_tag(alice_words)`
2. Check to make sure that the output looks sane: `print(alice_pos)`
3. The parts of speech tagger outputs a `tuple`, an immutable (un-changeable) datatype that resembles a list. It's a lot easier to deal with lists in Python than tuples (just trust me on this) so the first step is to split the tuple into a 2-dimensional `list`: `alice_pos = list(map(list, alice_pos))`
4. Next, we repurpose our previous example to identify every instance of a period, which we eventually want to remove, but will test first with the following:
    ```python
    clean_pos = alice_pos

    for token in alice_pos:
        if token[1] == ".":
            msg = ' '.join([str(token[0]), "is a ."])
            print(msg)
    ```
5. The above code reliably identifies ends of sentences (apparently also includes `!` and `?`), so lets go ahead and remove each token that meets this criteria from `clean_pos`:
    ```python
    clean_pos = alice_pos

    for token in alice_pos:
        if token[1] == ".":
            clean_pos.remove(token)
    ```
6. Check to make sure our output looks sane (there shouldn't be any `.`): `clean_pos`
7. So right now we could iteratively update the symbol to look for, or we could rewrite our code slightly. In the following we have a list of `symbols` and check every token against that list. If a token matches *anything* on a symbol list we remove it from our `clean_pos`:
    ```python
    clean_pos = alice_pos
    symbols = ['$',"''",'(',')',',','--','.',':','SYM',"``", '#']


    for token in alice_pos:
        for symbol in symbols:
            if symbol == token[1]:
                clean_pos.remove(token)
    ```
8. Check to make sure our output looks sane (there should be substantially less symbols): `clean_pos`
9. And now lets remove all stopwords:
    ```python
    from nltk.corpus import stopwords
    cleaner_pos = clean_pos

    for token in clean_pos:
        if token[0] in stopwords.words('english'):
            msg = ' '.join([str(token[0]), "is a stopword"])
            print(msg)
            cleaner_pos.remove(token)
    ```
11. And now let's unzip our list and trying running `FreqDist()` again and graphing our results:
    ```python
    a,b=zip(*cleaner_pos)
    list(a)

    freq = nltk.FreqDist(a)
    freq.plot(20, cumulative=False)
    ```
12. Still getting some information in the output that we don't really want, lets try throwing out everything but nouns next:
    ```python
    only_nouns = []
    nouns = [ 'NN', 'NNP', 'NNPS', 'NNS']

    for token in cleaner_pos:
        for symbol in nouns:
            if symbol == token[1]:
                only_nouns.append(token[0])
    ```
13. Finally, lets graph our noun results:
    ```python
    freq = nltk.FreqDist(only_nouns)
    freq.plot(20, cumulative=False)
    ```
