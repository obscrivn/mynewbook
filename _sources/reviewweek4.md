## Review Tokenization

#### Qestion 1. TF-IDF #### 

Write the formula for TF-IDF (no smoothing), make sure to describe each component:


#### Qestion 2. Lexical Diversity #### 

Corpus size = 100. Number of unique words = 75. What is the lexical diversity? And what does it mean?


#### Qestion 3. Constituency Tree #### 

Draw a constituency tree (include POS and Syntactic tags): I bought a new computer

#### Qestion 4. Dependency Tree #### 

Draw a dependency structure for the same sentence from Q3. Remember the arrows are unidirectional from the head to a dependent word. The root is typically a verb. Head can have many outcoming edges, the dependent word only has one incoming edge.


#### Qestion 5. Pipeline #### 

Write a standard pipeline for NLP tasks – use a linear sequence pipeline. Your starting point is Raw Text as Input:


#### Qestion 6. Lemmatization / Stemming ####

Your next NLP task requires a good recall but does not really need a good precision. Which preprocessing technique will you choose and why: lemmatizing or stemming?


#### Qestion 7. Lexical Diversity #### 

Your raw text has a lot of repetition in tokens, e.g. _soooooo_, _woooow_. Review Ch3 (Sarkar) and explain how the following code works:

#1
```
repeat_pattern = re.compile(r'(\w)\1\1*')
match_substitution = r'\1\1'
new_text = repeat_pattern.sub(match_substitution, text)
```
#2
```
repeat_pattern = re.compile(r'(\w)\1\1*')
match_substitution = r'\1'
new_text = repeat_pattern.sub(match_substitution, text)
```
What would be the output for  #1 and #2 for **“soooo cooooool waaayyyy”**

#### Qestion 8. Contraction Map #### 

In Week 2 Practice and slides you were provided an example of Contractions Mapping using regex.
```
replacement_patterns = [
(r'won\'t', 'will not'),
(r'(\w+)(n\'t)', r'\g<1> not'),
(r'(\w+)(\'ve)', r'\g<1> have'),
(r'(it)(\'s)', r'\g<1> is'),
(r'(\w+)(\'re)', r'\g<1> are'),
(r'(\w+)(\'d)', r'\g<1> would')
]

class RegexpReplacer(object):
    def __init__(self, patterns=replacement_patterns):
        self.patterns = [(re.compile(regex), repl) for (regex, repl) in patterns]
    def replace(self, text):
        s = text
        for (pattern, repl) in self.patterns:
            (s, count) = re.subn(pattern, repl, s)
        return s
```
Explain regex construct in ```replacement_patterns``` and ```re.subn()``` in RegexReplacer()
