## Neural Network

### Playground

Use the playground - http://playground.tensorflow.org/ (Links to an external site.) 

change activation, learning rate, regularization, # epochs
change problem types (classification versus regression)
change # of hidden layers
change datasets
examine loss
Question 1. Are you able to improve your training (loss is smaller)? What parameters can help you improve/optimize your training?

Question 2. What have you learned from this playground?


### Neural Zoo
<object data="_static/The_Neural_Network_Zoo.pdf" width="950" height="650" type='application/pdf'/></object>




### DL Models
<object data="_static/deep-learning-lecture.pdf" width="950" height="650" type='application/pdf'/></object>

### LSTM

<object data="_static/rnn-lstm-neural-network.pdf" width="950" height="650" type='application/pdf'/></object>


> LDA is a hierarchical Bayesian model that assumes topics are probability distributions over words, and documents are distributions over topics. The model assumes that topics follow a sparse Dirichlet distribution, which implies that documents reflect only a small set of topics, and topics use only a limited number of terms frequently.

- The percentage contribution of each topic to a document
- The probabilistic association of each word with a topic

### Key Phrase Extraction

> This is one of the simplest yet most powerful techniques of extracting important information from unstructured text documents.

Keyphrase extraction, also known as **terminology extraction** , is the process of extracting key terms or phrases from a body of unstructured text so that the core themes are captured. This technique falls under the broad umbrella of information retrieval and extraction. Keyphrase extraction is useful in many areas:

- Semantic web
- Query based search engines and crawlers
- Recommendation systems
- Tagging systems
- Document similarity
- Translation

### Concordances

> A concordance - every occurrence of a given word, together with some context. 


![](_static/concordance.png)

### Collocations
 
**Keyphrase extraction** is often the starting point for carrying out more complex tasks in text analytics or natural language processing and the output can act as features for more complex systems. There are various approaches for keyphrase extraction:

- Collocations
- Weighted tag-based phrase extraction

> A collocation a sequence or group of words that tend to occur frequently and this frequency tends to be more than what could be termed a random or chance occurrence. 

Various types of collocations can be formed based on parts of speech like nouns, verbs, and so on. There are various ways to extract collocations and one of the best ways to do it is to use an n-gram grouping or segmentation approach: 
- Construct n-grams out of a corpus 
- Count the frequency of each n-gram and rank them based on their frequency of occurrence to get the most frequent n-gram collocations
 
 #### Practice Collocations
 
 Google Colab: [NLTK Collocations](https://colab.research.google.com/drive/1uiflgnsTjirMEW2WlfKKK8jIcRSZebpT?usp=sharing)

### Topic Model Evaluation


> Unsupervised Learning: there is no objective metric to assess the quality of the result as in supervised learning. Human topic evaluation is considered the gold standard, but it is potentially expensive and not readily available at scale.

Two options to evaluate results more objectively include:

- perplexity to evaluate the model on unseen documents 
- topic coherence metrics to evaluate the semantic quality of the uncovered patterns

#### Perplexity

**Perplexity** measures how well the topic-word probability distribution recovered by the model predicts a sample of unseen text documents. It is based on the entropy H(p) - Measures closer to zero imply the distribution is better at predicting the sample.

#### Topic coherence

**Topic coherence** measures the semantic consistency of the topic model results, whether humans would perceive the words and their probabilities associated with topics as meaningful.Coherence measures are based on the probability of observing the set of words W that defines a topic together.

### LDA Practice

- Complete the following practice: [https://github.com/PacktPublishing/Machine-Learning-for-Algorithmic-Trading-Second-Edition/blob/master/15_topic_modeling/04_lda_with_sklearn.ipynb](https://github.com/PacktPublishing/Machine-Learning-for-Algorithmic-Trading-Second-Edition/blob/master/15_topic_modeling/04_lda_with_sklearn.ipynb)
- Data: Download bbc.zip file from [https://github.com/PacktPublishing/Machine-Learning-for-Algorithmic-Trading-Second-Edition/tree/master/data](https://github.com/PacktPublishing/Machine-Learning-for-Algorithmic-Trading-Second-Edition/tree/master/data)