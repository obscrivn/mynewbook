## Topic Modeling

<iframe width="560" height="315" src="https://www.youtube.com/embed/-pgNW8Zr_fg" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

<object data="_static/topic-modeling.pdf" width="950" height="650" type='application/pdf'/></object>

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
 
 ### Practice Collocations
 
 Google Colab: [NLTK Collocations](https://colab.research.google.com/drive/1uiflgnsTjirMEW2WlfKKK8jIcRSZebpT?usp=sharing)
