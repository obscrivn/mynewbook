## Sentiment

### Supervised Methods

- $X$ is features (input), $Y$ is labels (output)
    - Often labels are binary (positive = 1, negative = 0)
- Your task is **to predict output label based on features**
- $X$ goes into your prediction function to get the output estimate $\hat{Y}$
- Compare $\hat{Y}$ with the **true** Y
- Obtain COST and update the parameters $\theta$

![](_static/sentiment1.png)

### Sentiment Analysis
 
 
- Represent text as features
- Train Logistic Regression Classifier (for binary claassification: positive versus negative)
- Use the trained model to classify the new text
- If you data is not labeled for training, you can use pretrained models

![](_static/sentiment2.png) 
 

Poblems with a sparse representation:

$$
[\theta_1, \theta_2 ... \theta_n]

n = |V|,\; where\; V \;= \;vocabulary\; size
$$

1. Longer traning time
2. Longer prediction time
3. Number of parameters for LR classifier = n+1

Q1. How many parameters LR Classifier will have to learn for the following Twitter Corpus (assume you do not need to remove stopwords here)?
s
["I just bought a new airpod","I like this movie","I bought a movie ticket" ]

- 10, 11, 15, 16?

#### Vocabulary Mapping

["I am happy", "I am sad", "I am happy because I am learning NLP", "I am sad, I am not learning AI"]

Vocabulary V = [I, am, sad, happy, because, not, NLP, AI, learning]

| Positive(1)   |     Negative(0)      |
|:----------:|:-------------:|
| I am happy |  I am sad |
| I am happy because I am learning NLP |  I am sad, I am not learning AI |

**(Word, Class) Mapping**: each word has a dimension of 3

| Vocabulary   |     Positive(1)      | Negative(0) |
|:----------:|:-------------:|:-------------:|
| I | 3 | 3|
| am | 3  | 3|
| happy | 2 | 0 |
| because | 1  | 0|
| learning | 1 | 1|
| NLP | 1  | 0|
| sad | 0 | 2 | 
| AI | 0  | 1 |
| not | 0 | 1 |


#### Feature Extraction

$$
X_m = [1, \sum_w frequency(w,1), \sum_w frequency(w,0)]

where \; 1 = \; bias

\sum_w frequency(w,1) - \;all\; positive

\sum_w frequency(w,0) - \;all \;negative
$$

What are the values for <span style="color:red;">"I am sad, I am not learning AI"</span>:

[1, 7, 11]

1 corresponds to the bias, 7 - sum of positive frequencies, and 11 - sum of  negative frequencies


Q2. Calculate values for <span style="color:red;">"I am happy"</span>.

<small> Images source: DeepLearning.AI Natural Language Processing with Classification and Vector Spaces. </small>