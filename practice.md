# Practice with K-Means Clustering

Common steps for Document clustering:

- Loading or preparing the dataset [dataset link: https://github.com/PawanKrGunjan/Natural-Language-Processing/blob/main/Sarcasm%20Detection/sarcasm.json]
- Vectorizing the text using TfidfVectorizer
- Reducing the dimension using PCA
- Clustering the documents
- Ploting the cluster using matplotlib

We willl be loading data in `json` format and using `KMeans` from sklearn

```python
import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
```
We have a data from the github link: https://github.com/PawanKrGunjan/Natural-Language-Processing/blob/main/Sarcasm%20Detection/sarcasm.json

If you open the github, ther eis an option Raw text which gives us the following link: https://raw.githubusercontent.com/PawanKrGunjan/Natural-Language-Processing/main/Sarcasm%20Detection/sarcasm.json

```python
Dataset link: 
df=pd.read_json('https://raw.githubusercontent.com/PawanKrGunjan/Natural-Language-Processing/main/Sarcasm%20Detection/sarcasm.json')
```

- Extracting Features from headlines only
```python
sentence = df.headline
```

- creating vectorizer
```python
vectorizer = TfidfVectorizer(stop_words='english')
```
  
- vectorizering the text documents
```python
vectorized_documents = vectorizer.fit_transform(sentence)
```

- We are going to reduce dimensions from a typically sparse NLP data

   - Initializing a PCA (Principal Component Analysis) object
   - Specifying to reduce the dimensionality to 2 principal components
   
- PCA is applied to the vectorized_documents
- PCA will transform the data into a lower-dimensional representation

```python
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(vectorized_documents.toarray())
```
- We set the number of clusters (groups) to 2, meaning the K-Means will attempt to partition the data into two distinct clusters
```python
num_clusters = 2
kmeans = KMeans(n_clusters=num_clusters, n_init=5,
                max_iter=500, random_state=42)
kmeans.fit(reduced_data)
```

> `K-Means()`` is initialized with the specified number of clusters and some other parameters:
    - n_clusters specifies the number of clusters
    - n_init is the number of times the algorithm will be run with different centroid
    - max_iter is the maximum number of iterations the K-Means algorithm can run for each initialization
    - random_state is set to ensure reproducibility

- We store results
    - The original documents are stored in the column `document`
    - The cluster labels assigned by the K-Means algorithm are stored in a column called "cluster" 
    - Each document is assigned to one of the clusters
```python
results = pd.DataFrame()
results['document'] = sentence
results['cluster'] = kmeans.labels_
print(results.sample(5))
```
<center>
<img src="_static/sample_cluster.png" alt="sample" width = 350 />
</center>

- We will visualze the results
```python
colors = ['red', 'green']
cluster = ['Not Sarcastic','Sarcastic']
for i in range(num_clusters):
    plt.scatter(reduced_data[kmeans.labels_ == i, 0],
                reduced_data[kmeans.labels_ == i, 1], 
                s=10, color=colors[i], 
                label=f' {cluster[i]}')
plt.legend()
plt.show()
```
<center>
<img src="_static/cluster.png" alt="cluster" width = 350 />
</center>

