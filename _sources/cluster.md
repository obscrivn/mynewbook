# Document Similarity and Clustering

> Document similarity and clustering are two important concepts in information retrieval, the process of finding relevant and useful information from large collections of data

## Document Similarity
> Document similarity is the degree of relatedness or similarity between two or more documents based on their content, structure, or metadata

- Document similarity can be measured using cosine similarity, Jaccard similarity, or edit distance

## Document Clustering
> Document clustering is the process of grouping documents into clusters or categories based on their similarity

- Documents within a cluster are more similar to each other than to documents in other clusters
- Document clustering can be performed using k-means or hierarchical clustering

### The best text clustering algorithm

#### K-means
- **k-means** divides a dataset into k clusters, where k is a parameter that the user specifies
- The goal of k-means is to define spherical clusters, with each cluster having a centroid (or centre point)

The algorithm moves through two stages:

- Initialization: k initial centroids are randomly chosen
- Iteration: Each data point is assigned to the cluster with the nearest centroid

> K-means is sensitive to initial centroid placement and the presumption that all clusters are the same size and have a spherical shape

> It struggles with categorical data

#### Hierarchical Clustering
- **hierarchical clustering** creates a hierarchy of clusters, each of which is a subset of the cluster above it


- Agglomerative: This is a “bottom-up” approach
- Divisive: This is a “top-down” approach

> Hierarchical clustering has some advantages over k-means, such as the ability to handle categorical data