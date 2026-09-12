# 4.1 Semantic Similarity and Retrieval

Week 03 showed how text can be represented as vectors. This week asks the next question:

Once a document, sentence, or query is represented as a vector, how do we decide which items are similar and which ones should be retrieved?

The conceptual progression is simple:

text representation → similarity → ranking → retrieval → downstream NLP / AI applications

This is the foundation for semantic search, recommendation, duplicate detection, clustering, and the retrieval step in many modern AI systems.

:::{note}
This chapter builds directly on Week 03. It assumes students already understand bag-of-words, TF-IDF, sparse vectors, and dense embeddings at a conceptual level.
:::

## 1. What does “similar” mean?

Two pieces of text can be similar in different ways.

- Lexical similarity is about shared surface form: same words, same token overlap, or similar vocabulary.
- Semantic similarity is about shared meaning or intent, even when the wording differs.

For example, the sentences below are not identical in wording, but they are close in meaning:

- “The patient reported chest pain after exercise.”
- “After physical activity, the patient experienced discomfort in the chest.”

A bag-of-words or TF-IDF representation may score them low on lexical overlap, but a semantic embedding model may place them nearby in vector space.

This is why text similarity depends on representation choice, not only on the text itself.

## 2. Lexical similarity

A simple baseline for text similarity is lexical overlap. Jaccard similarity is a standard choice when we compare sets of tokens.

For two sets A and B:

$$
J(A, B) = \frac{|A \cap B|}{|A \cup B|}
$$

Interpretation:

- 1 means the sets are identical;
- 0 means they share no tokens;
- intermediate values mean partial overlap.

Jaccard is useful because it focuses on shared vocabulary. It tells us how much two documents overlap at the token-set level. It does not tell us whether they mean the same thing, and it is insensitive to repeated words if we compare sets rather than counts.

This is a good lexical baseline, but it is limited. Two texts may share few words and still be semantically related, or they may share many words and still be about different topics.

:::{important}
Lexical similarity is a strong baseline for surface overlap, but it does not capture meaning well. The representation matters.
:::

## 3. Distance and similarity in vector space

Once text has been converted into vectors, we can compare them numerically.

A distance metric tells us how far apart two vectors are. A similarity metric tells us how closely aligned they are.

### Euclidean and Manhattan distance

Euclidean distance is the straight-line distance between two vectors. Manhattan distance is the sum of absolute coordinate differences.

These are intuitive geometric measures, but they are not always the best choice for text.

Why? Because text vectors often have many dimensions and different lengths. A document with more words may naturally be farther away in raw Euclidean space even when the meaning is similar.

### Cosine similarity

Cosine similarity is the main metric for comparing text vectors. It measures the angle between two vectors rather than their absolute magnitude.

For vectors $x$ and $y$:

$$
\text{cosine}(x, y) = \frac{x \cdot y}{\|x\|\|y\|}
$$

This matters because text vectors may differ in length, but the important question is often whether they point in roughly the same direction.

```text
          vector A
         /|
        / |
       /  | angle
      /   |
vector B ----
```

A smaller angle means higher cosine similarity. A score near 1 means the vectors are highly aligned; a score near 0 means they are not aligned.

This is why cosine similarity is usually more useful than raw distance for text retrieval and semantic comparison. It emphasizes direction rather than absolute length.

:::{note}
For text, cosine similarity is usually more informative than Euclidean distance because the vector magnitude is often less important than the pattern of terms or meaning.
:::

## 4. Why cosine similarity matters

Cosine similarity is especially useful when vectors represent text features.

Consider the difference between:

- a sparse TF-IDF vector,
- a dense embedding vector,
- and a raw frequency vector.

A sparse lexical vector may treat each vocabulary item as a separate dimension. A dense embedding vector compresses these patterns into a latent space where semantically similar items can land near one another.

In practice, cosine similarity is the working default for many retrieval systems because it is simple, efficient, and interpretable.

The main lesson is not that cosine is magically “the truth.” The main lesson is that the similarity metric should match the representation and the task.

## 5. Sparse vs. dense similarity

Week 03 introduced sparse lexical vectors and dense embeddings. Week 04 asks how similarity behaves differently across those representations.

### Sparse TF-IDF similarity

TF-IDF vectors are sparse. They encode relative importance of words within a document and across a corpus. That makes them good for lexical matching and term-weighted comparison.

If two documents share the same jargon or important terms, TF-IDF cosine similarity can be high. If they talk about similar ideas without sharing many words, the score may be low.

### Dense embedding similarity

Dense embeddings represent words, sentences, or documents in a lower-dimensional space where nearby vectors often reflect semantic relatedness. This can capture paraphrases, synonyms, and related concepts that do not share the same words.

For example, two sentences may mean nearly the same thing while using very different vocabulary:

- “The meeting was postponed until Friday.”
- “We moved the discussion to next Friday.”

A lexical similarity measure may give them a low score; a semantic embedding model may give them a much higher score.

This difference is the heart of the Week 03 → Week 04 progression:

word/vector relationships → sentence/document embeddings → similarity → retrieval

## 6. Semantic textual similarity

Semantic textual similarity (STS) asks whether two texts mean similar things, even when the wording differs. This is different from lexical overlap, and it is one reason embeddings matter so much in modern NLP.

A semantic similarity system generally works like this:

1. represent each text as a vector,
2. compare the vectors,
3. interpret the score as a measure of semantic relatedness.

STS is useful in a range of applications:

- paraphrase detection,
- duplicate detection,
- near-duplicate filtering,
- semantic search,
- recommendation.

A high score does not guarantee the texts are equivalent in a strict logical sense. It only shows that the model’s representation finds them similar under a chosen similarity function.

## 7. Semantic search and retrieval

Semantic search is the modern, AI-era version of document retrieval.

In a standard retrieval workflow:

```text
query → representation → similarity scores → ranked documents
```

The process is:

1. The query is converted into a vector using the same representation used for the documents.
2. The document collection is also represented as vectors.
3. The system computes a similarity score between the query and each document.
4. The documents are ranked from most similar to least similar.
5. The top-ranked results are returned to the user or to a downstream system.

This is the conceptual engine behind search, recommendation, and retrieval-based AI systems.

A good retrieval system does not merely look for identical words. It tries to identify the documents whose vectors are closest to the query in the relevant representation space.

:::{important}
Retrieval is not “understanding.” It is a ranking process based on representations and a similarity function. The top result is only as good as the representation, the similarity choice, and the task.
:::

## 8. Retrieval in modern AI systems

Similarity and retrieval sit behind many modern AI workflows.

A simplified example:

```text
query → retrieval → selected context → generative model
```

This pattern is commonly associated with retrieval-augmented generation, or RAG. In a RAG system, the model does not rely only on its internal memory. It first retrieves relevant context from a document set, then uses that context to answer a question or complete a task.

The reading goal is not to teach full RAG architecture. It is to show that retrieval depends on the same conceptual ideas we have been discussing:

- represent the text,
- compare vectors,
- rank likely matches,
- then decide whether the result is relevant and trustworthy.

## 9. Clustering as a downstream application

Clustering is one application of similarity, not the central organizing topic of the week.

Once text is embedded or vectorized, documents can be grouped by proximity in a vector space. A simple example is k-means clustering, which assigns each item to a cluster whose centroid is closest.

This is useful for:

- topic grouping,
- document organization,
- duplicate detection,
- exploratory analysis.

But clustering should be treated as a downstream use of similarity. The key conceptual lesson is that once documents live in a vector space, similarity and proximity become operational tools.

Detailed clustering algorithms or long treatments of hierarchical clustering are not required for this week’s conceptual reading.

## 10. When similarity goes wrong

This is one of the most important lessons in the modern AI era.

A high similarity score is not the same as:

- truth,
- factual equivalence,
- correctness,
- task relevance,
- or usefulness.

Two documents can be highly similar in vector space and still be wrong for the user’s actual question.

Examples include:

- paraphrases that are semantically close but not equally trustworthy,
- documents that share keywords but not the same intent,
- retrieval results that are similar to the query but irrelevant to the user’s task,
- dense embeddings that prefer semantic relatedness over factual accuracy.

A retrieval system may rank a plausible answer above a better one, especially when the representation is imperfect or the user query is underspecified.

This is why similarity models require human judgment, validation, and task-aware interpretation.

:::{warning}
Similarity is a relationship between representations. It is not proof that the result is true, correct, or useful.
:::

## 11. Key takeaways

- Similarity in language is not one thing. It can be lexical, geometric, or semantic.
- Jaccard is a useful lexical baseline, but it does not capture meaning.
- Euclidean and Manhattan distance provide geometric intuition, but cosine similarity is usually the more important metric for text vectors.
- Sparse TF-IDF vectors and dense embedding vectors behave differently because they represent different kinds of information.
- Semantic search works by converting queries and documents to vectors, calculating similarity, and ranking matches.
- Retrieval systems are powerful but imperfect. Similarity is not the same as truth or usefulness.

The central idea of Week 04 is therefore straightforward:

Once language has been represented as vectors, similarity gives us a way to compare, rank, retrieve, and reason about texts in a computationally useful way.

## References

{cite}`jurafsky2026`

{cite}`mikolov2013efficient`

{cite}`manning2009introduction`