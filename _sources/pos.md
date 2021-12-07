## POS Tagging

<object data="_static/pos.pdf" width="950" height="650" type='application/pdf'/></object>

### POS Pipeline

![](_static/pos.png)

### PennTreeBank Tagset
- Read Section 4 [TagGuide](https://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/data/Penn-Treebank-Tagset.pdf)

Question 1: Find two ambiguous cases (with different grammatical categories)

### NP Chunks

Question 2: Create rules for NP-phrases in English and an example. Use Penn-Treebank Tagset (see [taglist](https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html))

```
grammar = """
NP: { <NNP>}
"""
```
example: **Yoda** (a proper name)

