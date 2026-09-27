# Week 06 Coding Practice Dataset

File: `week6_topics_dataset.csv` (52 documents)

## Source

Curated from the **20 Newsgroups** dataset (Usenet posts, early 1990s), loaded via
scikit-learn's `sklearn.datasets.fetch_20newsgroups` loader. The 20 Newsgroups
collection was originally assembled by Ken Lang and has been freely redistributed
for research and teaching for decades (it ships directly inside scikit-learn, NLTK,
and many other open ML tooling packages, with no known copyright restriction on
this kind of redistribution). See http://qwone.com/~jason/20Newsgroups/ for the
canonical dataset description.

## Curation method

- Four source newsgroups were selected for four recognizable, mostly-separable
  themes: `rec.sport.hockey` (Sports), `sci.space` (Space), `comp.graphics`
  (Computing), `sci.med` (Health).
- Headers, quoted-reply lines, and signature blocks were stripped from each raw
  post.
- Each cleaned post was trimmed to its first 2-4 sentences (roughly 180-500
  characters) to keep documents short and readable for a formative activity.
- Posts that were non-English, mostly numeric/code, or still contained leftover
  quote artifacts after cleaning were discarded.
- Each remaining post was required to contain at least two words from a small
  theme-specific keyword set (e.g. "shuttle", "orbit", "mission" for Space;
  "graphics", "image", "format" for Computing). This keeps each theme's 13
  documents lexically cohesive enough that NMF and LDA can discover a
  reasonably clean topic per theme, while still leaving genuine overlap cases
  (for example a Computing post about processing satellite elevation-data
  images) for students to find and discuss.
- Within each theme, documents with more keyword hits were prioritized, then
  13 were sampled per theme (random seed fixed for reproducibility), for 52
  documents total.

## Columns

- `doc_id`: integer identifier, 1-52.
- `source_newsgroup`: the original 20 Newsgroups category (e.g. `sci.med`).
- `theme_area`: a short human-readable theme label (e.g. `Health`).
- `text`: the cleaned, trimmed document text.

## Intended use

`source_newsgroup` and `theme_area` document where the data came from and are
useful for verifying the notebook and writing instructor answer keys. The Coding
Practice is an **unsupervised** topic-discovery activity: students should build
and interpret topic models using only the `text` column, not these labels. The
notebook says this explicitly.

## Privacy and sensitivity

Usenet posts occasionally reference the poster or people they know informally
(e.g. "my girlfriend", "my friend"), but no full names of private individuals,
contact details, or other personally identifying information appear in the
selected snippets. No sensitive categories of data (financial, legal, etc.) are
present.

## Regenerating or extending this dataset

The curation script used to build this file is not checked into the repository.
To reproduce it: fetch the four categories above via `fetch_20newsgroups` with
`remove=("headers", "footers", "quotes")`, strip residual quote-prefix lines,
keep the first 2-4 sentences of each post within the character bounds above,
filter for English/clean text, require at least two theme-keyword hits per
post, and sample 13 documents per category with a fixed random seed.
