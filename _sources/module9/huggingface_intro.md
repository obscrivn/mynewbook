# Optional: Exploring Pretrained Models with Hugging Face

**15–20 minutes · ungraded · CPU only**

After the [LSTM coding practice](week09_lstm_practice.ipynb), explore how a pretrained transformer classifies the same kinds of movie reviews. Use one ready-made model, inspect its tokenizer and token IDs, and investigate predictions on negation and contrast. This is a preview of Week 10's attention and transfer learning topics.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/obscrivn/mynewbook/blob/master/module9/week09_huggingface_intro.ipynb)

You can also [read the notebook with its verified outputs](week09_huggingface_intro.ipynb). Use a standard Colab CPU runtime and run cells in order. The first run downloads the public model; no account with Hugging Face, API key, GPU, or training is needed.

## What happens inside a pipeline?

A sentiment pipeline tokenizes text, passes the numerical inputs through a model, and turns the resulting scores into labeled probabilities. The notebook exposes those steps using the same checkpoint and tokenizer throughout. You will compare predictions with your own reading of the reviews and consider why confidence does not establish correctness.

For an optional video walkthrough beyond the activity's 15–20 minutes, watch Hugging Face's [What happens inside the pipeline function? (PyTorch)](https://www.youtube.com/watch?v=1pedAIvTWXk), also embedded in its official [Behind the pipeline tutorial](https://huggingface.co/learn/llm-course/en/chapter2/2).

<iframe width="560" height="315" style="max-width: 100%;" src="https://www.youtube-nocookie.com/embed/1pedAIvTWXk" title="Hugging Face: What happens inside the pipeline function? (PyTorch)" loading="lazy" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

If the embedded player is unavailable, use the video link above.

## Official references

- [Working with pipelines](https://huggingface.co/learn/llm-course/en/chapter1/3)
- [Behind the pipeline](https://huggingface.co/learn/llm-course/en/chapter2/2)
- [DistilBERT sentiment checkpoint and limitations](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english)
