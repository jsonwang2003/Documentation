> [!INFO]
> Measures how well a language model predicts a sample; lower values indicate better predictions.

## How It Works

Perplexity quantifies how **"surprised" a language model is by a sequence of tokens**.  
For a sequence $W = w_1, w_2, ..., w_N$:

$$
\text{Perplexity}(W) = \exp\left( -\frac{1}{N} \sum_{i=1}^{N} \log p(w_i \mid w_1, ..., w_{i-1}) \right)
$$

- $p(w_i \mid w_1, ..., w_{i-1})$: Probability of token $w_i$ given its preceding context  
- $N$: Number of tokens in the sequence  

## What to Look For

- **Lower perplexity = better fluency and grammatical structure**  
- Sensitive to **vocabulary** and **tokenization**  
- Depends on the evaluation model used (e.g., GPT, BERT, n-gram)

## Application Models

- [[Transformer]]
- [[Recurrent Neural Network (RNN)]]
- [[Long Short-Term Memory (LSTM)]]