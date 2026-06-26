> [!INFO]
> Evaluates the quality of machine translation by comparing output with reference translations.

## How It Works

BLEU measures the **precision of n-grams in the candidate translation that appear in the reference translation**.  
It includes a brevity penalty to discourage overly short outputs.

$$
\text{BLEU} = BP \cdot \exp\left( \sum_{n=1}^{N} w_n \log p_n \right)
$$

- $p_n$: Precision of n-grams of size $n$  
- $w_n$: Weight for each n-gram level (typically uniform)  
- $BP$: Brevity penalty to penalize short outputs  

## What to Look For

- **Higher BLEU = better alignment with reference**  
- Sensitive to **exact word matches**, **not semantics**  
- Best for *machine translation*, but also used in *summarization* and *captioning*

## Application Models

- [[Transformer]]
- [[Recurrent Neural Network (RNN)]]