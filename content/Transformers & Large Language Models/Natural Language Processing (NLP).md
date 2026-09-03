Manipulating Text

## Classification
![[Pasted image 20260901210701.png]]

- Sentiment extraction
- Intent detection
- Language detection
- Topic modeling
## 

Classification ― word (noun, verb, ...)
Multi-classification ―  "My dog had a walk inside the park"

### Evaluation Metics
- **Accuracy**: $\%$ of observations that were correctly predicted (could be heavily swayed by data collected in data set)
- **Precision**: $\%$ of predicted positive that were corrected
- **Recall** : $\%$ of actually positive that were corrected
- **F1-Score**: the harmonic mean of both **Precision** and **Recall** to give off $i_{1}$ number

BLEU - (Bilingual evaluation understudy) ― quality of text translated, similar to "precision"
ROUGE - () ― quality of text generated, similar to "recall"
Perplexity ― quantifies how 'surprised' the model is to see some words together

> [!note] these metrics always needs references


## Tokenization
How to cut sentences into the model

> [!note]
> the unit of text = token
> 
> So less things gemini gives back = less tokens spent
> The more token given = more time spent to process = more expensive

> [!example] Sentence: A cute teddy bear is reading.
> **Arbitrary**: cut by grammatical structure
> **word**: each word is a token (will end up with 2 different entities despite being the same word, eg. run vs. ran)
> **sub-word**: process the roots of the word separately with the extensions ( eg. 'ing' suffix)


| Method          | Pros                                                            | Cons                                                     |
| --------------- | --------------------------------------------------------------- | -------------------------------------------------------- |
| Word-Level      | Simple<br>Interpretable                                         | Risk of OOV<br>Does not leverage knowledge of root       |
| Subword-level   | Leverages common prefixes and suffixes<br>Learned from the data | Risk of OOV, though less than word-level\                |
| Character-level | Small change of OOV<br>Robust to casing and misspelling         | Makes computation slower<br>Embeddings not interpretable |

## Word Representation
### Naive Method: One-hot encoding
![[Pasted image 20260901214918.png]]

$$
\text{soft} = \begin{pmatrix}
1 \\
0 \\
0
\end{pmatrix}
$$

