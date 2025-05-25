 # 🧠 NLP Preprocessing for Embeddings

This guide explains how to properly preprocess text data when using different types of word embeddings in Natural Language Processing (NLP), including **GloVe**, **Word2Vec**, and **BERT**.

---

## 📌 Table of Contents

1. [Static Embeddings (GloVe, Word2Vec)](#1-static-embeddings-glove-word2vec)
2. [Contextual Embeddings (BERT, RoBERTa)](#2-contextual-embeddings-bert-roberta)
3. [Summary Table](#3-summary-table)
4. [Python Examples](#4-python-examples)
5. [Tips](#5-tips)

---

## 1. Static Embeddings (GloVe, Word2Vec)

Static embeddings map each word to a fixed vector, regardless of context.

### ✅ Preprocessing Steps:

- ✅ Convert text to **lowercase**
- ✅ **Remove punctuation** and non-alphabetic characters (optional)
- ✅ **Tokenize** the text
- ✅ **Remove stopwords** (optional)
- ✅ **Lemmatize or stem** words (optional)
- ✅ **Map tokens to vectors** using a pre-trained embedding

> 📦 Use with models like: LSTM, GRU, traditional ML

---

## 2. Contextual Embeddings (BERT, RoBERTa)

These embeddings generate **context-aware** representations for each word/token.

### 🚫 Do NOT:
- ❌ Lowercase manually (use correct tokenizer/model)
- ❌ Remove punctuation or stopwords
- ❌ Tokenize with external tools

### ✅ Instead:
- ✅ Use the **tokenizer provided by the model**
- ✅ Ensure correct casing (`bert-base-cased` vs `uncased`)
- ✅ Include special tokens (`[CLS]`, `[SEP]`)
- ✅ Return attention masks and token types if needed

> 📦 Use with transformer-based models like: BERT, RoBERTa, DistilBERT

---

## 3. Summary Table

| Step                     | Static Embeddings (GloVe/Word2Vec) | Contextual Embeddings (BERT) |
|--------------------------|------------------------------------|------------------------------|
| Lowercasing              | ✅ Yes                              | ✅ if `uncased`, ❌ if `cased` |
| Remove punctuation       | ✅ Often                            | ❌ No                         |
| Remove stopwords         | ✅ Optional                         | ❌ No                         |
| Tokenization             | ✅ Manual (e.g., `nltk`)            | ✅ Use model tokenizer        |
| Lemmatization/Stemming   | ✅ Optional                         | ❌ No                         |
| Embedding lookup         | ✅ One vector per word              | ✅ Contextualized per token   |

---

## 4. Python Examples

### 📌 GloVe Preprocessing Example:

```python
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

nltk.download('punkt')
nltk.download('stopwords')

text = "The quick brown fox jumps over the lazy dog."
text = text.lower()
tokens = word_tokenize(text)
tokens = [t for t in tokens if t not in string.punctuation]
tokens = [t for t in tokens if t not in stopwords.words('english')]
print(tokens)

