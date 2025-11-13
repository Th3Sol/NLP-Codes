<!-- <p align="center">
  <img src="https://capsule-render.vercel.app/api?type=venom&height=150&color=gradient&text=NLP%20Codes&section=header&reversal=false&textBg=false&animation=twinkling&fontColor=7cfc7b&fontSize=25">
</p> -->
# NLP Codes

## **EXP 1 – Brown & Penn Treebank Corpus**

```python
import nltk
from nltk.corpus import brown, treebank
nltk.download('brown')
nltk.download('treebank')

print("Brown Categories:", brown.categories())
print("Brown Sample:", brown.words(categories='news')[:20])
print("Penn Treebank Sample:", treebank.words()[:20])
```

---

## **EXP 2 – Sentence & Word Segmentation**

### SpaCy Sentence Segmentation

```python
import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp("I love coding. NLP is amazing.")

print("Sentences:")
for s in doc.sents:
    print(s)
```

### NLTK Word Tokenization

```python
from nltk.tokenize import word_tokenize, RegexpTokenizer

text = "Hi! Let's test segmentation."
print("NLTK Word Tokenize:", word_tokenize(text))
print("Regex Tokenize:", RegexpTokenizer(r'\s+', gaps=True).tokenize("I Love Python"))
```

---

## **EXP 3 – Tokenization Techniques**

```python
import nltk
from nltk.tokenize import TreebankWordTokenizer, wordpunct_tokenize, sent_tokenize, WhitespaceTokenizer
nltk.download('punkt')

text = "Hello World! Let's test tokenizers."
print("Treebank:", TreebankWordTokenizer().tokenize(text))
print("wordpunct:", wordpunct_tokenize(text))
print("Sentences:", sent_tokenize(text))
print("Whitespace:", WhitespaceTokenizer().tokenize(text))
```

---

## **EXP 4 – Lemmatization & Stemming**

```python
import nltk
from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer, WordNetLemmatizer
nltk.download('wordnet')

words = ["running", "flies", "wolves"]
ps = PorterStemmer()
ls = LancasterStemmer()
ss = SnowballStemmer("english")
lm = WordNetLemmatizer()

print("Word -> Porter | Lancaster | Snowball | Lemmatizer")
for w in words:
    print(w, "->", ps.stem(w), ls.stem(w), ss.stem(w), lm.lemmatize(w))
```

---

## **EXP 5 – Text Normalization & N-Grams**

```python
import nltk, re, contractions
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
nltk.download('punkt')

text = "I'm learning NLP!!! It's fun, isn't it?"
text = contractions.fix(text)
clean = re.sub(r'[^a-zA-Z\s]', '', text).lower()
tokens = word_tokenize(clean)

print("Tokens:", tokens)
print("Unigrams:", list(ngrams(tokens,1)))
print("Bigrams:", list(ngrams(tokens,2)))
print("Trigrams:", list(ngrams(tokens,3)))
```

---

## **EXP 6 – POS Tagging**

```python
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

text = "The quick brown fox jumps over the lazy dog."
print(nltk.pos_tag(word_tokenize(text)))
```

---

## **EXP 7 – Named Entity Recognition**

```python
import spacy
nlp = spacy.load("en_core_web_sm")

doc = nlp("Barack Obama was born in Hawaii.")
for ent in doc.ents:
    print(ent.text, "->", ent.label_)
```

---

## **EXP 8 – Dependency Parsing & Chunking**

### NLTK Chunking

```python
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.chunk import RegexpParser
nltk.download('punkt')

text = "The quick brown fox jumps over the lazy dog."
tokens = pos_tag(word_tokenize(text))

grammar = "NP: {<DT>?<JJ>*<NN>}"
cp = RegexpParser(grammar)
print(cp.parse(tokens))
```

### SpaCy Dependency Parsing

```python
import spacy
nlp = spacy.load("en_core_web_sm")

doc = nlp("The young developer solved the issue quickly.")
for t in doc:
    print(t.text, "->", t.dep_, "->", t.head.text)
```

---

## **EXP 9 – Word Embeddings**

### Word2Vec

```python
from gensim.models import Word2Vec
model = Word2Vec([["this","is","word2vec","test"]], vector_size=20, min_count=1)
print(model.wv["word2vec"][:10])
```

### BERT Embeddings

```python
from transformers import BertTokenizer, BertModel
import torch

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

text = "I love NLP."
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

cls_embedding = outputs.last_hidden_state[0][0]
print(cls_embedding[:10])
```

---

## **EXP 10 – Sentiment Analysis & Fake News Detection**

### TextBlob Sentiment

```python
from textblob import TextBlob

text = "I really love this NLP practical!"
blob = TextBlob(text)

print("Text:", text)
print("Sentiment Polarity:", blob.sentiment.polarity)
print("Sentiment Subjectivity:", blob.sentiment.subjectivity)
```

### Fake News Classification (TF-IDF + Logistic Regression)

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

texts = ["Fake news spreading!", "Government released report"]
labels = [1, 0]

X = TfidfVectorizer().fit_transform(texts)
clf = LogisticRegression().fit(X, labels)

print(clf.predict(X))
```

---

## **EXP 11 – Fine-Tuning HuggingFace Model**

```python
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

ds = load_dataset("imdb", split="train[:1%]").train_test_split(0.2)
tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def encode(e): 
    return tok(e["text"], truncation=True, padding="max_length")

ds = ds.map(encode)
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

trainer = Trainer(
    model=model,
    args=TrainingArguments("out", per_device_train_batch_size=4, num_train_epochs=1)
)

print("Model ready. Run trainer.train() to fine-tune.")
```

<p align="center">
  <a href="https://github.com/Th3-C0der/NLP-Codes/blob/main/NLP_Practicals_Th3_Complete.ipynb">
    <img src="https://img.shields.io/badge/📘 VIEW_NOTEBOOK-1976d2?style=for-the-badge&logo=jupyter&logoColor=white&labelColor=0d47a1&color=1976d2">
  </a>
</p>

<p align="center">
  <b>View the complete notebook with code + outputs (no need to run)</b>
</p>

<br>

<p align="center">
  <a href="https://colab.research.google.com/github/Th3-C0der/NLP-Codes/blob/main/NLP_Practicals_Th3_Complete.ipynb">
    <img src="https://img.shields.io/badge/🚀 OPEN_IN_COLAB-f9a825?style=for-the-badge&logo=googlecolab&logoColor=000&labelColor=f57f17&color=f9a825">
  </a>
</p>

<p align="center">
  <b>Run, test, and edit the notebook yourself in Google Colab</b>
</p>
