# Hinglish Sentiment Analyser 🎯

A sentiment classifier for **Hindi-English mixed (Hinglish) text** — the actual language Indian users write reviews, tweets, and comments in. Standard English NLP models fail completely on this. This model doesn't.

**Live Demo:** [Add Streamlit Cloud link here after deployment]

---

## What it does

Paste any Hinglish text and get:
- Sentiment label — Positive / Negative / Neutral
- Confidence score
- Breakdown across all three classes

**Example:** Input:  "yaar ye phone ekdum bekar hai, total waste of money"
Output: Negative (94.2%)
---

## Why this problem is hard

Standard English NLP models fail on Hinglish because:
- Same word spelled multiple ways — "nahi", "nhi", "nahin"
- No standard dictionary for transliterated Hindi
- Sentiment carried by Hindi words missed entirely by English tokenizers
- Sarcasm expressed differently in mixed script

---

## Model Architecture
Input (Hinglish tweet)
↓
TextVectorization (20,000 token vocab)
↓
Embedding Layer (128 dimensions)
↓
Conv1D (64 filters, kernel=3) — captures local patterns like "ekdum bekar", "bahut acha"
↓
MaxPooling1D
↓
Bidirectional LSTM (64 units) — reads sentence forward + backward
↓
Dropout (0.5) + L2 Regularization
↓
Dense (32) → Dense (3, softmax)
↓
Output: Negative / Positive / Neutral

**Total parameters:** 2.6M  
**Model size:** 10MB

---

## Results

| Metric | Score |
|--------|-------|
| Macro F1 | 92% |
| Validation Accuracy | 91.99% |
| Negative F1 | 94% |
| Positive F1 | 91% |
| Neutral F1 | 92% |

Trained on 14,565 tweets, validated on 2,998 tweets from SemEval-2020 Task 9 (SentiMix).

---

## Dataset

**SemEval-2020 Task 9 — Sentiment Analysis for Code-Mixed Social Media Text**
- 17,500+ labelled Hinglish tweets
- Labels: Positive / Negative / Neutral
- Source: Real Twitter data, Hindi-English code-mixed

---

## Project Structure
hinglish-sentiment/
├── data/
│   └── processed/        ← cleaned train/val CSVs
├── notebooks/
│   ├── 01_eda.ipynb          ← data exploration
│   ├── 02_tokenizer.ipynb    ← TextVectorization setup
│   └── 03_model.ipynb        ← model training + evaluation
├── app/
│   └── streamlit_app.py      ← live demo
├── models/                   ← saved .keras model
├── FAILURES.md               ← what didn't work and why
├── requirements.txt
└── README.md

---

## Run Locally

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/hinglish-sentiment-analyser.git
cd hinglish-sentiment-analyser

# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Run the app
venv\Scripts\streamlit run app\streamlit_app.py
```

---

## What I Learned / Challenges

See [FAILURES.md](./FAILURES.md) for full documentation of what didn't work and why.

Key findings:
- Pretrained English embeddings fail on Hindi words — trained from scratch instead
- Dataset contains noisy political labels — documented limitation
- Conv1D destroys mask information from TextVectorization — known Keras tradeoff
- EarlyStopping patience must match learning rate — too aggressive with low LR

---

## Tech Stack

- **TensorFlow 2.21** — model training
- **Keras** — TextVectorization, Embedding, Conv1D, Bi-LSTM
- **Pandas / NumPy** — data processing
- **Streamlit** — deployment
- **Scikit-learn** — evaluation metrics

---

## Author

Built as a portfolio project demonstrating NLP skills from the  
**TensorFlow Developer Professional Certificate — DeepLearning.AI (Coursera)**