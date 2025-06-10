# 🧠 Plagiarism Checker

This project is a **Plagiarism Detection System** that uses machine learning (Random Forest Classifier) to detect whether a given pair of sentences are plagiarized or not. It includes a simple **Streamlit-based UI** for user interaction.

## 📌 Problem Statement

Plagiarism is the act of copying someone else’s content without proper citation. In this project, we are given a dataset consisting of **pairs of sentences**, and the goal is to determine whether the second sentence is plagiarized from the first one.

## 🔍 Project Features

- Input: A pair of sentences (original + suspected).
- Preprocessing using TF-IDF vectorization.
- Machine Learning classification using **Random Forest**.
- Simple and interactive **Streamlit UI** for user input and output.
- Plagiarism Prediction (Yes/No).

## 🛠️ Tech Stack

- Python
- Scikit-learn (RandomForestClassifier, TF-IDF Vectorizer)
- Streamlit (UI)
- Pandas, NumPy
- Pickle (for model saving/loading)

## 📁 Dataset

The dataset contains:
- `sentence1`: The original sentence.
- `sentence2`: The suspected plagiarized sentence.
- `label`: 1 if plagiarized, 0 otherwise.

## 🧪 How It Works

1. **Preprocess**: Convert text to lowercase, remove punctuation, stopwords, etc.
2. **Feature Extraction**: Use **TF-IDF** vectorization on the sentence pairs.
3. **Model Training**: Train a **Random Forest Classifier** to predict plagiarism.
4. **UI Integration**: Use Streamlit to build a web interface for testing new sentence pairs.

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/plagiarism-checker.git
cd plagiarism-checker
