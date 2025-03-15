# **Cross-Domain Sentiment Analysis (CD-SA)**
This repository contains the **Cross-Domain Sentiment Analysis (CD-SA) project**, where we explore various sentiment classification methods across multiple datasets (IMDB, Amazon, Twitter). The study evaluates different models, feature representations, and preprocessing strategies to improve generalization in cross-domain sentiment classification.

---

## 📊 **Datasets**
This project utilizes three publicly available datasets from Kaggle:
- **IMDB**: Movie reviews dataset with dual sentiments, processed into **50,000 samples**.
- **Amazon Reviews**: Product review dataset, processed into **400,000 samples**.
- **Twitter (X) Data**: Originally **1.6M tweets**, with **400,000 samples randomly selected**.

- ## 📂 Data Resources & Usage
All dataset files are hosted on **Google Drive** due to GitHub file size limitations.  
📌 **Download all datasets here**: [Google Drive Link](https://drive.google.com/drive/folders/1yZyu8aV3b_M5zMXHt-PUO1snLfphTx9-?usp=drive_link)

### **1️⃣ Raw Data (Kaggle Source)**
**Location**: `Data Resource/`  
**Files**:
- `Amazon.txt` – Amazon product reviews dataset
- `IMDB.txt` – IMDB movie reviews dataset
- `test.ft.txt` – Additional test dataset  

These are raw datasets from Kaggle. Users can download and use them **directly for new research or model training**.

---

### **2️⃣ Processed CSV Data**
**Location**: `processed data.zip` (Google Drive)  
This ZIP file contains **preprocessed CSV files**, formatted for direct use in machine learning models.  
**Contents**:
- `amazon.csv` – Processed Amazon reviews
- `IMDB.csv` – Processed IMDB reviews
- `X_50000.csv`, `X_tokenized.csv`, `X.csv` – Processed Twitter data  

Users can **directly use these CSV files** to train sentiment classification models.

---

### **3️⃣ Tokenized & Vectorized Data**
**Location**: `The Raw Data.zip` (Google Drive)  
This ZIP file contains **fully tokenized and vectorized** data for sentiment analysis.  
**Contents**:
- TF-IDF vectorized text data
- Word2Vec embeddings
- FastText embeddings  

These files **allow for immediate model training without extra preprocessing**.

---

## 📌 **How to Use the Data**
### **Download & Extract**
1. **Download the required dataset from [Google Drive](https://drive.google.com/drive/folders/1yZyu8aV3b_M5zMXHt-PUO1snLfphTx9-?usp=drive_link)**.
2. **Extract the ZIP files**:
   ```bash
   unzip processed\ data.zip
   unzip The\ Raw\ Data.zip

---

## 🚀 **Experiment Process**
1. **Baseline Model:**
   - Used **SBERT (all-MiniLM-L6-v2) + Logistic Regression** for Amazon → IMDB sentiment transfer.
   - Achieved **77% accuracy**, serving as the baseline.

2. **IMDB Optimization:**
   - **TF-IDF + Logistic Regression**: Improved accuracy by **10%**.
   - **Spacy + Word2Vec (trained on dataset) + LSTM**: Also achieved **10% improvement**, with better efficiency.

3. **Twitter Data Optimization:**
   - Implemented **spelling correction, slang normalization, and emoji translation** in tokenization.
   - Combined **TF-IDF + FastText** for feature extraction.
   - Trained **XGBoost, LightGBM, Logistic Regression** but saw **only minor improvements in positive sentiment detection**.
   - **Fine-tuned Twitter-trained BERT (twitter-roberta-base-sentiment)**, achieving **77% accuracy**, outperforming all other methods.

---

## 📈 **Key Enhancements**
- **Baseline Boosting:** **SBERT (all-MiniLM-L6-v2) provided the strongest cross-domain sentiment transfer.**
- **Feature Engineering:** **Combining TF-IDF + FastText significantly improved classification accuracy.**
- **Hyperparameter Optimization:** **Optuna was used for automated tuning in XGBoost.**
- **Tokenizer Comparison:** **SpaCy was more efficient, while NLTK provided more customization options.**

---

## ❗ **Challenges & Limitations**
- **Excessive text preprocessing did not always improve results.** Over-refining text sometimes introduced noise rather than clarity.
- **Advanced methods underperformed on Twitter compared to simple approaches.** Despite detailed optimizations, **TF-IDF + Logistic Regression remained competitive**, likely due to **Twitter's highly informal language style.**
- **Computational Constraints:** Training **BERT-based models** was infeasible due to hardware limitations.
- **Complex spelling correction was time-consuming.** We limited typo correction to short phrases for efficiency.

---

## 🔍 **Key Insights**
- **Preserving essential textual information is more important than excessive cleaning.**  
- **CD-SA models must balance extracting key sentiment cues while minimizing domain-specific noise.**  
- **Domain-adapted transformer models (e.g., Twitter-RoBERTa) significantly outperform general NLP models.**  

---

## 📄 **Paper & Citation**
This research resulted in an **ACL research paper**, available in this repository as:  
📄 **[`NLP-CD-SA_Zhutong Zhang.pdf`](./NLP-CD-SA_Zhutong%20Zhang.pdf)**  

If you find this research useful, consider citing:
```bibtex
@article{Zhang2025CDSA,
  author    = {Zhutong Zhang},
  title     = {Cross-Domain Sentiment Analysis with Adaptive Text Preprocessing},
  journal   = {ACL 2025},
  year      = {2025}
}
