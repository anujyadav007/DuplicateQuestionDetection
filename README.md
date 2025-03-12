# Duplicate Question Detection Using NLP and Transformers

## Description
This project addresses the challenge of detecting duplicate questions on online platforms using advanced NLP techniques and machine learning models. By identifying similar questions, we enhance the user experience by reducing repetitive content and making it easier to find high-quality answers quickly.

The solution integrates two distinct approaches for classification:
- **Bag of Words (BoW) with Random Forest**
- **DistilBERT (Transformer-based model)**

Each approach offers unique insights into the semantic relationships between questions.

---

## Project Overview
The project classifies question pairs as duplicates or non-duplicates using a combination of traditional feature engineering methods and advanced deep learning techniques. The key steps include:
- Data preprocessing
- Feature engineering
- Model training with scikit-learn and Hugging Face's Transformers library
- Deployment for efficient evaluation

---

## Models
### Approach 1: Bag of Words (BoW) with Random Forest Classifier
#### **Vectorization:**
- BoW and TF-IDF are used to represent questions as numerical features.

#### **Feature Engineering:**
- Includes basic, token-based, length-based, and fuzzy features.

#### **Modeling:**
- Random Forest Classifier is trained on engineered features to classify question pairs as duplicates or non-duplicates.

#### **Performance:**
- Achieved **81.67% accuracy** in detecting duplicate questions.

---

### Approach 2: DistilBERT Transformer (Deep Learning Model)
#### **Tokenization:**
- The DistilBERT tokenizer splits questions into subword units, converting them into embeddings that capture contextual meaning.

#### **Self-Attention Mechanism:**
- DistilBERT weighs the relevance of each word in relation to others, allowing a deeper understanding of semantic relationships.

#### **Modeling:**
- Fine-tuned on the Quora Question Pairs dataset to classify duplicate questions.

#### **Performance:**
- Achieved **89.89% accuracy**, significantly outperforming the BoW approach.
- Effectively handles variations in phrasing, making it highly suitable for detecting semantic duplicates.

#### **Key Benefit:**
- Deep learning enables nuanced understanding of sentence structure and context, improving detection of semantically similar questions.

---

## Notebooks
### Approach 1:
- **Bag of Words Model:** [BoW Implementation](https://www.kaggle.com/code/gyanbardhan/bow-00)
- **TF-IDF Model:** [TF-IDF Implementation](https://github.com/Gyanbardhan/Duplicate-Question/blob/main/TF-IDF.ipynb)

### Approach 2:
- **DistilBERT Model:** [DistilBERT Implementation](https://huggingface.co/spaces/gyanbardhan123/Bert_DuplicateQuestionDetection/blob/main/Bert%20Duplicate%20Question%20Detection.ipynb)

---

## Dataset
We use the **Quora Question Pairs dataset**, which contains pairs of questions labeled as duplicates or non-duplicates.
- [Quora Question Pairs Dataset](https://www.kaggle.com/datasets/gyanbardhan/quora-duplicate-questions-copy)

---

## Text Preprocessing
Preprocessing ensures clean and structured text data for modeling:
- **Tokenization:** Splitting text into words/subwords.
- **Lowercasing:** Converting text to lowercase for uniformity.
- **Stop Words Removal:** Eliminating non-contributory words (e.g., "is", "the").
- **Stemming/Lemmatization:** Reducing words to their base form.
- **Special Character Removal:** Removing unwanted symbols.

---

## Feature Engineering
### **Basic Features:**
- `q1_len`, `q2_len`: Character lengths of the two questions.
- `q1_words`, `q2_words`: Word count in each question.
- `words_common`: Common words between both questions.
- `words_total`: Total number of words in both questions combined.
- `word_share`: Ratio of common words to total words.

### **Token Features:**
- `cwc_min`, `cwc_max`: Ratios of common words to smaller and larger question lengths.
- `csc_min`, `csc_max`: Ratios of common stop words to smaller and larger stop word counts.
- `first_word_eq`, `last_word_eq`: Binary features indicating if the first or last words are the same.

### **Length-Based Features:**
- `mean_len`: Average length of the two questions.
- `abs_len_diff`: Absolute word count difference.
- `longest_substr_ratio`: Ratio of the longest common substring to the smaller question length.

### **Fuzzy Features:**
- `fuzz_ratio`, `fuzz_partial_ratio`, `token_sort_ratio`, `token_set_ratio`: Various fuzzy similarity scores.

---

## Model Evaluation
Models are evaluated using accuracy and confusion matrices:
- **Random Forest with BoW:** **81.67% Accuracy**
- **DistilBERT:** **89.89% Accuracy** (Best Performing Model)

---

## Deployment
Both models are deployed on **Hugging Face Spaces**, optimized for efficient evaluation:
- **BoW with Random Forest:** [Hugging Face Deployment](https://huggingface.co/spaces/gyanbardhan123/Duplicate_Question_Detection)
- **DistilBERT:** [Hugging Face Deployment](https://huggingface.co/spaces/gyanbardhan123/Bert_DuplicateQuestionDetection)

---

## Key Takeaways
- **BoW with Random Forest:** A traditional, efficient solution with **81.67% accuracy**.
- **DistilBERT:** Leverages transformers for superior **89.89% accuracy**, effectively handling question context and variations.

---

## Join Us
Join us in improving duplicate question detection and enhancing user experience by reducing repetitive content. Together, we can make online platforms more efficient and informative.

---

## Keywords
*NLP, Duplicate Question Detection, Machine Learning, DistilBERT, Random Forest, BoW, TF-IDF, Feature Engineering, Hugging Face, Transformer Models, Quora Dataset*

---



---

## License
This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.
