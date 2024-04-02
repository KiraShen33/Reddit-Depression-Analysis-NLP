# Overview

This Python script is designed to analyze Reddit posts for symptoms of depression. It involves data loading, preprocessing, tokenizing, and applying natural language processing (NLP) techniques, such as Latent Dirichlet Allocation (LDA) and RoBERTa embeddings, for feature extraction. The extracted features are then used to train a Random Forest Classifier for symptom detection.

# Key Libraries Used

- Numpy
- Pandas
- NLTK
- PyTorch
- Gensim
- scikit-learn
- spaCy
- Transformers

# Features

1. Data Loading and Preprocessing: The script includes functions to load data from a specified file path and preprocess it for analysis.

2. Dataset Generation: It generates two datasets: one for symptom-related posts and another for control posts, based on specified subreddits.

3. Tokenization: Utilizes the happiestfuntokenizing library for tokenizing the text data.

4. Feature Extraction:
    - LDA Analysis: Performs topic modeling using Latent Dirichlet Allocation.
    - RoBERTa Embeddings: Extracts embeddings from the RoBERTa model for the textual data.

5. Model Training and Evaluation:
    - Trains a Random Forest Classifier.
    - Uses cross-validation for model evaluation.
    - Computes and displays AUC scores for each symptom.

# Usage

1. Mount Google Drive (if using Google Colab):
'''python
from google.colab import drive
drive.mount('/content/drive')
'''

2. Load Data:
    - Specify the file path and use the load function to load your dataset.

3. Preprocess and Generate Datasets:
    - Use the dataset_generation function to create symptom and control datasets.

4. Tokenization and Feature Extraction:
    - Tokenize the data using the tokenize function.
    - Extract features using LDA and RoBERTa models.

5. Train and Evaluate the Model:
    - Train the Random Forest Classifier using symptom features.
    - Evaluate the model using cross-validation.

# Requirements

- Ensure you have the above-mentioned libraries installed.
- Python 3.x
- For spaCy, download the en_core_web_sm model.
- NLTK punkt package: nltk.download('punkt')
- For RoBERTa, ensure you have an appropriate environment (like Google Colab) with GPU support.

# Note

- This script assumes the data is in a specific format, particularly for the Reddit posts.
- Modify the subreddit list and symptom mapping as per your dataset.
- Adjust the parameters of the LDA and RoBERTa models according to your computational resources and data size.

# Video
- Link: https://drive.google.com/file/d/1HFymbGC6tT3Q8fSMdr96lVLSOjDJpncv/view?usp=drive_link

# Conclusion

This script is a comprehensive tool for analyzing text data, specifically Reddit posts, for symptoms of depression using advanced NLP techniques. It is versatile and can be adapted to various datasets and requirements.# final-project-KiraShen33
