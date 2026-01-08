# _AutoJudge: Predicting Programming Problem Difficulty_

# Project Overview

Online coding platforms such as Codeforces, CodeChef, and Kattis categorize programming problems into difficulty levels (Easy / Medium / Hard) and assign numerical difficulty scores. These classifications are often subjective and based on human judgment and user feedback.

AutoJudge is an end-to-end Machine Learning system that automatically predicts:

_Difficulty Class: Easy / Medium / Hard (Classification)_

_Difficulty Score: A numerical difficulty value (Regression)_

    The predictions are made using only the textual information of a programming problem, including its description, input format, and output format.
The project also includes a local web interface that allows users to paste a new problem description and instantly receive predictions.

# Dataset Used

A dataset of 4,112 programming problems was used. Each problem includes:

1.title

2.description

3.input_description

4.output_description

5.problem_class (Easy / Medium / Hard)

6.problem_score (numerical difficulty score)

The dataset was sourced via web scraping from competitive programming platforms and was provided with pre-labeled difficulty classes and scores. No manual labeling was performed.

# Approach and Methodology
_1️. Data Preprocessing_

✔ Combined all textual fields (description, input_description, output_description) into a single text corpus

✔ Removed HTML tags, punctuation, and special characters

✔ Converted text to lowercase

✔ Normalized whitespace

✔ Handled missing values

✔ Created additional handcrafted features such as: Text length, Number of mathematical symbols and Frequency of algorithm-related keywords (dp, graph, tree, recursion, binary search)

_2. Feature Extraction_

✔ TF-IDF Vectorization

    n-grams: (1, 3)

    Maximum features: 30000
    
    Sublinear term frequency scaling

✔ Handcrafted Numeric Features: To capture problem complexity beyond keywords:

Log-scaled text length

Mathematical symbol count

Constraint awareness (e.g., large input sizes, time limits)

Algorithm-specific keyword groups: Dynamic Programming, Graph Algorithms, Data Structures, Mathematics, Geometry, String Algorithms, Greedy Techniques

    ✔ Numeric features were scaled using StandardScaler and combined with TF-IDF features.

_3️. Classification Models_

The following classification models were evaluated:

✔ Logistic Regression (balanced class weights)

✔ Linear Support Vector Machine (SVM)

✔ Multinomial Naive Bayes (TF-IDF only, baseline)

    Hyperparameter tuning was performed using GridSearchCV with stratified cross-validation

✅ _Final choice: Tuned Linear SVM, due to the best balance between precision, recall, and macro-F1 score._

Performance (3-Class Classification):

✅ Accuracy: ~54%

✅ Macro F1-score: ~0.50

✅ Hard class recall: ~0.73

    The results highlight the inherent ambiguity between Easy and Medium problems while demonstrating strong detection of Hard problems



_4️. Regression Model (Difficulty Score)_

Models Evaluated

✔ Linear Regression (baseline)

✔ Gradient Boosting Regressor

✔ Random Forest Regressor

    RandomizedSearchCV was used for hyperparameter tuning of ensemble models

✅Final Regression Model: Random Forest Regressor

Performance:

✅MAE: 1.635

✅RMSE: 1.948

    The model predicts difficulty scores within ±2 points on average, which is reasonable for text-only inference.

    Chosen for robustness and ability to model non-linear relationships

_The regression model is independent of the classification model and provides a smoother difficulty estimate._

    _Note: Difficulty classification is inherently subjective, and significant overlap exists between Easy and Medium problems. Results reflect the realistic performance ceiling for text-only classification._


# 🌐 Web Interface

A Streamlit-based web application is provided that allows users to demonstrate end-to-end functionality.

Features:

✅ Text input boxes for:

Problem Description

Input Description

Output Description

✅ Outputs:

Predicted Difficulty Class

Predicted Difficulty Score

    The app loads pre-trained models only—no retraining occurs at runtime.


# ▶️ Steps to Run the Project Locally

1️⃣ Clone the Repository

    git clone <your-github-repo-link>
    
    cd AutoJudge

2️⃣ Install Dependencies

    pip install -r requirements.txt

3️⃣ Run the Web Application
         
    streamlit run app.py

4️⃣ Open Browser

Open the URL shown in the terminal:

    http://localhost:8501

# Saved Trained Models

The repository includes all pre-trained models: (Zip file included)

1. final_classifier.pkl – Tuned Linear SVM Classifier

2. final_regressor.pkl – Random Forest regression model 

3. tfidf.pkl – TF-IDF vectorizer

4. scaler.pkl – Feature scaler

These models are loaded directly by the web app.

# Demo Video

Link: https://drive.google.com/file/d/1YgTEQVDoveiFoCXT10MdJfVfrqNhgVK2/view?usp=sharing


# Author Details

Name: Riddhi Sidana

Enrollment No. 23322023

Program: BS-MS Economics




