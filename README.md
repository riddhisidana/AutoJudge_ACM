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

✔ Handled missing values

✔ Created additional handcrafted features such as: Text length, Number of mathematical symbols and Frequency of algorithm-related keywords (dp, graph, tree, recursion, binary search)

_2. Feature Extraction_

✔ TF-IDF Vectorization

    n-grams: (1, 3)

    Maximum features: 8000

✔ Numeric features were scaled using StandardScaler

✔ Final feature matrix created by concatenating TF-IDF vectors with numeric features

_3️. Classification Models_

The following classification models were evaluated:

✔ Logistic Regression (balanced class weights)

✔ Linear Support Vector Machine (SVM)

✔ Multinomial Naive Bayes (TF-IDF only, baseline)

_Final choice: Linear SVM, due to the best balance between precision, recall, and macro-F1 score._

✔ A hierarchical classification strategy was also explored:

_Stage 1: Hard vs Not-Hard_

_Stage 2: Easy vs Medium_-

This analysis highlighted strong signals for detecting Hard problems while exposing overlap between Easy and Medium classes.

_4️. Regression Model_

✔ Random Forest Regressor

    Used to predict a continuous difficulty score

    Chosen for robustness and ability to model non-linear relationships

_The regression model is independent of the classification model and provides a smoother difficulty estimate._

# 📈 Evaluation Metrics
_Classification Performance (Linear SVM)_

✔ Accuracy: ~50%

✔ Macro F1-score: ~0.49

✔ Hard class recall: ~0.61

_Note: Difficulty classification is inherently subjective, and significant overlap exists between Easy and Medium problems. Results reflect the realistic performance ceiling for text-only classification._

_Regression Performance (Random Forest)_

✔ Mean Absolute Error (MAE): ~1.541

✔ Root Mean Squared Error (RMSE): ~1.94

Residual analysis shows no systematic bias and stable predictions across difficulty classes.

# 🌐 Web Interface

A Streamlit-based web application is provided that allows users to:

Paste:

Problem Description

Input Description

Output Description

Click Predict Difficulty

View:

✔ Predicted Difficulty Class (Easy / Medium / Hard)

✔ Predicted Difficulty Score (numerical)

The application runs locally and loads pre-trained models without retraining.



