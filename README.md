# Fake News Detection Project

This project implements a machine learning solution to detect whether a given news article is "FAKE" or "REAL". Several models are trained and used to predict the authenticity of news articles using natural language processing techniques like TF-IDF for feature extraction.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [File Structure](#file-structure)
- [Usage](#usage)
- [Preprocessing Steps](#preprocessing-steps)
- [Model Details](#model-details)
- [Prediction](#prediction)
- [License](#license)

## Introduction

Fake news detection is a growing concern in the era of rapid information sharing. The goal of this project is to implement a machine learning-based system that can classify news articles as either "FAKE" or "REAL" based on their content.

### Models Used:
1. **Logistic Regression**
2. **Decision Tree Classifier**
3. **Random Forest Classifier**
4. **Gradient Boosting Classifier**

All models are evaluated on a test set to compare their performance.

## Dataset

The dataset used in this project consists of news articles labeled as either "FAKE" or "REAL". Each news article is represented by a textual content and a label. The dataset is assumed to be in a CSV file named `news.csv` with the following columns:
- **text**: The content of the news article.
- **label**: The label (either "FAKE" or "REAL").

The dataset is cleaned, processed, and used to train and test the machine learning models.

## Installation

To run this project locally, follow the steps below:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/fake-news-detection.git
   cd fake-news-detection
   ```

2. **Install Required Libraries:**
   Create a virtual environment (optional but recommended):
   ```bash
   python3 -m venv env
   source env/bin/activate    # for Linux/Mac
   env\Scripts\activate       # for Windows
   ```

   Then, install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. **Dataset Preparation:**
   - Ensure that the dataset `news.csv` is placed in the root folder of your project directory.

## File Structure

```
fake-news-detection/
│
├── news.csv                  # The dataset (ensure this is present)
├── fake_news_detection.py     # Main code file for the project
├── requirements.txt           # Python libraries required
└── README.md                  # This README file
```

## Usage

### Run the Code

To run the code, execute the `fake_news_detection.py` file. It will load the dataset, train the models, and allow you to input news articles for manual prediction:

```bash
python fake_news_detection.py
```

### Example of Manual Testing

You will be prompted to enter a news article:

```bash
Enter the news article for prediction: <Your news article here>
```

The system will then output the predictions from the Logistic Regression (Lr), Random Forest Classifier (RFC), and Gradient Boosting Classifier (GBC) models. Example:

```
Lr Prediction : It is a Real news
GBC Prediction : It is a Fake news
RFC Prediction : It is a Real news
```

## Preprocessing Steps

1. **Data Cleaning**: 
    - Converts all text to lowercase.
    - Removes URLs, HTML tags, punctuation, digits, and newlines.
    - Rows with missing values in either the "label" or "text" column are dropped.

2. **TF-IDF Vectorization**:
    - Converts textual data into numerical data using the **TF-IDF (Term Frequency-Inverse Document Frequency)** technique.
    - The text is split into a training and testing set, with 70% used for training and 30% used for testing.

## Model Details

### 1. **Logistic Regression**
   - A basic classifier that uses a logistic function to model binary outcomes.
   - It assigns a probability to each class (FAKE or REAL).

### 2. **Decision Tree Classifier**
   - A tree-based model where data is split into decision nodes based on feature values.
   - Each leaf node represents the final classification.

### 3. **Random Forest Classifier**
   - An ensemble method that combines multiple decision trees.
   - It improves the accuracy of predictions by averaging over many trees.

### 4. **Gradient Boosting Classifier**
   - A sequential ensemble model that builds trees to minimize the classification errors of previous trees.

The accuracy of these models and their performance on the test set can be evaluated using the **classification_report** function (though this part of the code is commented out for manual prediction purposes).

## Prediction

For manual prediction, you can input any news article, and the code will:
1. Clean the input using the same preprocessing steps as the training data.
2. Convert the input into numerical features using the TF-IDF vectorizer.
3. Use the trained models to predict whether the news is **"FAKE"** or **"REAL"**.

The results are returned in a human-readable format.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## Example Code File (`fake_news_detection.py`)

```python
import numpy as np
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Load Dataset
news = pd.read_csv('news.csv', low_memory=False)

# Label Mapping
news['label'] = news['label'].map({'FAKE': 0, 'REAL': 1})

# Drop unnecessary columns and clean data
news = news.drop(['title'], axis=1)
news = news.loc[:, ~news.columns.str.contains('^Unnamed')]

# Text Preprocessing
def wordpot(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'http?://\S+|www\.\S+', '', text)
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d', '', text)
        text = re.sub(r'\n', '', text)
    else:
        text = ''
    return text

news["text"] = news["text"].apply(wordpot)
news = news.dropna(subset=['label', 'text'])

# Splitting data into training and testing sets
x = news["text"]
y = news["label"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

# TF-IDF Vectorization
vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)

# Model Training
Lr = LogisticRegression()
Lr.fit(xv_train, y_train)

DTC = DecisionTreeClassifier()
DTC.fit(xv_train, y_train)

rfc = RandomForestClassifier()
rfc.fit(xv_train, y_train)

gbc = GradientBoostingClassifier()
gbc.fit(xv_train, y_train)

# Manual Testing
def output_label(n):
    return "It is a Fake news" if n == 0 else "It is a Real news"

def manual_testing(news):
    testing_news = {"text": [news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordpot)
    new_xv_test = vectorization.transform(new_def_test["text"])
    
    pred_lr = Lr.predict(new_xv_test)
    pred_gbc = gbc.predict(new_xv_test)
    pred_rfc = rfc.predict(new_xv_test)
    
    return "\n\nLr Prediction: {} \nGBC Prediction: {} \nRFC Prediction: {}".format(
        output_label(pred_lr[0]), output_label(pred_gbc[0]), output_label(pred_rfc[0]))

# Get user input
news_article = str(input("Enter the news article for prediction: "))
result = manual_testing(news_article)
print(result)
```

---

## Example `requirements.txt`

```txt
numpy
pandas
scikit-learn
```

---

This README provides all necessary information about the project, including setup instructions, file structure, usage, and an example of how to manually test the model for fake news detection.
