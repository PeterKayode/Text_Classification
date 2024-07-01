# Text Classification using Naive Bayes

## Description
This project implements text classification using the Naive Bayes algorithm. The primary goal is to classify text data into predefined categories.

## Usage

1. Open the Jupyter Notebook:
    ```bash
    jupyter notebook Text_Classification_Naive_Bayes_Classification.ipynb
    ```
2. Execute the notebook cells to perform text classification.

## Explanation of Code

The notebook demonstrates the following steps for text classification using the Naive Bayes algorithm:

1. **Data Loading**:
    - Loads sample text data into a pandas DataFrame.

2. **Text Vectorization**:
    - Converts the text data into numerical features using `CountVectorizer`.

3. **Model Training**:
    - Uses a `MultinomialNB` classifier to train the model on the vectorized text data.

4. **Prediction**:
    - Makes predictions on new text data using the trained model.

### Example Code Snippet

```python
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Sample data
data = {'text': ["I love programming", "Python is great", "I hate bugs"],
        'label': [1, 1, 0]}
df = pd.DataFrame(data)

# Model pipeline
model = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', MultinomialNB())
])

# Training the model
model.fit(df['text'], df['label'])

# Predicting
predictions = model.predict(["I love Python", "I hate programming"])
print(predictions)
