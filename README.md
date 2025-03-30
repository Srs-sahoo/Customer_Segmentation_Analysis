# Customer_Segmentation_Analysis
**Sentiment Analysis Report**

## **Introduction**
Sentiment analysis is a Natural Language Processing (NLP) technique used to determine the sentiment expressed in customer reviews. This project analyzes customer feedback to identify positive and negative sentiments, helping businesses improve their products and services.

## **Objectives**
- Preprocess textual data.
- Apply NLP techniques to analyze sentiment.
- Visualize sentiment distribution.

## **Methodology**

### **1. Data Loading & Cleaning**
#### **Importing Libraries**
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
```
- Libraries like Pandas and NumPy are used for data manipulation.
- Seaborn and Matplotlib help in data visualization.
- Warnings are suppressed for a cleaner output.

#### **Loading the Dataset**
```python
data = pd.read_csv('sentiment_analysis.csv')
data.head()
```
- Reads the dataset containing customer reviews.
- Displays the first few rows to understand the structure.

#### **Handling Missing and Duplicate Data**
```python
data.isnull().sum()
data.duplicated().sum()
data.drop_duplicates(inplace=True)
```
- Checks for and removes duplicate entries to ensure clean data.

### **2. Text Preprocessing**
#### **Downloading NLP Resources**
```python
import re
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
```
- Downloads necessary NLP resources for tokenization, stopword removal, and lemmatization.

#### **Text Cleaning Function**
```python
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text).lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)
```
- Removes special characters and converts text to lowercase.
- Tokenizes text into words and removes stopwords.
- Applies lemmatization to convert words to their base forms.

#### **Applying Text Cleaning**
```python
data['cleaned_review'] = data['text'].apply(preprocess_text)
```
- Cleans all customer reviews in the dataset.

### **3. Sentiment Analysis**
#### **Applying Sentiment Analysis Model**
```python
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()
```
- Uses the VADER model for sentiment analysis.

#### **Calculating Sentiment Scores**
```python
data['sentiment_score'] = data['cleaned_review'].apply(lambda x: sia.polarity_scores(x)['compound'])
data['predicted_sentiment'] = data['sentiment_score'].apply(lambda x: 'positive' if x >= 0 else 'negative')
```
- Computes a sentiment score for each review.
- Classifies reviews as positive or negative based on the score.

### **4. Visualization**
#### **Displaying Sentiment Distribution**
```python
sns.countplot(x='predicted_sentiment', data=data)
plt.title('Sentiment Distribution')
plt.show()
```
- Creates a bar plot showing the number of positive and negative reviews.

#### **Generating Word Cloud (Optional)**
```python
!pip install wordcloud
from wordcloud import WordCloud
wordcloud = WordCloud().generate(' '.join(data['cleaned_review']))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()
```
- Displays a word cloud to visualize the most common words in customer reviews.

## **Conclusion**
- The sentiment analysis successfully identified positive and negative sentiments in customer reviews.
- The cleaned dataset and visualization help in understanding customer feedback trends.
- Further improvements can include using deep learning models for enhanced accuracy.

