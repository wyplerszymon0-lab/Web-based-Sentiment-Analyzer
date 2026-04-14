import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

def train():
    data = {
        'text': [
            'Excellent quality', 'Bad service', 'Great experience', 
            'Very disappointed', 'Highly recommended', 'Waste of money'
        ],
        'sentiment': [1, 0, 1, 0, 1, 0]
    }
    df = pd.DataFrame(data)
    model = make_pipeline(TfidfVectorizer(), MultinomialNB())
    model.fit(df['text'], df['sentiment'])
    joblib.dump(model, '../models/model.pkl')

if __name__ == "__main__":
    train()
