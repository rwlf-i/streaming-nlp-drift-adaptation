# tfidf_logreg.py

import sys
import os

sys.path.append(os.path.abspath('../src'))


from model_adapter import ModelAdapter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


class TfidfLogregAdapter(ModelAdapter):
    def __init__(self, max_features=3000, stop_words="english", seed=42):
        self.max_features = max_features
        self.stop_words = stop_words
        self.seed = seed

        self.vectorizer = TfidfVectorizer(max_features=self.max_features, stop_words=self.stop_words)
        self.model = LogisticRegression(max_iter=1000, random_state=self.seed)

    def fit(self, texts, labels):
        X = self.vectorizer.fit_transform(texts)
        self.model.fit(X, labels)

    def predict(self, texts):
        X = self.vectorizer.transform(texts)
        return self.model.predict(X)

    def update(self, texts, labels):
        pass

    def evaluate(self, texts, labels):
        from sklearn.metrics import accuracy_score, f1_score
        X = self.vectorizer.transform(texts)
        y_pred = self.model.predict(X)
        acc = accuracy_score(labels, y_pred)
        f1 = f1_score(labels, y_pred, average='macro')
        return acc, f1
