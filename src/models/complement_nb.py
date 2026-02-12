# complement_nb_adapter.py

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.naive_bayes import ComplementNB
from model_adapter import ModelAdapter

class ComplementNBAdapter(ModelAdapter):
    def __init__(self, max_features=3000, stop_words="english", seed=42, alpha=1.0):
        self.max_features = max_features
        self.stop_words = stop_words
        self.seed = seed
        self.alpha = alpha

        self.vectorizer = HashingVectorizer(
            n_features=self.max_features,
            stop_words=self.stop_words,
            alternate_sign=False,
            norm=None   
        )
        self.model = None
        self._classes = None

    def _new_model(self):
        return ComplementNB(alpha=self.alpha)

    def fit(self, texts, labels):
        X = self.vectorizer.transform(texts)
        y = np.asarray(labels)

        self.model = self._new_model()
        self.model.fit(X, y)

        self._classes = np.unique(y)

    def predict(self, texts):
        X = self.vectorizer.transform(texts)
        return self.model.predict(X)

    def update(self, texts, labels):
        X = self.vectorizer.transform(texts)
        y = np.asarray(labels)

        if self.model is None:
            self.model = self._new_model()

        if self._classes is None:
            self._classes = np.unique(y)
            self.model.partial_fit(X, y, classes=self._classes)
        else:
            self.model.partial_fit(X, y)

    def evaluate(self, texts, labels):
        from sklearn.metrics import accuracy_score, f1_score
        y_true = np.asarray(labels)
        y_pred = self.predict(texts)
        return accuracy_score(y_true, y_pred), f1_score(y_true, y_pred, average="macro")
