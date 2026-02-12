from abc import ABC, abstractmethod


class ModelAdapter(ABC):
    # базовый интерфейс адаптера модели
    @abstractmethod
    def fit(self, texts, labels):
        # обучение модели на данных
        raise NotImplementedError

    @abstractmethod
    def predict(self, texts):
        # получение предсказаний по текстам
        raise NotImplementedError

    def update(self, texts, labels):
        # обновление модели на новых данных (опционально)
        pass
