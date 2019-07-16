import numpy as np
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

class Predict:
    def __init__(self, modelPath, maxlen):
        self._model = load_model(modelPath)
        self._maxLen = maxlen

    def __tokenizeX(self, X):
        tokenizer = Tokenizer(num_words=5000)
        tokenizer.fit_on_texts(X)
        tokens = tokenizer.texts_to_sequences(X)
        tokens = pad_sequences(tokens, padding='post', maxlen=self._maxLen)
        return tokens
    
    def predict(self, sentence):
        X = self.__tokenizeX(sentence)

        pred = self._model.predict(X)
        for _, p in enumerate(pred):
            if p[0] > 0:
                print('Spam ', p)
            else:
                print('Ham ', p)



