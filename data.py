#? Using Word Embeddings
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import pandas as pd

from pprint import pprint as pp

class DataPrep:
    def __init__(self, path, names):
        self._path = path
        self._names = names
        self._maxLen = 100

    #- Import
    def __importData(self):
        data = pd.read_csv(self._path, names=self._names, sep='\t')
        self._data = data['message']
        self._labels = data['label']


    #- Binarize Label
    def __binarizeLabels(self):
        lb = LabelBinarizer()
        self._labels = lb.fit_transform(self._labels)

    #- Seperate  
    def __seperateData(self):
        self._trainX, self._testX, self._trainY, self._testY = train_test_split(self._data, self._labels, test_size=0.33, random_state=42)
    
    #- Tokenize
    def __tokenize(self):
        tokenizer = Tokenizer(num_words=5000)
        tokenizer.fit_on_texts(self._trainX)
        self._trainX = tokenizer.texts_to_sequences(self._trainX)
        self._testX = tokenizer.texts_to_sequences(self._testX)

        #! Adding 1 because of reserved 0 index
        self._vocab_size = len(tokenizer.word_index) + 1


        self._trainX = pad_sequences(self._trainX, padding='post', maxlen= self._maxLen)
        self._testX = pad_sequences(self._testX, padding='post', maxlen= self._maxLen)

    

    #- you guessed it... prepare !
    def prepare(self):
        print("[INFO] Importing Data ...")
        self.__importData()
        self.__binarizeLabels()
        self.__seperateData()        
        print("[INFO] Tokenizing Data ...")
        self.__tokenize()

        return  self._trainX, self._testX, self._trainY, self._testY


