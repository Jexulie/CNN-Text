from model import Embedding_Model, Train
from predict import Predict
from data import DataPrep

path = './datasets/SMSSpamCollection.txt'
names = ['label', 'message']
EPOCH = 100
BATCH_SIZE = 10


def startTraining():
    d = DataPrep(path, names)
    trainX, testX, trainY, testY = d.prepare()
    vocab_size = d._vocab_size
    maxlen = d._maxLen

    # maxlen
    e = Embedding_Model(vocab_size, maxlen)
    model = e.defineModelText()

    t = Train(model, trainX, testX, trainY, testY, EPOCH, BATCH_SIZE)
    t.train()


def pred():
    p = Predict('text-cnn.model', 100)

    #! Not working -_- preds are way off
    p.predict(['Whats up champ', 
    'call us to get some free goods', 
    'when will you are going ?'])

# startTraining()
pred()     