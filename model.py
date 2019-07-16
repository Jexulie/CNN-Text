from keras.models import Sequential
from keras import layers
import matplotlib
import matplotlib.pyplot as plt


#- Model Class
class Embedding_Model:
    def __init__(self, input_dim, input_len):
        self._input_dim = input_dim
        self._output_dim = 50 # _embedding_dim
        self._input_len = input_len


    def defineModelText(self):
        print("[INFO] Defining Model ...")
        model = Sequential()

        model.add(layers.Embedding(
            input_dim=self._input_dim,
            output_dim=self._output_dim,
            input_length=self._input_len
        ))
        #- Actual CNN layer with pooling
        model.add(layers.Conv1D(128, 5, activation='relu'))
        model.add(layers.GlobalMaxPooling1D())
        model.add(layers.Dense(10, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        print(model.summary())
        return model

    def defineModelImage(self, input_shape, classes, optimizer='adam'):
        print("[INFO] Defining Model ...")
        model = Sequential()
        model.add(layers.Conv2D(32, (3, 3), padding='same', input_shape=input_shape))
        model.add(layers.Activation('relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(classes))
        model.add(layers.Activation("softmax"))
        
        if classes > 1:
            loss = 'binary_crossentropy'
        else:
            loss = 'categorical_crossentropy'        

        model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

        print(model.summary())
        return model


#- Training Class
class Train:
    def __init__(self, model, trainX, testX, trainY, testY, epochs, batch_size):
        self._model = model
        self._trainX = trainX
        self._trainY = trainY
        self._testX = testX
        self._testY = testY
        self._epochs = epochs
        self._batchSize = batch_size

    
    def plot_history(self, history):
        acc = history.history['acc']
        val_acc = history.history['val_acc']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        x = range(1, len(acc) + 1)

        plt.figure(figsize=(12, 5))
        plt.grid()
        plt.subplot(1, 2, 1)
        plt.plot(x, acc, 'b', label='Training acc')
        plt.plot(x, val_acc, 'r', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(x, loss, 'b', label='Training loss')
        plt.plot(x, val_loss, 'r', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()

        plt.savefig('acc-loss-plot.png')
        plt.show()

    def train(self):
        print("[INFO] Training Model ...")
        history = self._model.fit(
            x=self._trainX, y=self._trainY,
            epochs=self._epochs, 
            verbose=False,
            validation_data=(self._testX, self._testY),
            batch_size=self._batchSize
        )

        print("[INFO] Saving Model ...")
        self._model.save('text.model')

        print("[INFO] Evaluating Training ...")

        tr_loss, tr_accuracy = self._model.evaluate(
            self._trainX, self._trainY, verbose=False
        )

        print(f'Training Accuracy: {tr_accuracy:.4f} | Training Loss: {tr_loss:.4f}')

        te_loss, te_accuracy = self._model.evaluate(
            self._testX, self._testY, verbose=False
        )

        print(f'Testing Accuracy: {te_accuracy:.4f} | Testing Loss: {te_loss:.4f}')

        self.plot_history(history)



