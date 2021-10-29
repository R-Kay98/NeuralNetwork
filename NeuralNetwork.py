import tensorflow as tf

def buildModel():
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(300, activation='relu'),
        tf.keras.layers.Dense(10, activation='sigmoid')
    ])
    model.compile(
        optimizer='Adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def normalizeData(train, test):
    normTrain = train/255.0
    normTest = test/255.0
    return normTrain, normTest

def getMNistDataSet():
    (xTrain, yTrain), (xTest, yTest) = tf.keras.datasets.mnist.load_data()
    xTrain, xTest = normalizeData(xTrain, xTest)
    yTrain = tf.keras.utils.to_categorical(yTrain, 10)
    yTest = tf.keras.utils.to_categorical(yTest, 10)
    return xTrain, yTrain, xTest, yTest

def main():
    xTrain, yTrain, xTest, yTest = getMNistDataSet()
    model = buildModel()
    model.fit(xTrain, yTrain, epochs=10)
    testLoss, testAccuracy = model.evaluate(xTest, yTest, verbose=2)
    print('Test accuracy: {}'.format(testAccuracy))

if __name__ == "__main__":
    main()