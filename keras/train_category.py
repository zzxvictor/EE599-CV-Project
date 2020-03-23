from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from data2 import polyvore_dataset
from utils import Config
from tensorflow.keras.applications.resnet50 import ResNet50
import tensorflow.keras as tfk
import tensorflow as tf
if __name__=='__main__':

    # data generators
    dataset = polyvore_dataset()
    trainList, valList, nClass = dataset.readMeta()
    if Config['debug']:
        trainList = trainList[:100]
        valList = valList[:100]
    trainData = dataset.load(trainList, batchSize=Config['batch_size'])
    valData = dataset.load(valList, batchSize=Config['batch_size'])

    # build model
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        base_model = ResNet50(weights='imagenet', include_top=False)
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        predictions = Dense(nClass, activation = 'softmax', kernel_regularizer=tfk.regularizers.l2())(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        for layer in base_model.layers:
            layer.trainable = False

        optimizer = tfk.optimizers.RMSprop(Config['learning_rate'])
    # define optimizers
        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # training - num worker is obsolete now
    model.fit(trainData, validation_data=valData, epochs=Config['num_epochs'],
              steps_per_epoch=100, validation_steps=10, callbacks=[tfk.callbacks.EarlyStopping(patience=1)])


'''

    # build model
#strategy = tf.distribute.MirroredStrategy()
#with strategy.scope():
base_model = ResNet50(weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu', kernel_regularizer=tfk.regularizers.l2())(x)
predictions = Dense(nClass, activation = 'softmax', )(x)
model = Model(inputs=base_model.input, outputs=predictions)

optimizer = tfk.optimizers.RMSprop(Config['learning_rate'])
    # define optimizers
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

    # training - num worker is obsolete now
model.fit(trainData, validation_data=valData, epochs=Config['num_epochs'],
              steps_per_epoch=1000, validation_steps = 100)
'''



