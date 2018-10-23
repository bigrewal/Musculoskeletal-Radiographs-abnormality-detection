from keras.applications.densenet import DenseNet169
from keras.layers import Dense
from keras import optimizers
from keras.models import Model


def get_dense169(input_shape, learning_rate):
    # create the base pre-trained model
    dense_169_model = DenseNet169(include_top=False, weights='imagenet', input_shape=(input_shape, input_shape, 3))
    x = dense_169_model.output
    predictions = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=dense_169_model.input, outputs=predictions)

    for layer in dense_169_model.layers:
        layer.trainable = True

    adam = optimizers.Adam(lr=learning_rate)
    model.compile(optimizer=adam,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model
