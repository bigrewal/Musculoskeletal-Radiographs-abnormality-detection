from keras.applications.xception import Xception
from keras.layers import Dense, Flatten, Dropout
from keras import optimizers
from keras.models import Model


def get_xception(input_shape, learning_rate):
    xception_model = Xception(weights='imagenet', include_top=False, input_shape=(input_shape, input_shape, 3))

    x = xception_model.output
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=xception_model.input, outputs=predictions)

    for layer in xception_model.layers:
        layer.trainable = True

    adam = optimizers.Adam(lr=learning_rate)
    model.compile(optimizer=adam,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model
