import warnings
import numpy as np
from sklearn.utils import class_weight
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.applications.densenet import preprocess_input as densenet_preprocess
from keras.applications.xception import preprocess_input as xception_preprocess

# Ignore python warnings
warnings.filterwarnings('ignore')

# Import custom modules
from src.data.dataloader import create_dir, load_images
from src.model.xception import get_xception
from src.model.dense169 import get_dense169

# Create directories for Keras ImageDataGenerator.flow_from_directory
train_dir_abnormal = "./train_data/abnormal"
train_dir_normal = "./train_data/normal"
valid_dir_abnormal = "./valid_data/abnormal"
valid_dir_normal = "./valid_data/normal"

def create_directories():
    create_dir(train_dir_abnormal)
    create_dir(train_dir_normal)
    create_dir(valid_dir_abnormal)
    create_dir(valid_dir_normal)

def load_image_data():
    # Load Train Images
    train_images_path = 'MURA-v1.1/train_image_paths.csv'
    train_image_paths = load_images(train_images_path)
    print("Training Set Images loaded!")

    # Load Validation set images
    valid_images_path = 'MURA-v1.1/valid_image_paths.csv'
    valid_image_paths = load_images(valid_images_path)
    print("Validation Set Images loaded!")

    return train_image_paths, valid_image_paths

def create_data_generators(input_shape, batch_size, preprocess_input):
    train_datagen = ImageDataGenerator(
        rotation_range=30,
        horizontal_flip=True,
        preprocessing_function=preprocess_input)

    train_generator = train_datagen.flow_from_directory(
        'train_data/',
        target_size=(input_shape, input_shape),
        batch_size=batch_size,
        class_mode='binary')

    valid_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input)

    valid_generator = valid_datagen.flow_from_directory(
        'validation_data',
        target_size=(input_shape, input_shape),
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False)

    return train_generator, valid_generator

def compute_class_weights(generator):
    return class_weight.compute_class_weight('balanced', np.unique(generator.classes), generator.classes)

def create_callbacks(model_name):
    filepath = f"{model_name}-improvement-{{epoch:02d}}-{{val_acc:.2f}}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    tensorboard = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=False)
    return [checkpoint, tensorboard]

def train_model(model, train_generator, valid_generator, weights, callbacks, batch_size, epochs):
    training_data_size = len(train_generator.filenames)
    validation_data_size = len(valid_generator.filenames)

    print("Number of Training examples: ", training_data_size)
    print("Number of Validation examples: ", validation_data_size)

    model.fit_generator(
        train_generator,
        validation_data=valid_generator,
        steps_per_epoch=training_data_size // batch_size,
        class_weight=weights,
        callbacks=callbacks,
        validation_steps=validation_data_size // batch_size,
        epochs=epochs
    )

def main():
    # Hyperparameters
    input_shape = 320
    batch_size = 8
    epochs = 10
    learning_rate = 0.0001

    create_directories()
    train_image_paths, valid_image_paths = load_image_data()

    # Train Dense169 Model
    dense169_model = get_dense169(input_shape, learning_rate)
    dense_train_generator, dense_valid_generator = create_data_generators(input_shape, batch_size, densenet_preprocess)
    dense_weights = compute_class_weights(dense_train_generator)
    dense_callbacks = create_callbacks("dense169")
    
    print("Training Dense169 Model...")
    train_model(dense169_model, dense_train_generator, dense_valid_generator, dense_weights, dense_callbacks, batch_size, epochs)

    # Train Xception Model
    xception_model = get_xception(input_shape, learning_rate)
    xception_train_generator, xception_valid_generator = create_data_generators(input_shape, batch_size, xception_preprocess)
    xception_weights = compute_class_weights(xception_train_generator)
    xception_callbacks = create_callbacks("xception")
    
    print("Training Xception Model...")
    train_model(xception_model, xception_train_generator, xception_valid_generator, xception_weights, xception_callbacks, batch_size, epochs)

if __name__ == "__main__":
    main()
