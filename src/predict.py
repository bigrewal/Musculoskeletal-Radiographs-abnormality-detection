import warnings
warnings.filterwarnings('ignore')  # Ignore python warnings

import numpy as np
import pandas as pd
import sys

from data.postprocessor import average_probabilities
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.xception import preprocess_input


class Predict(object):

    def __init__(self, model_name):
        print("Loading Model...")
        self.model_mura = load_model(model_name)
        print("Model loaded!")

    def predict(self):
        # Hyperparameters
        batch_size = 1
        input_shape = 320

        image_paths = pd.read_csv(input_data_csv_file, header=None).as_matrix().flatten().tolist()

        # Get the prefix from the image path. For example: MURA-v1.1/train/
        prefix = image_paths[0]
        prefix = prefix.split("/")[:2]
        prefix = '/'.join(prefix) + "/"

        test_dir = prefix

        test_datagen = ImageDataGenerator(
            rescale=1. / 255,
            preprocessing_function=preprocess_input)

        test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=(input_shape, input_shape),
            batch_size=batch_size,
            class_mode=None,
            shuffle=False)

        N = len(test_generator.filenames)

        print("Predicting ...")
        probs = self.model_mura.predict_generator(test_generator, steps=N)
        print("Predictions Done!")

        file_names = test_generator.filenames
        file_names = [prefix + file for file in file_names]

        average_probabilities(probs, file_names, output_file_path)


if __name__ == '__main__':
    input_data_csv_file = sys.argv[1]
    output_file_path = sys.argv[2]

    trained_model_path = "src/xception_v2-improvement-06-0.82.hdf5"

    print("Predicting ...")

    p = Predict(trained_model_path)
    p.predict()

    print("File created!")
