import pandas as pd
import numpy as np

'''
   This method is used to average the probabilties
   of all the images in a study.
   
   @:param probs. Predictions of the neural network
   @:param image_paths. file paths of all the images predicted
   @:param output file. File to be created which will contain the predictions per study
'''
def average_probabilities(probs, image_paths, output_file):
    averaged_probs = __average(probs, image_paths)
    
    # Create CSV filE 
    df = pd.DataFrame(list(averaged_probs.keys()), columns=['study'])
    df['label'] = list(averaged_probs.values())
    
    df.to_csv(output_file, index=False, header=None)
    print(output_file + " Created!")


def __average(probs, image_paths):
    img_paths = image_paths
    predictions = probs

    averaged_probabilities = {}

    for path in img_paths:
        study_name = '/'.join(path.split('/')[0:-1]) + "/"
        if study_name in averaged_probabilities:
            continue

        indices = [i for i, s in enumerate(img_paths) if study_name in s]
        probs = [predictions[i] for i in indices]

        # Based on the current directory structure, Keras has assigned the value 0 (Zero) to abnormal
        # and 1 (One) to Normal so in order to make sure that the class indices are in the right order
        # we have to subtract the predictions from 1.
        average = 1 - int(np.round(np.mean(probs)))

        averaged_probabilities[study_name] = average

    return averaged_probabilities
