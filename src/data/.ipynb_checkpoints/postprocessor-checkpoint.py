import pandas as pd
import numpy as np


def average_probabilities(probs, image_paths, output_file):
    averaged_probs = __average(probs, image_paths)
    
    # Create CSV filE 
    df = pd.DataFrame(list(averaged_probs.keys()), columns=['study'])
    df['label'] = list(averaged_probs.values())
    
    df.to_csv(output_file, index=False, header=None)
    return averaged_probs


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

        average = 1 - int(np.round(np.mean(probs)))

        averaged_probabilities[study_name] = average

    return averaged_probabilities
