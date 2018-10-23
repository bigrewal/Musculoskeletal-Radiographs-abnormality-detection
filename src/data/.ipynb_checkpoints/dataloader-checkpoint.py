import pandas as pd
from PIL import Image
import pathlib


'''
    Utility function to
    create a directory
    
    @:param directory name
    
    NOTE: create the directory before running the load_images_train() method
'''


def create_dir(dir_name):
    pathlib.Path(dir_name).mkdir(parents=True, exist_ok=True)
    print("Created Directorty: ", dir_name)


'''
    This method is responsible for adding
    each image in the following directory
    structure.
    
    train_data / valid_data
        |
        |
        ---> abnormal
        ---> normal
    
    Note: Directory names are hardcoded for now, so please create the directory
    structure above for training and validation.
'''
def load_images(images_path):
    image_paths = pd.read_csv(images_path, header=None).as_matrix().flatten().tolist()

    i = 1
    for image_path in image_paths:

        img = Image.open(image_path)
        name = "_".join(image_path.split("/")[2:])

        if "train" in image_path:
            if "positive" in image_path:
                img.save("./train/abnormal/" + name, "PNG")   # TO-DO: Need to make the directory names dynamic
            else:
                img.save("./train/normal/" + name, "PNG")

        elif "valid" in image_path:
            if "positive" in image_path:
                img.save("./train/abnormal/" + name, "PNG")
            else:
                img.save("./train/normal/" + name, "PNG")

        i += 1

        if i % 1000:
            print("1000/" + len(image_paths) + " moved!")
