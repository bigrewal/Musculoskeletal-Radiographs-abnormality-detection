##  Musculoskeletal Radiographs Abnormality Detection

Bone X-Ray Deep Learning Competition. Read more [here](https://stanfordmlgroup.github.io/competitions/mura/).

This was my first attempt at this challenge, I used [Xception](https://arxiv.org/pdf/1610.02357.pdf) architecture for this and received a Cappa score of `0.687` on the test set. [Leaderboard](https://stanfordmlgroup.github.io/competitions/mura/) Model name: `xception(single model) bimal`

#### Metrics based on study type per image in the Validation Set

Xception model received an accracy of `83%` on the Validation set.
    
    ===== ELBOW ======
    roc_auc_score:  0.8655411655874191
    Sensitivity:  0.9702127659574468
    Specificity:  0.7608695652173914
    Cohen-Cappa-Score:  0.7327090673094752
    F1 Score:  0.8803088803088802

    ===== FINGER ======
    roc_auc_score:  0.7782454879110069
    Sensitivity:  0.8925233644859814
    Specificity:  0.6639676113360324
    Cohen-Cappa-Score:  0.546259842519685
    F1 Score:  0.7827868852459016

    ===== FOREARM ======
    roc_auc_score:  0.8475496688741722
    Sensitivity:  0.96
    Specificity:  0.7350993377483444
    Cohen-Cappa-Score:  0.6945780209114572
    F1 Score:  0.8622754491017965

    ===== HAND ======
    roc_auc_score:  0.7469005642437376
    Sensitivity:  0.959409594095941
    Specificity:  0.5343915343915344
    Cohen-Cappa-Score:  0.5262369439474013
    F1 Score:  0.840064620355412

    ===== HUMERUS ======
    roc_auc_score:  0.8754826254826255
    Sensitivity:  0.8581081081081081
    Specificity:  0.8928571428571429
    Cohen-Cappa-Score:  0.7500964134207482
    F1 Score:  0.8758620689655172

    ===== SHOULDER ======
    roc_auc_score:  0.7810299129117759
    Sensitivity:  0.8210526315789474
    Specificity:  0.7410071942446043
    Cohen-Cappa-Score:  0.5625817210210153
    F1 Score:  0.7918781725888325

    ===== WRIST ======
    roc_auc_score:  0.8414648910411623
    Sensitivity:  0.9642857142857143
    Specificity:  0.7186440677966102
    Cohen-Cappa-Score:  0.6986424046110609
    F1 Score:  0.8796992481203008

### Steps to train the Neural Network: (Train this on the GPU)

1. Clone this project.

2. Download the dataset from the competetion home page and place it in the cloned project directory. For example: `Musculoskeletal-Radiographs-abnormality-detection/`

3. Run the `main_train.ipynb`. 

### Project Dependencies
1. `pip install tensorflow-gpu`
2. `pip install keras`
3. `pip install scipy`
4. `pip install sklearn`
5. `pip install pandas`
6. `pip install numpy`
7. `pip install jupyter`

Neural Net was trained in AWS using the `p2.xlarge` instance, I have created an AMI which comes with all of the above dependencies installed. AMI-ID: `ami-0337b3fba4a212c7f`, make sure you've selected the `Ireland` region if you decide to use this AMI. 

#### Project Structure:
1. `src/data/dataloader.py`: Filters out normal and abnormal x-ray images and adds them to their own directories for keras    `ImageDataGenerator.flow_from_directory()`. For example `train/normal/` and `train/abnormal/`

2. `src/data/postprocessor.py`: Avergaes out the probablities of images per patient study type. 

3. `src/model/*`: Creates a `Xception` or `Dense169` neural network model

4. `src/predict.py`: Run this when ready for submission. Note: Change the value of the `trained_model_path` variable.
