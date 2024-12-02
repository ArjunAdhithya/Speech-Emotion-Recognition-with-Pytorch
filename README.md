# Speech-Emotion-Recognition-with-Pytorch

This project implements **Speech Emotion Recognition** (SER) using **machine learning** techniques. The model classifies emotions from speech signals using audio features extracted from the **TESS (Toronto Emotional Speech Set)** dataset. The primary focus of this project is on **audio signal processing**, followed by training a machine learning model (specifically, **Neural Networks**) to predict emotions from speech.

## Project Overview

Emotion recognition from speech is a challenging task that involves processing raw audio signals to extract meaningful features. The following steps are involved in the process:

1. **Audio Signal Preprocessing**: The raw audio files are preprocessed using various audio signal processing techniques, including **MFCC (Mel Frequency Cepstral Coefficients)** extraction.
2. **Feature Extraction**: We extract features such as **MFCCs**, **chroma**, **spectral contrast**, and **zero-crossing rate** that capture the emotional tone of speech.
3. **Model Training**: A **fully connected neural network (FCNN)** is trained using these extracted features.
4. **Emotion Classification**: The model predicts the emotion associated with a given speech sample.

### Features Extracted from Audio

1. **MFCC (Mel-Frequency Cepstral Coefficients)**:
   - **MFCCs** are one of the most important features used in speech and emotion recognition tasks. They represent the short-term power spectrum of sound.
   - In this project, we extract 13 MFCCs for each audio sample to use as input features for the model.

2. **Chroma Features**:
   - **Chroma** features capture the harmonic content in the audio, which is important for tonal and emotional recognition.
   
3. **Spectral Contrast**:
   - **Spectral Contrast** measures the difference in amplitude between peaks and valleys in a sound spectrum. It helps distinguish speech tones and emotional states.

4. **Zero-Crossing Rate**:
   - **Zero-Crossing Rate (ZCR)** counts the number of times the audio signal crosses the zero axis. It’s used to capture speech rhythm and energy.

---

## Dataset

The **TESS (Toronto Emotional Speech Set)** dataset is used for training and testing. The dataset consists of **2000+** audio clips spoken by professional actors, each representing different emotions such as happy, sad, angry, fearful, and surprised.

- **Emotion Classes**:
  - `angry`
  - `happy`
  - `fear`
  - `sad`
  - `surprise`
  - `neutral`

The dataset is organized into directories where each folder corresponds to an emotion label, and each file contains a speech sample.

---

## Model Architecture

The model used in this project is a simple **Fully Connected Neural Network (FCNN)**. The network architecture consists of the following layers:

1. **Input Layer**: The input layer receives a vector of features (e.g., MFCCs, Chroma, Spectral Contrast).
2. **Hidden Layers**: Multiple fully connected layers that process the features. **ReLU (Rectified Linear Unit)** activation function is used for non-linearity.
3. **Output Layer**: A final softmax layer for classification into one of the six emotions.

### Model Design

- **Loss Function**: **Cross-Entropy Loss** is used since this is a multi-class classification problem.
- **Optimizer**: **Adam Optimizer** is used for efficient model training.
- **Activation Function**: **ReLU** is used in hidden layers, and **Softmax** is used for the output layer to predict class probabilities.

---

