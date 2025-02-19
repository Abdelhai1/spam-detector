# Spam Classifier using LSTM

## Overview

This project is a spam detection model using an LSTM (Long Short-Term Memory) neural network. The model is trained on a dataset of SMS messages, classifying them as either spam or ham (not spam).

## Features

- Uses an LSTM model for text classification.
- Tokenizes and pads sequences for input processing.
- Trained on a dataset of spam and ham messages.
- Saves and loads the trained model using TensorFlow/Keras.
- Predicts whether a given message is spam or ham.

## Installation

### Prerequisites

Ensure you have the following dependencies installed:

```bash
pip install tensorflow numpy pandas scikit-learn nltk
```

## Dataset

This model is trained on an SMS spam dataset. The dataset contains labeled messages categorized as spam or ham.

## Model Training

1. **Preprocessing**:

   - Tokenization and padding of text sequences.
   - Splitting the dataset into training and testing sets.

2. **Model Architecture**:

   - Embedding layer for word representation.
   - LSTM layers to capture sequential dependencies.
   - Dense layers for classification.

3. **Training**:

   - Model is compiled with `binary_crossentropy` loss and `adam` optimizer.
   - Trained on labeled SMS data.

4. **Saving Model**:

   - The trained model is saved as `spam_classifier_lstm.h5`.

## Usage

### Load the Model & Make Predictions

```python
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the trained model
loaded_model = load_model('spam_classifier_lstm.h5')

# Sample message for testing
sample_message = ["Congratulations! You've won a $1000 Walmart gift card. Click here to claim now."]
sample_seq = tokenizer.texts_to_sequences(sample_message)
sample_pad = pad_sequences(sample_seq, maxlen=max_len, padding='post', truncating='post')

# Make prediction
prediction = (loaded_model.predict(sample_pad) > 0.5).astype(int)
print("Prediction:", "Spam" if prediction == 1 else "Ham")
```

## Potential Issues

- Model might misclassify messages if they are well-disguised spam.
- Preprocessing (tokenization, padding) must match the settings used during training.
- Ensure the tokenizer used for training is the same as the one used for inference.

## Future Improvements

- Train on a larger dataset for better accuracy.
- Use a transformer-based model (e.g., BERT) for improved text understanding.
- Implement real-time email/SMS classification API.

## License

This project is open-source and available under the MIT License.

