import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Dropout, Conv1D, MaxPooling1D, SpatialDropout1D
from tensorflow.keras.initializers import Constant
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import joblib
import json

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

def load_glove_embeddings(file_path):
    """Load pre-trained GloVe embeddings."""
    embeddings_index = {}
    with open(file_path, encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index

def preprocess_text(text):
    """Clean and normalize text data."""
    text = re.sub(r'\W', ' ', text).lower()
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])
    return text

def create_embedding_matrix(word_index, embeddings_index, embedding_dim, max_words):
    """Create an embedding matrix for the Embedding layer."""
    embedding_matrix = np.zeros((max_words, embedding_dim))
    for word, i in word_index.items():
        if i < max_words:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
    return embedding_matrix

# Load GloVe embeddings
print("Loading pre-trained GloVe embeddings...")
embeddings_index = load_glove_embeddings('./glove.6B.300d.txt')
embedding_dim = 300

# Load datasets
print("Loading datasets...")
data_training = pd.read_csv('https://github.com/albahnsen/MIAD_ML_and_NLP/raw/main/datasets/dataTraining.zip', encoding='UTF-8', index_col=0)
data_testing = pd.read_csv('https://github.com/albahnsen/MIAD_ML_and_NLP/raw/main/datasets/dataTesting.zip', encoding='UTF-8', index_col=0)

# Drop 'rating' column from training data
data_training.drop(columns=['rating'], inplace=True)

# Preprocess genres
print("Preprocessing genres...")
data_training['genres'] = data_training['genres'].map(lambda x: eval(x))
mlb = MultiLabelBinarizer()
y_genres = mlb.fit_transform(data_training['genres'])

# Clean and normalize text data
print("Preprocessing text data...")
data_training['plot'] = data_training['plot'].apply(preprocess_text)
data_testing['plot'] = data_testing['plot'].apply(preprocess_text)

# Tokenize and sequence text data
print("Tokenizing and sequencing text data...")
max_words = 20000
maxlen = 600
tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(data_training['plot'])

X_train_seq = tokenizer.texts_to_sequences(data_training['plot'])
X_train_pad = pad_sequences(X_train_seq, maxlen=maxlen)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train_pad, y_genres, test_size=0.2, random_state=42)

# Create embedding matrix
print("Creating embedding matrix...")
embedding_matrix = create_embedding_matrix(tokenizer.word_index, embeddings_index, embedding_dim, max_words)

# Build BiLSTM model with pre-trained embeddings
print("Building BiLSTM model...")
model = Sequential([
    Embedding(input_dim=max_words, output_dim=embedding_dim, embeddings_initializer=Constant(embedding_matrix), input_length=maxlen, trainable=False),
    SpatialDropout1D(0.3),
    Bidirectional(LSTM(128, dropout=0.3, recurrent_dropout=0.3, return_sequences=True)),
    Conv1D(64, 5, activation='relu'),
    MaxPooling1D(pool_size=4),
    Bidirectional(LSTM(128, dropout=0.3, recurrent_dropout=0.3)),
    Dense(100, activation='relu'),
    Dropout(0.3),
    Dense(y_genres.shape[1], activation='sigmoid')
])

# Compile the model
print("Compiling the model...")
model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

# Set up callbacks for training
print("Setting up callbacks...")
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001)

# Train the model
print("Training the model...")
history = model.fit(X_train, y_train, epochs=20, batch_size=64, validation_data=(X_val, y_val), callbacks=[early_stopping, reduce_lr], verbose=1)

# Save the model
print("Saving the model...")
model.save('bilstm_model.h5')

# Save the tokenizer
print("Saving the tokenizer...")
with open('tokenizer.json', 'w') as f:
    f.write(tokenizer.to_json())

# Save the MultiLabelBinarizer
print("Saving the MultiLabelBinarizer...")
joblib.dump(mlb, 'mlb.pkl')
