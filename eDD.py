from nltk.corpus import stopwords
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import string
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dropout, Dense, Bidirectional, LSTM
from pymorphy2 import MorphAnalyzer


df_train = pd.read_csv('dataxy2.0.txt', sep=';', names=['text', 'emotions'])



df_train = df_train.dropna()


morph = MorphAnalyzer()
stopwords = stopwords.words("russian")


df_train['text'] = df_train['text'].apply(morph.normal_forms)


def text_process(mess):

  nopunc = [char for char in mess if char not in string.punctuation]

  nopunc = ''.join(nopunc)

  word_seq = [word for word in nopunc.split() if word.lower() not in stopwords]
  return word_seq

text = df_train['text'].apply(text_process)


max_len = 100
max_words = 20000
tokenizer = Tokenizer(num_words=max_words, filters="[A-Za-z0-9!#$%&'()*+,./:;<=>?@[\]^_`{|}~—\"\-]+")

tokenizer.fit_on_texts(text)
sequences = tokenizer.texts_to_sequences(text)


data = pad_sequences(sequences, maxlen=max_len)


encode = LabelEncoder()

y = encode.fit_transform(df_train['emotions'])
y_data = np_utils.to_categorical(y)



x_train = data
y_train = y_data



print(df_train.emotions.value_counts())
print(df_train.shape)

model = Sequential()
model.add(Embedding(max_words, 64, input_length=max_len))
model.add(Bidirectional(LSTM(128, return_sequences=True)))
model.add(Bidirectional(LSTM(64)))
model.add(Dropout(0.5))
model.add(Dense(y_data.shape[1], activation='softmax'))
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

predictions = model.fit(data, y_data, batch_size=32, validation_split=0.1, epochs=20)

model.save('emotion_detection.h5')

