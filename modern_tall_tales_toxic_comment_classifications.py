
import tensorflow as tf
tf.__version__

resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='')
tf.config.experimental_connect_to_cluster(resolver)
# This is the TPU initialization code that has to be at the beginning.
tf.tpu.experimental.initialize_tpu_system(resolver)
print("All devices: ", tf.config.list_logical_devices('TPU'))

!pip install pandas matplotlib sklearn

# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'

# Upload the dataset to Drive
from google.colab import files
upload = files.upload()

# Import the dataset 
df = pd.read_csv('train.csv')

# Check the first 5 of the dataset
df.head()

df.shape, df.dtypes

# Check for the data anomaly and missing values (summary of the dataset)
df.info(verbose=True, show_counts=False)

# Check the types of toxicity (on the name of the columns)
df.columns[2:]

# Check for the unique values on each variables
df.nunique(axis=0)

# Check for the count, mean, std dev, min and max of the dataset
df.describe().apply(lambda s:s.apply(lambda x:format(x, 'f')))

df_1 = df.drop(df.columns[:2], axis=1, inplace =False)

print("MEDIAN: " + str(df_1.median())), print("\n"), print("MODE: " + str(df_1.mode()))

# Check for the percentage of missing values for the entire dataset
df.isnull().sum()/df.shape[0]*100

from tensorflow.keras.layers import TextVectorization

X = df['comment_text']
y = df[df.columns[2:]].values

# Specifying the number of words in the vocab (aka create the dict space)
MAX_FEATURES = 200000

# Initialize TextVectorization Layer
vectorizer = TextVectorization(max_tokens=MAX_FEATURES,
                              output_sequence_length=1800,
                              output_mode='int')

print(X), print('\n'), print(type(X)), print('\n')

print(X.values), print('\n'), print(type(X.values))

# Teach Vectorizer to adapt the vocabularies 
vectorizer.adapt(X.values)

vectorizer('I think I like to eat Sushi')[:6]

vectorized_text = vectorizer(X.values)

# Create a source of dataset from input data
dataset = tf.data.Dataset.from_tensor_slices((vectorized_text, y))

# Applying the subsequent transformations to preprocess the data
dataset = dataset.cache () # --> will save and remember the old data so that shuffle will keep the first 160000 that has been shuffle in buffer

# Fills the buffer size elements with 16000 
# shuffles will select a random element from only the first 16000 elements in the buffer
dataset = dataset.shuffle(160000)

# Samples per batch --> to ease the process instead of going to a whole dataset. can go to batch per batch instead
dataset = dataset.batch(16)

# To prevent bottleneck, prefetch will take 8 samples out of every batch
dataset = dataset.prefetch(8)

# Inspect the contect of 1 batch
dataset.as_numpy_iterator().next()

batch_X, batch_y = dataset.as_numpy_iterator().next()

# Check for the shape of both X and y
batch_X.shape, batch_y.shape

train = dataset.take(int(len(dataset)*.7))
val = dataset.skip(int(len(dataset)*.7)).take(int(len(dataset)*.2))
test = dataset.skip(int(len(dataset)*.9)).take(int(len(dataset)*.1))

print('The number of batches are: ' + str(len(dataset)))
print('The total number of samples: ' + str(int(len(dataset)*16)))
print('The number samples from the first 70% for training is: ' + str(len(train)))
print('The number of samples from the last 20% for validation is: ' + str(len(val)))

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Bidirectional, Dense, Embedding

def create_model():
  return tf.keras.Sequential(
      [tf.keras.layers.Embedding(MAX_FEATURES+1, 32),
       tf.keras.layers.Bidirectional(LSTM(32, activation='tanh')),
       tf.keras.layers.Dense(128, activation='relu'),
       tf.keras.layers.Dense(256, activation='relu'),
       tf.keras.layers.Dense(128, activation='relu'),
       tf.keras.layers.Dense(6, activation='sigmoid')])

strategy = tf.distribute.TPUStrategy(resolver)

with strategy.scope():
  model = create_model()
  model.compile(loss='BinaryCrossentropy', 
                steps_per_execution = 50, 
                optimizer='Adam') # SGD too is fine but Adam is quite okay

model.summary()

# Fit the model
history = model.fit(train, epochs=1, validation_data=val, verbose=1)

history.history

plt.figure(figsize=(8,5))
pd.DataFrame(history.history).plot()
plt.show();

# Give a sentence for vectorizer to predict
input_text = vectorizer ("OMG you suck! I don't like you")

# Check how it's sequence numerically represented
input_text, input_text[:7], input_text.shape

res = model.predict(np.expand_dims(input_text, 0))

res

(res > 0.5).astype(int)

batch_X, batch_y = test.as_numpy_iterator().next()

(model.predict(batch_X) > 0.5).astype(int)

res.shape

from tensorflow.keras.metrics import Precision, Recall, CategoricalAccuracy

pre = Precision()
re = Recall()
acc = CategoricalAccuracy()

for batch in test.as_numpy_iterator(): 
  # Unpack the batch 
  X_true, y_true = batch
  # Make a prediction 
  yhat = model.predict(X_true)
  
  # Flatten the predictions
  y_true = y_true.flatten()
  yhat = yhat.flatten()
  
  pre.update_state(y_true, yhat)
  re.update_state(y_true, yhat)
  acc.update_state(y_true, yhat)

print(f'Precision: {pre.result().numpy()}, Recall:{re.result().numpy()}, Accuracy:{acc.result()}')

!pip install gradio jinja2

import tensorflow as tf
import gradio as gr

model.save('toxicity.h5')

input_str = vectorizer('what the fuck! fuck you!')

res = model.predict(np.expand_dims(input_str,0))

df.columns[2:]

res

def score_comment(comment):
  vectorized_comment = vectorizer([comment])
  results = model.predict(vectorized_comment)

  text = ''
  for idx, col in enumerate(df.columns[2:]):
    text += '{}: {}\n'.format(col, results[0][idx] > 0.5)

  return text

interface = gr.Interface(fn=score_comment,
                         inputs=gr.inputs.Textbox(lines=2, placeholder='Comment to score'),
                         outputs='text')

interface.launch(share=True)
