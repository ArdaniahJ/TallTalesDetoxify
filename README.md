<h1 align="center">👮🏻‍♀️ Tall Tales Detoxify aka Toxic Comments Classification</h1>


Tall Tales?
Why do I called it tall tales, a modern one even?

Threat of abuse and harrassment has started even in traditional tall tales until now especially when modernization has taken the world. If it continues, people will stop expressing themselves and giving up seeking different opinions. This shall be the first step to make a better world thus making a modern tall tales that consisted only a good tales for the future generation. Call it a pipedream but I did try my best to stop harmful online content by interpreting visual concent in context thus making the world a better place for now and future generation. 

# Project Description
Trained models & code to predict toxic comments using below modules from TensorFlow and Keras;
1. ```Text Vectorization```
2. ```Sequential```
3. ```LSTM, Dropout, Bidirectional, Dense, Embedding```

# How to train the model with TPU on Google Colab
The TPU are usually on Cloud TPU worker where these are different from local process running the user program in Python. As there is need some initialization work which is to be done to connect to the remote cluster an the initiliaze the TPU. ```TPUClusterResolver``` is a special address just for Google Colab. 
1. Required a resolver object for TPU initialization in order to connect with Cloud TPU
```
resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='')
tf.config.experimental_connect_to_cluster(resolver)

# This is the TPU initialization code that has to be at the beginning.
tf.tpu.experimental.initialize_tpu_system(resolver)

# Check if TPU is connected to notebook
print("All devices: ", tf.config.list_logical_devices('TPU'))
```

Distribution strategies were introduced as a part of TensorFlow2 to help distribute training across TPUs with minimum code changes specifically creating a model under TPUStrategy. It will place the model in a replicated (same weights on each of the cores) manner on the TPU and will keep the replica weights in sync by adding appropriate collective communications (all by reducing the gradients). Below are the steps to perform the distribution. 

2. Connect resolver to NN through TPUStrategy object
```strategy = tf.distribute.TPUStrategy(resolver)```

Afterwards, make use of customized model creation by calling ```strategy.scope():``` ;
Inside ```strategy.scope():``` put:
a. model creation
b. instantiation of the metrics
c. compilation of the model

Below is the example of it; 
```
with strategy.scope():
  model = create_model()
  model.compile(loss='BinaryCrossentropy', 
                steps_per_execution = 50, 
                optimizer='Adam')
```

# How to check if TensorFlow is connected GPU (local machine Jupyter Notebook) before training
1. As of TensorFlow 2.1, ```tf.test.gpu_device_name()``` has been deprecated. 😔 
Then, in the terminal/cmd use ```nvidia-smi``` to check how much GPU memory has been alloted
  * at the same time, using ```watch -n K nvidia-smi``` would tell for every K seconds, how much memory the local machine currenly used (use ```K = 1``` for real-time)
  * to see constantly how much GPU memory used; use ```nvidia-smi -1 10```
  * If there're multiple GPUs in the local achine, multiple networks might needs to be used, each one on a separated GPU;
  
  ```with tf.device('GPU:0'):
          neural_network_1 = initialize_network_1 ()
     with tf.device('/GPU:1'):
          nueral_network_2 = initialize_network_2 ()
  ```

2. Alternatively; use ```tf.debuggin.set_log_device_placement(True)```
An output like this should appear; 
```Executing op_EagerConst in device /job:localhost/replica:0/task:0/device:GPU:0```

# Limitations and ethical considerations for future 
If words that are associated with swearing, insults or profanity are present in a comment, it is likely that it'll be classified as toxic, regardless of the tone or the intent of the author (eg; humorous/self-deprecating). This could present some biases towards the already vulnerable minority groups. 

The intended use of this library is for research purposes, fine-tuning on carefully constructured datasets that refelcted real world demographics and/or to aid content moderators in flagging out harmful content quicker. 
