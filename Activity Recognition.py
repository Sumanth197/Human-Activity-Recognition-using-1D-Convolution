from __future__ import print_function
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn import preprocessing

import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape, GlobalAveragePooling1D
from keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D
from keras.utils import np_utils


# In[2]:


file_path = "/home/sumanth/Documents/WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt"
column_names = ['user_id', 'activity', 'timestamp', 'x_axis', 'y_axis', 'z_axis']
df = pd.read_csv(file_path, header=None, names=column_names)

df['z_axis'].replace(regex=True, inplace=True, to_replace=r';', value=r'')

df['z_axis'] = df.z_axis.astype(float)

df.dropna(axis=0, how='any', inplace=True)

df['activity'].value_counts().plot(kind='barh',
                                   title='Training Examples by Activity Type')
plt.show()


# In[4]:


df['user_id'].value_counts().plot(kind='pie',
                                  title='Training Examples by User')
plt.show()


# In[6]:


LABEL = "ActivityEncoded"
le = preprocessing.LabelEncoder()
df[LABEL] = le.fit_transform(df["activity"].values.ravel())

#df


# In[7]:


#scaler = preprocessing.StandardScaler()

names = ['x_axis', 'y_axis', 'z_axis']

df[names] = (df[names] - df[names].mean()) / (df[names].max() - df[names].min())

df_test = df[df['user_id'] > 28]
df_train = df[df['user_id'] <= 28]

#df_train


# In[8]:

def plot_axis(ax, x, y, title):

    ax.plot(x, y)
    ax.set_title(title)
    ax.xaxis.set_visible(False)
    ax.set_ylim([min(y) - np.std(y), max(y) + np.std(y)])
    ax.set_xlim([min(x), max(x)])
    ax.grid(True)


def plot_activity(activity, data):

    fig, (ax0, ax1, ax2) = plt.subplots(nrows=3,
         figsize=(15, 10),
         sharex=True)
    plot_axis(ax0, data['timestamp'], data['x_axis'], 'x_axis')
    plot_axis(ax1, data['timestamp'], data['y_axis'], 'y_axis')
    plot_axis(ax2, data['timestamp'], data['z_axis'], 'z_axis')
    plt.subplots_adjust(hspace=0.2)
    fig.suptitle(activity)
    plt.subplots_adjust(top=0.90)
    plt.show()

for activity in np.unique(df["activity"]):
    subset = df[df["activity"] == activity][:180]
plot_activity(activity, subset)


def create_segments_and_labels(df, time_steps, step, label_name):
    
    N_FEATURES = 3
    segments = []
    labels = []
    for i in range(0, len(df) - time_steps, step):
        xs = df['x_axis'].values[i: i + time_steps]
        ys = df['y_axis'].values[i: i + time_steps]
        zs = df['z_axis'].values[i: i + time_steps]
        # Retrieve the most often used label in this segment
        label = stats.mode(df[label_name][i: i + time_steps])[0][0]
        segments.append([xs, ys, zs])
        labels.append(label)

    # Bring the segments into a better shape
    reshaped_segments = np.asarray(segments, dtype= np.float32).reshape(-1, time_steps, N_FEATURES)
    labels = np.asarray(labels)

    return reshaped_segments, labels

TIME_PERIODS = 80
STEP_DISTANCE = 40
x_train, y_train = create_segments_and_labels(df_train,
                                              TIME_PERIODS,
                                              STEP_DISTANCE,LABEL)


# In[9]:


num_time_periods, num_sensors = x_train.shape[1], x_train.shape[2]
num_classes = le.classes_.size

input_shape = (num_time_periods*num_sensors)
x_train = x_train.reshape(x_train.shape[0], input_shape)

x_train = x_train.astype("float32")
y_train = y_train.astype("float32")

y_train = np_utils.to_categorical(y_train, num_classes)


# In[ ]:
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('acc')>0.99):
      print("\nReached 99% accuracy so cancelling training!")
      self.model.stop_training = True

callbacks = myCallback()


model_m = Sequential()
model_m.add(Reshape((TIME_PERIODS, num_sensors), input_shape=(input_shape,)))
model_m.add(Conv1D(100, 10, activation='relu', input_shape=(TIME_PERIODS, num_sensors)))
model_m.add(Conv1D(100, 10, activation='relu'))
model_m.add(MaxPooling1D(3))
model_m.add(Conv1D(160, 10, activation='relu'))
model_m.add(Conv1D(160, 10, activation='relu'))
model_m.add(GlobalAveragePooling1D())
model_m.add(Dropout(0.5))
model_m.add(Dense(num_classes, activation='softmax'))
model_m.compile(loss='categorical_crossentropy',
                optimizer='adam', metrics=['accuracy'])

BATCH_SIZE = 1024
EPOCHS = 50
history = model_m.fit(x_train,
                      y_train,
                      batch_size=BATCH_SIZE,
                      epochs=EPOCHS,
                      callbacks=[callbacks],
                      validation_split=0.2,
                      verbose=1)



# Test Cases

LABELS = ["Downstairs",
          "Jogging",
          "Sitting",
          "Standing",
          "Upstairs",
"Walking"]


x_test, y_test = create_segments_and_labels(df_test,
                                            TIME_PERIODS,
                                            STEP_DISTANCE,
					LABEL)

x_test = x_test.reshape(x_test.shape[0], input_shape)

x_test = x_test.astype("float32")
y_test = y_test.astype("float32")

y_test = np_utils.to_categorical(y_test, num_classes)

score = model_m.evaluate(x_test, y_test, verbose=1)

print("\nAccuracy on test data: %0.2f" % score[1])
print("\nLoss on test data: %0.2f" % score[0])


y_pred_test = model_m.predict(x_test)
# Take the class with the highest probability from the test predictions
max_y_pred_test = np.argmax(y_pred_test, axis=1)
max_y_test = np.argmax(y_test, axis=1)

matrix = metrics.confusion_matrix(max_y_test, max_y_pred_test)
plt.figure(figsize=(6, 4))
sns.heatmap(matrix,
            cmap="coolwarm",
            linecolor='white',
            linewidths=1,
            xticklabels=LABELS,
            yticklabels=LABELS,
            annot=True,
            fmt="d")
plt.title("Confusion Matrix")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.show()


print("\n--- Classification report for test data ---\n")

print(classification_report(max_y_test, max_y_pred_test))





