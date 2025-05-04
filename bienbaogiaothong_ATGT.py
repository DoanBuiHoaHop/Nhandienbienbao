# ********* Libraries *******
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
from sklearn.model_selection import train_test_split
import os
import pandas as pd
import random
from sklearn.utils import shuffle
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import seaborn as sn
from sklearn.metrics import confusion_matrix
from tensorflow import keras

# ********* Load training data *********
path = "./Data/Train"
labelFile = './Data/Train.csv'
count = 0
images = []
label = []
classes_list = os.listdir(path)
print("Total Classes Detected:", len(classes_list))
noOfClasses = len(classes_list)
print("Importing Classes.....")
for x in range(0, len(classes_list)):
    imglist = os.listdir(path + "/" + str(count))
    for y in imglist:
        img = cv2.imread(path + "/" + str(count) + "/" + y)
        img = cv2.resize(img, (32, 32))
        images.append(img)
        label.append(count)
    print(count, end=" ")
    count += 1
print(" ")

images = np.array(images)
classNo = np.array(label)

# Shuffle and split data
images, classNo = shuffle(images, classNo)
X_train, X_val, Y_train, Y_val = train_test_split(images, classNo, test_size=0.2, random_state=42)
print("Train dataset:", X_train.shape, Y_train.shape)
print("Valid dataset:", X_val.shape, Y_val.shape)

# ********* Load test data safely *********
df_test = pd.read_csv('./Data/Test.csv')
Y_test = df_test['ClassId'].values
test_images = df_test["Path"].values
data_test = []
for img_path in test_images:
    full_path = os.path.join("./Data", img_path.replace("\\", "/"))
    if os.path.exists(full_path):
        image = Image.open(full_path)
        image = image.resize((32, 32))
        data_test.append(np.array(image))
    else:
        print(f"File not found: {full_path}")
X_test = np.array(data_test).reshape(-1, 32, 32, 3)

# ********* Visualization *********
data = pd.read_csv(labelFile)
num_of_samples = []
cols = 3
fig, axs = plt.subplots(nrows=noOfClasses, ncols=cols, figsize=(30, 300))
fig.tight_layout()
for i in range(cols):
    for j, row in data.iterrows():
        x_selected = X_train[Y_train == j]
        if len(x_selected) == 0:
            continue
        axs[j][i].imshow(x_selected[random.randint(0, len(x_selected) - 1), :, :], cmap=plt.get_cmap("gray"))
        axs[j][i].axis("off")
        if i == 2:
            axs[j][i].set_title(str(j) + " - " + str(row["ClassId"]))
            num_of_samples.append(len(x_selected))

plt.figure(figsize=(7, 4))
plt.bar(range(0, noOfClasses), num_of_samples)
plt.title("Distribution of the training dataset")
plt.xlabel("Class number")
plt.ylabel("Number of images")
plt.show()

# ********* Preprocessing *********
def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def equalize(img):
    return cv2.equalizeHist(img)

def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    return img / 255

X_train = np.array(list(map(preprocessing, X_train))).reshape(-1, 32, 32, 1)
X_val = np.array(list(map(preprocessing, X_val))).reshape(-1, 32, 32, 1)
X_test = np.array(list(map(preprocessing, X_test))).reshape(-1, 32, 32, 1)

# ********* Augmentation *********
aug = ImageDataGenerator(width_shift_range=0.1,
                         height_shift_range=0.1,
                         zoom_range=0.2,
                         shear_range=0.1,
                         rotation_range=10)
aug.fit(X_train)

Y_train = to_categorical(Y_train, noOfClasses)
Y_val = to_categorical(Y_val, noOfClasses)

# ********* Model *********
def create_model():
    model = Sequential()
    model.add(Input(shape=(32, 32, 1)))
    model.add(Conv2D(60, (5, 5), activation='relu'))
    model.add(Conv2D(60, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(30, (3, 3), activation='relu'))
    model.add(Conv2D(30, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(noOfClasses, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# ********* Train *********
model = create_model()
print("Bắt đầu huấn luyện...")
history = model.fit(aug.flow(X_train, Y_train, batch_size=30),
                    steps_per_epoch=500,
                    epochs=15,
                    validation_data=(X_val, Y_val),
                    shuffle=True)

# ********* Plot *********
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('Loss')
plt.xlabel('Epoch')

plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'validation'])
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.show()

# ********* Save model *********
model.save('bienbaogiaothong.h5')

# ********* Evaluate *********
model = keras.models.load_model('bienbaogiaothong.h5')
Y_pred = model.predict(X_test)
Y_pred = np.argmax(Y_pred, axis=1)

print('Test Accuracy:', accuracy_score(Y_test, Y_pred) * 100)
print('Test Precision:', precision_score(Y_test, Y_pred, average='macro') * 100)
print('Test Recall:', recall_score(Y_test, Y_pred, average='macro') * 100)
print('Test F1-micro:', f1_score(Y_test, Y_pred, average='micro') * 100)
print('Test F1-macro:', f1_score(Y_test, Y_pred, average='macro') * 100)

# ********* Confusion Matrix *********
cm = confusion_matrix(Y_test, Y_pred)
plt.figure(figsize=(10, 7))
sn.heatmap(cm, annot=True, fmt='g', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('confusionmatrix.png', dpi=300, bbox_inches='tight')
