from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from os import listdir
from os.path import isfile, join
import tensorflow as tf
from keras.callbacks import TensorBoard

NAME = "CNN p"
img_width, img_height = 150, 150

model = load_model('model.h5')
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

mypath = "predict/"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
print(onlyfiles)
# predicting images
dog_counter = 0 
cat_counter  = 0
for file in onlyfiles:
    img = image.load_img(mypath+file, target_size=(img_width, img_height))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    
    images = np.vstack([x])
    classes = model.predict_classes(images, batch_size=10)
    classes = classes[0][0]
    #tf.summary.histogram(“classes”, classes)
    if classes == 0:
        print(file + ": " + 'cat')
        cat_counter += 1
    else:
        print(file + ": " + 'dog')
        dog_counter += 1

#tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))
print("Total Dogs :",dog_counter)
print("Total Cats :",cat_counter)


