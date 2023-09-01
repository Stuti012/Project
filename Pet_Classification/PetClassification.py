#!/usr/bin/env python
# coding: utf-8

# # Pet Classification using CNN

# #####Mounting the drive

# In[50]:


from google.colab import drive
drive.mount('/content/gdrive')


# #####Importing libraries

# In[ ]:


from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.regularizers import l2


# ###Designing CNN

# In[ ]:


classifier = Sequential()
classifier.add(Conv2D(32, 5, 5, input_shape = (512, 512, 3), activation = 'relu',kernel_regularizer=l2(l2=0.001)))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Dropout(0.4))
classifier.add(Conv2D(64, 5, 5, activation = 'relu',kernel_regularizer=l2(l2=0.001)))
#classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Dropout(0.4))
classifier.add(Flatten())


# #####Full connection/ANN part

# In[ ]:


classifier.add(Dense(32, activation = 'relu'))
classifier.add(Dropout(0.4))
classifier.add(Dense(1, activation = 'sigmoid'))


# In[ ]:


classifier.summary()


# In[ ]:


classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy', Precision(), Recall()])


# #####Data Augmentation

# In[ ]:


# To generate more images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2, # Rotate img anticlockwise
                                   zoom_range = 0.2,
                                   horizontal_flip = True)


# In[ ]:


test_datagen = ImageDataGenerator(rescale = 1./255)


# In[ ]:


training_set = train_datagen.flow_from_directory('/content/gdrive/MyDrive/Pet_Classification/data/train',
                                                 target_size = (512, 512),
                                                 batch_size = 8,
                                                 class_mode = 'binary')


# In[ ]:


test_set = test_datagen.flow_from_directory('/content/gdrive/MyDrive/Pet_Classification/data/test',
                                            target_size = (512, 512),
                                            batch_size = 8,
                                            class_mode = 'binary')


# In[ ]:


get_ipython().system(' pip install livelossplot')


# In[ ]:


from livelossplot import PlotLossesKerasTF


# In[ ]:


classifier.fit(training_set,epochs=300,validation_data = test_set,callbacks=[PlotLossesKerasTF()])


# In[ ]:


import numpy as np
from keras.preprocessing import image
from tensorflow.keras.preprocessing import image

test_image = image.load_img('/content/gdrive/MyDrive/Pet_Classification/data/test/dogs/107.jpg',target_size=(512,512))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image,axis=0)
result = classifier.predict(test_image)
training_set.class_indices

if result[0][0]<=0.5:
    prediction= 'cat'
    print('Result is',result[0][0])
else:
    prediction = 'dog'
    print('Result is',result[0][0])

print(prediction)


# In[ ]:


import numpy as np
from keras.preprocessing import image
from tensorflow.keras.preprocessing import image

test_image = image.load_img('/content/gdrive/MyDrive/Pet_Classification/data/test/cats/104.jpg',target_size=(512,512))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image,axis=0)
result = classifier.predict(test_image)
training_set.class_indices

if result[0][0]<=0.5:
    prediction= 'cat'
    print('Result is',result[0][0])
else:
    prediction = 'dog'
    print('Result is',result[0][0])

print(prediction)


# In[ ]:


classifier.save('Classification.h5')


# In[ ]:


from keras.models import load_model
loaded_model = load_model("Classification.h5")


# In[ ]:


test_image = image.load_img('/content/gdrive/MyDrive/Pet_Classification/data/test/cats/108.jpg',target_size=(512,512))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image,axis=0)
result = loaded_model.predict(test_image)
training_set.class_indices

if result[0][0]<=0.5:
    prediction= 'cat'
    print('Result is',result[0][0])
else:
    prediction = 'dog'
    print('Result is',result[0][0])

print(prediction)


# In[ ]:


from keras.models import model_from_json
model_json = classifier.to_json()
with open("Classification.json", "w") as json_file:
  json_file.write(model_json)

classifier.save_weights("Classification.h5")

json_file = open("Classification.json", 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights("Classification.h5")


# In[ ]:


from google.colab import files
files.download('Classification.h5')
files.download('Classification.json')


# In[ ]:


import shutil
classifier.save('Classification.h5')

destination_path = '/content/gdrive/MyDrive/Pet_Classification/Classification.h5'

shutil.copyfile('Classification.h5', destination_path)


# In[51]:


from keras.models import load_model

destination_path = '/content/gdrive/MyDrive/Pet_Classification/Classification.h5'

loaded_model = load_model(destination_path)


# In[52]:


import numpy as np
from keras.preprocessing import image
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from tensorflow import io

img = image.load_img('/content/gdrive/MyDrive/Pet_Classification/data/test/dogs/103.jpg',target_size=(512,512))
test_image = image.img_to_array(img)
test_image = np.expand_dims(test_image,axis=0)
result = loaded_model.predict(test_image)

if result[0][0]<=0.5:
    prediction= 'cat'
    print('Result is',result[0][0])
else:
    prediction = 'dog'
    print('Result is',result[0][0])

print(prediction)


plt.imshow(img)
plt.axis('off')
plt.show()


# In[53]:


img = image.load_img('/content/gdrive/MyDrive/Pet_Classification/data/test/cats/103.jpg',target_size=(512,512))
test_image = image.img_to_array(img)
test_image = np.expand_dims(test_image,axis=0)
result = loaded_model.predict(test_image)

if result[0][0]<=0.5:
    prediction= 'cat'
    print('Result is',result[0][0])
else:
    prediction = 'dog'
    print('Result is',result[0][0])

print(prediction)


plt.imshow(img)
plt.axis('off')
plt.show()


# In[ ]:




