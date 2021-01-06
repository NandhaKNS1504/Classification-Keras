import numpy as np
import pandas as pd
from keras.preprocessing.image import image
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import load_model
#from keras.models import Sequential
#from keras import preprocess_input

# test the model
img_path = &#39;pear.jpg&#39;
label = [&#39;Guava&#39;,&#39;Pear&#39;,&#39;Jack&#39;]
img = image.load_img(img_path, target_size=(64,64))
#img.show()
x = image.img_to_array(img)
#print(x)
x = np.expand_dims(x, axis=0)
#x = preprocess_input(x)

# load the model and label binarizer
print(&quot;[INFO] loading network and VGG16...&quot;)
model = load_model(&quot;Best-weights-my_model.hdf5&quot;)
features = model.predict(x)
#print(type(features))
print(&quot;features=&quot;,features)
ind = np.where(features == 1)[1][0]
#ind=int(features)
print(&quot;index=&quot;,ind)
print(&#39;Predicted Array:&#39;,features)
print(&#39;Predicted Label:&#39;,label[ind])
