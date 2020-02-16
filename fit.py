
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os
LABEL = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
img_width, img_height = 224, 224
model = load_model('m.hdf5')
for i in os.listdir('test'):
    test_image = image.load_img(f'test/{i}', target_size = (img_width, img_height))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    predict = model.predict(test_image)
    predicted_class = np.argmax(predict[0])
    print(i)
    for x in predict[0]:
        print("{:1.3f}".format(x))
    print(LABEL[ predicted_class])
    print('\n')


# Model's prediction


