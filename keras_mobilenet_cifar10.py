!pip install -U tensorboardcolab
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorboardcolab import TensorBoardColab, TensorBoardColabCallback
#tbc=TensorBoardColab()
from timeit import default_timer as timer

# Settings
num_classes = 10
batch_size = 256
epochs = 5

# Data setup
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255 
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

option = 1

# No pretrained weights, default MobileNet
if option == 1: 
    model = MobileNet(weights=None, input_shape=x_train.shape[1:], classes=num_classes)
    
# ImageNet weights, replicated default MobileNet (fully convolutional)
elif option == 2:
    image_input = tf.keras.Input(shape=x_train.shape[1:], name='image_input')
    x = MobileNet(include_top=False)(image_input)
    x = tf.keras.layers.GlobalAveragePooling2D(name='global_average_pooling2d_12')(x)
    x = tf.keras.layers.Reshape((1,1,1024), name='reshape_1')(x)
    x = tf.keras.layers.Dropout(rate=1e-3, name='dropout')(x)
    x = tf.keras.layers.Conv2D(num_classes, 1, name='conv_preds')(x)
    x = tf.keras.layers.Activation('softmax', name='act_softmax')(x)
    x = tf.keras.layers.Reshape((10,), name='reshape_2')(x)  
    model = tf.keras.Model(inputs=image_input, outputs=x)

# ImageNet weights, basic fully connected (dense) top
elif option == 3:
    image_input = tf.keras.Input(shape=x_train.shape[1:], name='image_input')
    x = MobileNet(weights='imagenet', include_top=False)(image_input)
    x = tf.keras.layers.Flatten(name='flatten')(x)
    x = tf.keras.layers.Dense(4096, activation='relu', name='fc1')(x)
    x = tf.keras.layers.Dense(4096, activation='relu', name='fc2')(x)
    x = tf.keras.layers.Dense(num_classes, activation='softmax', name='predictions')(x)
    model = tf.keras.Model(inputs=image_input, outputs=x)

model.summary()

# Select loss, optimizer, metric
model.compile(loss='categorical_crossentropy',
                                                        optimizer=tf.train.AdamOptimizer(0.001),
                                                        metrics=['accuracy'])

# Train
start = timer()
model.fit(x_train, y_train,
                                        batch_size=batch_size,
                                        epochs=epochs,
                                        verbose=1,
                                        validation_data=(x_test, y_test))
# callbacks=[TensorBoardColabCallback(tbc)]

# Evaluate
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
timer()-start
