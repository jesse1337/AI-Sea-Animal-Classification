# Authors: Daniel Chang and Jesse Ge

# This file contains the main code to develop, train, and test the model.
# It also graphs the results.


import tensorflow as tf
from tensorflow.keras.applications import ResNet50, VGG16, InceptionV3
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, Average
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import BatchNormalization
import matplotlib.pyplot as plt


train_data_dir = 'Data/Train'
test_data_dir = 'Data/Test'

# Prepare image for processing. In this case, we will make all images 128x128
height = 128
width = 128

# Data preprocessing for model training:
#   rescale: Normalize pixels (0-255) to speed up training
#   rotation_range: Rotating image randomly between (-10, 10) degrees for pattern recognition.
#   width_shift_range: Moving image left or right by 10% of width.
#   height_shift_range: Same as above, but moving vertically. 10% as well.
#   shear_range: Changing slant/angle of the image to train the model on different angles.
#   zoom_range: Zooms in or out.
#   horizontal_flip: Flips images randomly. Since images are randomly facing left or right, this doesn't do much but acts as a premeasure.
#   fill_mode: If an image is transformed, this will fill empty pixels that results from the transformation.
model_train = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Data preprocessing for model testing:
# Only need rescaling here for image quality; otherwise everything else is pretty much done in training.
model_test = ImageDataGenerator(rescale=1./255)

# Data generator parameters for training:
#   (training data directory,
#   size of each image,
#   batch size (how many images to process at once),
#   class mode (labels of arrays - 2D arrays)
model_training_data = model_train.flow_from_directory(
    train_data_dir,
    target_size=(height, width),
    batch_size=64,
    class_mode='categorical'
)

# Data generator parameters for testing:
#   (testing data directory,
#   size of each image,
#   batch size (how many images to process at once),
#   class mode (labels of arrays - 2D arrays)
model_testing_data = model_test.flow_from_directory(
    test_data_dir,
    target_size=(height, width),
    batch_size=64,
    class_mode='categorical'
)

# Prepare the three models we will use (ResNet50, VGG16, InceptionV3)
#   For each, we use imagenet database as weight for pre-training
#   Not connecting the model's top layers because we use custom layers
#   Input information for the image (height, width, RGB color)
resnet50 = ResNet50(weights='imagenet', include_top=False,
                    input_tensor=Input(shape=(height, width, 3)))
vgg16 = VGG16(weights='imagenet', include_top=False,
              input_tensor=Input(shape=(height, width, 3)))
inceptionv3 = InceptionV3(weights='imagenet', include_top=False,
                          input_tensor=Input(shape=(height, width, 3)))

# We decided to not use all layers for each model.
# Using last 20 layers.
for i in (resnet50, vgg16, inceptionv3):
    for j in i.layers[:-20]:
        j.trainable = False

# Adjusting the top layers of models:
#   Flatten = turning 2D/XD output into 1D for dense layer input
#   Dense = Layer, Rectified Linear Unit = non-linearity
#   BatchNormalization = Normalizes batch
#   Dropout = Prevents overfitting


def adjust(base_model):
    x = Flatten()(base_model.output)
    x = Dense(1024, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    # Another dense layer that outputs probability distribution for classifying class
    predictions = Dense(len(model_training_data.class_indices),
                        activation='softmax')(x)

    return Model(inputs=base_model.input, outputs=predictions)


# Apply adjustment of layers to each model
resnet_top = adjust(resnet50)
vgg_top = adjust(vgg16)
inception_top = adjust(inceptionv3)

# Input takes height, width, RGB of image
model_input = Input(shape=(height, width, 3))

# For each model, take the model_input of those models.
resnet50_res = resnet_top(model_input)
vgg16_res = vgg_top(model_input)
inceptionv3_res = inception_top(model_input)

# Ensemble for average of those model outputs
ensemble_res = Average()([resnet50_res, vgg16_res, inceptionv3_res])

# Run the ensemble model using the inputs of each model
ensemble_model = Model(inputs=model_input, outputs=ensemble_res)

# Compile it
#   Adam = Adaptive Moment Estimation; optimization algorithm that reduces noise
ensemble_model.compile(optimizer=Adam(learning_rate=0.0001),
                       loss='categorical_crossentropy', metrics=['accuracy'])


# For each epoch, learning rate is ajdusted. Only five epochs instead of 10.
def lr_adjust(epoch, lr):
    if epoch < 5:
        return lr
    else:
        return lr * tf.math.exp(-0.1)


callback_learningrate = LearningRateScheduler(lr_adjust)

# Model Checkpoint
checkpoint = ModelCheckpoint(
    'best_ensemble_model.keras', save_best_only=True, monitor='val_loss', mode='min')

# We train the ensemble model through 5 epochs, using test data as validation, learning rate as callback
history = ensemble_model.fit(
    model_training_data,
    epochs=5,
    validation_data=model_testing_data,
    callbacks=[callback_learningrate, checkpoint]
)

ensemble_model.save('sea_classification.keras')

# Print train and validation accuracies
print("Train Accuracy:", history.history['accuracy'][-1])
print("Test Accuracy:", history.history['val_accuracy'][-1])


print("Done training.\nProceed to graph.")
# ----------------------- GRAPHING


def plot_graph(history):
    epoch = range(1, len(history.history['accuracy']) + 1)

    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epoch, history.history['accuracy'], label='Training Accuracy')
    plt.plot(epoch, history.history['val_accuracy'],
             label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(epoch, history.history['loss'])
    plt.plot(epoch, history.history['val_loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.tight_layout()
    plt.show()


plot_graph(history)
