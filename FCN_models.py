import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Add, Activation, Dropout, BatchNormalization, LayerNormalization

def fcn(input_size=(128,128,3), output_channels=3):

    vgg = tf.keras.applications.vgg19.VGG19(weights='imagenet', include_top=False, input_shape=input_size)

    pool3 = vgg.get_layer('block3_pool').output  
    pool4 = vgg.get_layer('block4_pool').output  
    pool5 = vgg.get_layer('block5_pool').output  

    score_pool5 = Conv2D(output_channels, (1,1), padding='same')(pool5)

    upscore_pool5 = Conv2DTranspose(output_channels, kernel_size=4, strides=2, padding='same')(score_pool5)

    score_pool4 = Conv2D(output_channels, (1,1), padding='same')(pool4)


    fuse_pool4 = Add(name='fuse_pool4')([upscore_pool5, score_pool4])

    upscore_pool4 = Conv2DTranspose(output_channels, kernel_size=4, strides=2, padding='same')(fuse_pool4)

    score_pool3 = Conv2D(output_channels, (1,1), padding='same')(pool3)

    fuse_pool3 = Add(name='fuse_pool3')([upscore_pool4, score_pool3])

    upscore_final = Conv2DTranspose(output_channels, kernel_size=16, strides=8, padding='same')(fuse_pool3)

    outputs = Activation('softmax')(upscore_final)

    model = Model(inputs=vgg.input, outputs=outputs)

    return model

def fcn_batch(input_size=(128,128,3), output_channels=3):

    vgg = tf.keras.applications.vgg19.VGG19(weights='imagenet', include_top=False, input_shape=input_size)

    pool3 = vgg.get_layer('block3_pool').output  
    pool4 = vgg.get_layer('block4_pool').output 
    pool5 = vgg.get_layer('block5_pool').output  

    score_pool5 = Conv2D(output_channels, (1,1), padding='same')(pool5)
    score_pool5 = BatchNormalization()(score_pool5)
    score_pool5 = Dropout(0.1)(score_pool5)

    upscore_pool5 = Conv2DTranspose(output_channels, kernel_size=4, strides=2, padding='same')(score_pool5)
    upscore_pool5 = BatchNormalization()(upscore_pool5)
    upscore_pool5 = Dropout(0.1)(upscore_pool5)

    score_pool4 = Conv2D(output_channels, (1,1), padding='same')(pool4)
    score_pool4 = BatchNormalization()(score_pool4)
    score_pool4 = Dropout(0.1)(score_pool4)

    fuse_pool4 = Add(name='fuse_pool4')([upscore_pool5, score_pool4])

    upscore_pool4 = Conv2DTranspose(output_channels, kernel_size=4, strides=2, padding='same')(fuse_pool4)
    upscore_pool4 = BatchNormalization()(upscore_pool4)
    upscore_pool4 = Dropout(0.1)(upscore_pool4)

    score_pool3 = Conv2D(output_channels, (1,1), padding='same')(pool3)
    score_pool3 = BatchNormalization()(score_pool3)
    score_pool3 = Dropout(0.1)(score_pool3)

    fuse_pool3 = Add()([upscore_pool4, score_pool3])

    upscore_final = Conv2DTranspose(output_channels, kernel_size=16, strides=8, padding='same')(fuse_pool3)
    upscore_final = BatchNormalization()(upscore_final)
    upscore_final = Dropout(0.1)(upscore_final)

    outputs = Activation('softmax')(upscore_final)

    model = Model(inputs=vgg.input, outputs=outputs)

    return model

def fcn_norm(input_size=(128,128,3), output_channels=3):

    vgg = tf.keras.applications.vgg19.VGG19(weights='imagenet', include_top=False, input_shape=input_size)

    # Extract the layers we need ( pool3, pool4, and pool5)
    pool3 = vgg.get_layer('block3_pool').output  
    pool4 = vgg.get_layer('block4_pool').output  
    pool5 = vgg.get_layer('block5_pool').output  

    score_pool5 = Conv2D(output_channels, (1,1), padding='same')(pool5)
    score_pool5 = LayerNormalization(epsilon=1e-6)(score_pool5)
    score_pool5 = Dropout(0.1)(score_pool5)

    upscore_pool5 = Conv2DTranspose(output_channels, kernel_size=4, strides=2, padding='same')(score_pool5)
    upscore_pool5 = LayerNormalization(epsilon=1e-6)(upscore_pool5)
    upscore_pool5 = Dropout(0.1)(upscore_pool5)

    score_pool4 = Conv2D(output_channels, (1,1), padding='same')(pool4)
    score_pool4 = LayerNormalization(epsilon=1e-6)(score_pool4)
    score_pool4 = Dropout(0.1)(score_pool4)

    fuse_pool4 = Add(name='fuse_pool4')([upscore_pool5, score_pool4])

    upscore_pool4 = Conv2DTranspose(output_channels, kernel_size=4, strides=2, padding='same')(fuse_pool4)
    upscore_pool4 = LayerNormalization(epsilon=1e-6)(upscore_pool4)
    upscore_pool4 = Dropout(0.1)(upscore_pool4)


    score_pool3 = Conv2D(output_channels, (1,1), padding='same')(pool3)
    score_pool3 = LayerNormalization(epsilon=1e-6)(score_pool3)
    score_pool3 = Dropout(0.1)(score_pool3)

    fuse_pool3 = Add()([upscore_pool4, score_pool3])

    upscore_final = Conv2DTranspose(output_channels, kernel_size=16, strides=8, padding='same')(fuse_pool3)
    upscore_final = LayerNormalization(epsilon=1e-6)(upscore_final)
    upscore_final = Dropout(0.1)(upscore_final)

    outputs = Activation('softmax')(upscore_final)

    model = Model(inputs=vgg.input, outputs=outputs)

    return model