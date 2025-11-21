import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, UpSampling2D, 
    Concatenate, BatchNormalization, Dropout, 
    GlobalAveragePooling2D, Dense, Activation,
    Conv2DTranspose, Add, Multiply
)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2


def create_multi_input_switchback_cnn(
    ricker_shape,  # (n_scales, time_length, 1) - for both Br and Vr
    haar_shape,    # (time_length, 1) - but will be expanded to 2D
    dropout_rate=0.3,
    l2_reg=1e-4
):
    """
    Creates a multi-input CNN for magnetic switchback detection
    
    Architecture:
    - Two Ricker channels: detect presence of events across scales
    - One Haar channel: provides precise temporal localization
    - Fuses information for final detection and timing
    """
    
    # input layers
    # litteraly just inputs them
    ricker_br_input = Input(shape=ricker_shape, name='ricker_br_input')
    ricker_vr_input = Input(shape=ricker_shape, name='ricker_vr_input')
    haar_br_input = Input(shape=haar_shape, name='haar_br_input')
    
    # expand Haar to 2D for CNN processing (match Ricker dimensions)
    # use lambda layer to expand: (time_len, 1) -> (time_len, 1, 1) -> (n_scales, time_len, 1)
    # replicate along scale dimension to match Ricker shape
    haar_expanded = tf.expand_dims(haar_br_input, axis=1)  # Add scale dimension
    haar_expanded = tf.repeat(haar_expanded, repeats=ricker_shape[0], axis=1)  # Repeat to match scales
    
    # feature extraction branch for Ricker Br
    ricker_br_features = ricker_input_branch(ricker_br_input, dropout_rate, l2_reg, 'ricker_br')
    
    # feature extraction branch for Ricker Vr  
    ricker_vr_features = ricker_input_branch(ricker_vr_input, dropout_rate, l2_reg, 'ricker_vr')
    
    # feature extraction branch for Haar Br (temporal precision usage only)
    haar_features = haar_input_branch(haar_expanded, dropout_rate, l2_reg, 'haar_br')
    
    # cross-channel attention
    ## Ricker channels detect, Haar provides timing
    # Combine Ricker features (event detection) with Haar features (timing)
    combined_ricker = Add(name='combined_ricker_detection')([ricker_br_features, ricker_vr_features])
    
    # apply attention from Haar timing to Ricker detection
    timing_attention = Multiply(name='timing_attention')([
        combined_ricker, 
        haar_features
    ])
    
    # final processing layers
    x = Conv2D(128, (3, 3), padding='same', activation='relu', 
               kernel_regularizer=l2(l2_reg), name='final_conv1')(timing_attention)
    x = BatchNormalization(name='final_bn1')(x)
    x = Dropout(dropout_rate, name='final_dropout1')(x)
    
    x = Conv2D(64, (3, 3), padding='same', activation='relu',
               kernel_regularizer=l2(l2_reg), name='final_conv2')(x)
    x = BatchNormalization(name='final_bn2')(x)
    x = Dropout(dropout_rate, name='final_dropout2')(x)
    
    # output layer for switchback detection
    # use sigmoid for binary detection at each time step
    detection_output = Conv2D(1, (1, 1), activation='sigmoid', 
                             name='switchback_detection')(x)
    
    # global detection output (for overall event presence)
    global_detection = GlobalAveragePooling2D(name='global_detection_pool')(detection_output)
    global_output = Dense(1, activation='sigmoid', name='global_switchback_presence')(global_detection)
    
    # create the model
    model = Model(
        inputs=[ricker_br_input, ricker_vr_input, haar_br_input],
        outputs=[detection_output, global_output],
        name='multi_input_switchback_cnn'
    )
    
    return model
