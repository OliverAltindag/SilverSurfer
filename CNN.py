import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, UpSampling2D, 
    Concatenate, BatchNormalization, Dropout, 
    GlobalAveragePooling2D, Dense, Activation,
    Conv2DTranspose, Add, Multiply
)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2


# helpers for the model, need to be defined BEFORE the model
# may be moved to make this look prettier
def ricker_input_branch(input_tensor, dropout_rate, l2_reg, prefix):
    """
    Processing branch for Ricker wavelet spectrograms (detection)
    """
    # first conv block
    # capture multi-scale patterns
    x = Conv2D(32, (3, 3), padding='same', activation='relu',
               kernel_regularizer=l2(l2_reg), name=f'{prefix}_conv1')(input_tensor)
    x = BatchNormalization(name=f'{prefix}_bn1')(x)
    x = Dropout(dropout_rate, name=f'{prefix}_dropout1')(x)
    
    # second conv block
    # refine scale-time relationships
    x = Conv2D(64, (3, 3), padding='same', activation='relu',
               kernel_regularizer=l2(l2_reg), name=f'{prefix}_conv2')(x)
    x = BatchNormalization(name=f'{prefix}_bn2')(x)
    x = MaxPooling2D((2, 2), name=f'{prefix}_pool1')(x)
    x = Dropout(dropout_rate, name=f'{prefix}_dropout2')(x)
    
    # third conv block
    # deeper feature extraction
    x = Conv2D(64, (3, 3), padding='same', activation='relu',
               kernel_regularizer=l2(l2_reg), name=f'{prefix}_conv3')(x)
    x = BatchNormalization(name=f'{prefix}_bn3')(x)
    x = Dropout(dropout_rate, name=f'{prefix}_dropout3')(x)
    
    return x

def haar_input_branch(input_tensor, dropout_rate, l2_reg, prefix):
    """
    Processing branch for Haar wavelet (timing precision)
    Emphasizes temporal localization over scale complexity
    """
    # Focus on temporal precision with 1x1 convs along time axis
    x = Conv2D(16, (1, 3), padding='same', activation='relu',
               kernel_regularizer=l2(l2_reg), name=f'{prefix}_conv1')(input_tensor)
    x = BatchNormalization(name=f'{prefix}_bn1')(x)
    x = Dropout(dropout_rate, name=f'{prefix}_dropout1')(x)
    
    # Temporal refinement
    x = Conv2D(32, (1, 3), padding='same', activation='relu',
               kernel_regularizer=l2_reg, name=f'{prefix}_conv2')(x)
    x = BatchNormalization(name=f'{prefix}_bn2')(x)
    x = Dropout(dropout_rate, name=f'{prefix}_dropout2')(x)
    
    # Upsample to match Ricker feature map dimensions
    x = UpSampling2D((2, 1), name=f'{prefix}_upsample')(x)  # Scale up in time dimension
    
    # Final conv to match channel dimensions
    x = Conv2D(64, (1, 1), activation='relu',
               kernel_regularizer=l2(l2_reg), name=f'{prefix}_conv3')(x)
    
    # Sigmoid to create attention mask - highlights precise timing locations
    timing_mask = Activation('sigmoid', name=f'{prefix}_timing_attention')(x)
    
    return timing_mask


# here is where the model is defined
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
