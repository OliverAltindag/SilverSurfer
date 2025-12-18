import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, UpSampling2D, 
    Concatenate, BatchNormalization, Dropout, 
    GlobalAveragePooling2D, GlobalAveragePooling1D, GlobalMaxPooling1D, 
    Dense, Activation,Conv2DTranspose, Add, Multiply, Reshape
)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.metrics import BinaryIoU, Recall, Precision
import tensorflow.keras.backend as K

# custum focal loss definiton
# the model was crappily guessing bc of the varying background levels
# this helped to isolate the events
def binary_focal_loss(gamma=2., alpha=0.25):
    """
    Returns
    -------
    function
        A loss function that returns the mean scalar loss 
        over the batch.
    """
    def focal_loss_fixed(y_true, y_pred):
        # the focal loss model
        # want to calculate the loss for the Switchbacks
        # y_true is 1, keep the prediction (y_pred)
        # If y_true is 0, replace it with 1.0
        # effectively zeroes out this term for the background pixels
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        # basically the opposite
        # effectively zeroes out this term for the switchback pixels
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        # missed events and false flags
        loss = -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1 + K.epsilon())) \
               -K.sum((1-alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0 + K.epsilon()))
        return loss / tf.cast(tf.size(y_true), tf.float32)
    return focal_loss_fixed

# Dice Loss definition
def dice_loss(y_true, y_pred):
    """
    Returns
    -------
    tf.Tensor
        Scalar loss value ranging from 0 (perfect overlap) to 1 (no overlap).
    """
    # smooths so it doesnt nan out
    smooth = 1e-6
    # convert the 2D time/batch map into a single 1D vector
    # only cares about statistical overlap
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    # multiply by ground probability
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    # bc they minimize need to do this
    return 1. - score

# Combined Loss
def combined_loss(alpha=0.25, gamma=2.0):
    """
    Returns
    -------
    function
        A loss function accepting (y_true, y_pred) that returns the scalar sum 
        of the two losses.
    """
    def loss_fn(y_true, y_pred):
        # the focal loss
        focal = binary_focal_loss(gamma=gamma, alpha=alpha)(y_true, y_pred)
        # dice loss
        dice = dice_loss(y_true, y_pred)

        # adds them together
        return focal + dice
    return loss_fn

# Ricker branch with skip connection
def ricker_input_branch(input_tensor, dropout_rate, l2_reg, prefix):
    """
    Returns:
    - x: The downsampled/processed features (for main path)
    - x_skip: The high-res features before pooling (for skip connection)
    """
    # high-res block (Time = T)
    x = Conv2D(32, (3, 3), padding='same', activation='relu',
               kernel_regularizer=l2(l2_reg), name=f'{prefix}_conv1')(input_tensor)
    x = BatchNormalization(name=f'{prefix}_bn1')(x)
    x_skip = Dropout(dropout_rate, name=f'{prefix}_dropout1')(x) # SAVE THIS SKIP, if not y even do it?
    
    # refinement
    x = Conv2D(64, (3, 3), padding='same', activation='relu',
               kernel_regularizer=l2(l2_reg), name=f'{prefix}_conv2')(x_skip)
    x = BatchNormalization(name=f'{prefix}_bn2')(x)
    
    # pooling (Time = T/2, Scale = S/2)
    x = MaxPooling2D((2, 2), name=f'{prefix}_pool1')(x)
    x = Dropout(dropout_rate, name=f'{prefix}_dropout2')(x)
    
    # low-res block
    x = Conv2D(64, (3, 3), padding='same', activation='relu',
               kernel_regularizer=l2(l2_reg), name=f'{prefix}_conv3')(x)
    x = BatchNormalization(name=f'{prefix}_bn3')(x)
    x = Dropout(dropout_rate, name=f'{prefix}_dropout3')(x)
    
    return x, x_skip

def haar_input_branch(input_tensor, dropout_rate, l2_reg, prefix):
    """
    Returns
    -------
    tf.Tensor
        A temporal attention mask of shape (Batch, 1, Time/2, 64) with values in [0, 1].
    """
    # Haar branch acting as Attention mask
    x = Conv2D(16, (1, 3), padding='same', activation='relu',
               kernel_regularizer=l2(l2_reg), name=f'{prefix}_conv1')(input_tensor)
    x = BatchNormalization(name=f'{prefix}_bn1')(x)
    x = Dropout(dropout_rate, name=f'{prefix}_dropout1')(x)
    
    # downsample Time to T/2 to match Ricker
    x = Conv2D(32, (1, 3), padding='same', strides=(1, 2), activation='relu',
               kernel_regularizer=l2(l2_reg), name=f'{prefix}_conv2')(x)
    x = BatchNormalization(name=f'{prefix}_bn2')(x)
    x = Dropout(dropout_rate, name=f'{prefix}_dropout2')(x)

    # mask generation
    x = Conv2D(64, (1, 1), activation='linear', 
               kernel_regularizer=l2(l2_reg), name=f'{prefix}_conv3')(x)
    # the gate itself
    timing_mask = Activation('sigmoid', name=f'{prefix}_timing_attention')(x)
    
    return timing_mask

def create_multi_input_switchback_cnn(
    ricker_shape,   # (n_scales, time_length, 1)
    haar_shape,     # (time_length, 1)
    dropout_rate=0.3,
    l2_reg=1e-4
):
    """
    U-Net Multi-Input CNN with Gated Attention (Multiply)
    Other methods were rather shite so improvised a bit.
    """
    
    # inputs
    ricker_br_input = Input(shape=ricker_shape, name='ricker_br_input')
    ricker_vr_input = Input(shape=ricker_shape, name='ricker_vr_input')
    haar_br_input = Input(shape=haar_shape, name='haar_br_input')
    
    haar_reshaped = Reshape((1, -1, 1))(haar_br_input)
    
    # Feature Extraction (with Skips)
    ricker_br_features, br_skip = ricker_input_branch(ricker_br_input, dropout_rate, l2_reg, 'ricker_br')
    ricker_vr_features, vr_skip = ricker_input_branch(ricker_vr_input, dropout_rate, l2_reg, 'ricker_vr')
    
    haar_features = haar_input_branch(haar_reshaped, dropout_rate, l2_reg, 'haar_br')
    # Note: haar_features has 64 filters (channels) from the last Conv2D in haar_input_branch
    
    # BOTTLENECK
    # bang
    combined_ricker = Add(name='combined_ricker_detection')([ricker_br_features, ricker_vr_features])
    
    # upsample "Timing Attention" to match scales dimension
    # ricker_features: (Batch, Scales/2, Time/2, Filters)
    # haar_features:   (Batch, 1, Time/2, Filters)
    # moving on
    scale_downsampled = ricker_shape[0] // 2 
    haar_features_broadcasted = UpSampling2D(size=(scale_downsampled, 1), name='haar_broadcast')(haar_features)
    
    # need matching channels for Multiply
    # combined_ricker (Batch, Scales/2, Time/2, 64)
    # haar_features_broadcasted (Batch, Scales/2, Time/2, 64)
    # (assuming haar_input_branch ends with 64 filters, which it does)
    
    # apply attention via Multiplication, bc it fails any other way 
    # and the whole point of the Haar is to do this
    # this suppresses ricker features where timing_mask is near 0
    # edge detector
    timing_attention = Multiply(name='attention_gate')([combined_ricker, haar_features_broadcasted])
    
    # processing
    x = Conv2D(128, (3, 3), padding='same', activation='relu', 
               kernel_regularizer=l2(l2_reg), name='bottleneck_conv1')(timing_attention)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)

    # expand to 128 filters
    # builds the relationship
    x = Conv2D(64, (3, 3), padding='same', activation='relu',
               kernel_regularizer=l2(l2_reg), name='bottleneck_conv2')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    
    # fix scales in bottleneck (Scales/2 -> 1)
    # (1, T/2, 64)
    x = Conv2D(64, (scale_downsampled, 1), padding='valid', activation='relu', name='squash_scales_bottleneck')(x)
    
    # DECODER with SKIP
    # one of the newer additions
    # upsample time (1, T/2) -> (1, T)
    x_up = UpSampling2D(size=(1, 2), name='upsample_main')(x)
    
    # process kkip connection
    # combine Br and Vr skips
    skip_combined = Add(name='combined_skip')([br_skip, vr_skip]) # Shape: (Scales, T, 32)
    
    # The skip has full scales
    # squash them to 1 to match the main branch.
    # (Scales, 1) kernel -> (1, T, 32)
    skip_processed = Conv2D(32, (ricker_shape[0], 1), padding='valid', activation='relu', name='squash_scales_skip')(skip_combined)
    
    # Concatenate
    # Main: (1, T, 64), Skip: (1, T, 32) -> Result: (1, T, 96)
    x_fused = Concatenate(axis=-1, name='unet_skip_connection')([x_up, skip_processed])
    
    # Decoder Convs, these are the final ones
    x = Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(l2_reg), name='decoder_conv1')(x_fused)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)

    # 32 filters
    # another convolution block
    # prepping for single channel
    x = Conv2D(32, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(l2_reg), name='decoder_conv2')(x)
    x = BatchNormalization()(x)
    
    # outputs
    # global branch
    features_flat = Reshape((-1, 1), name='features_for_global')(x) # (T, 32)
    global_pool = GlobalMaxPooling1D(name='global_max_pool')(features_flat)
    global_output = Dense(1, activation='sigmoid', name='global_switchback_presence')(global_pool)
    # localization
    # added the bias so that the model knows theres not many events
    # this is bc its edge detection model now
    # lost a shit ton of data
    output_bias = tf.keras.initializers.Constant(-4.6)
    detection_output = Conv2D(1, (1, 1), activation='sigmoid', name='switchback_detection',bias_initializer=output_bias)(x)
    detection_output = Reshape((-1, 1), name='final_flat_output')(detection_output)
    
    # creating the model
    model = Model(
        inputs=[ricker_br_input, ricker_vr_input, haar_br_input],
        outputs=[detection_output, global_output],
        name='unet_switchback_cnn'
    )

    # compiling the model here
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
        loss={
            'final_flat_output': combined_loss(alpha=0.85, gamma=2.0), # keep tweaking these, like it so far though
            'global_switchback_presence': 'binary_crossentropy'
        },
        loss_weights={
            'final_flat_output': 10.0, # so it does not lazy guess
            'global_switchback_presence': 1.0
        },
        # metrics for the paper
        # iou will be ass bc of how i labeled the data
        # alas
        metrics={
            'final_flat_output': [
                BinaryIoU(target_class_ids=[1], name='io_u'), 
                Recall(name='recall'),
                Precision(name='precision')
            ], 
            'global_switchback_presence': 'accuracy'
        }
    )
    
    return model
