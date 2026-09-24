import os
import math
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint
import tensorflow.keras.backend as K

IMG_SIZE = (512, 512)
BATCH_SIZE = 16
CLASS_NAMES = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']

def focal_loss_ordinal_echilibrat(gamma=2.0):
    def focal_loss_fn(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1.0 - K.epsilon())
        bce = -y_true * tf.math.log(y_pred) - (1.0 - y_true) * tf.math.log(1.0 - y_pred)
        p_t = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
        focal_factor = tf.math.pow(1.0 - p_t, gamma)
        loss = focal_factor * bce
        return tf.reduce_mean(tf.reduce_sum(loss, axis=-1))
    return focal_loss_fn

def get_cosine_warmup_scheduler(lr_max, total_epochs, warmup_epochs=2, lr_min=1e-6):
    def scheduler(epoch, lr):
        if epoch < warmup_epochs:
            return (lr_max / warmup_epochs) * (epoch + 1)
        progress = (epoch - warmup_epochs) / max(1, (total_epochs - warmup_epochs))
        return float(lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress)))
    return scheduler

def build_ordinal_efficientnet():
    base_model = EfficientNetB3(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE, 3))
    for layer in base_model.layers:
        layer.trainable = False
        
    x = GlobalAveragePooling2D()(base_model.output)
    x = BatchNormalization(name='batch_norm_top')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(4, activation='sigmoid', name='clasificator_ordinal')(x)
    
    model = Model(inputs=base_model.input, outputs=outputs)
    return model

fazele_antrenarii = [
    {"nume": "Faza_1_Cap", "desc": "Cap Ordinal", "keywords": [], "lr": 1e-3, "epoci": 10},
    {"nume": "Faza_2_Top", "desc": "+ Block 7", "keywords": ['block7'], "lr": 5e-5, "epoci": 10},
    {"nume": "Faza_3_Mid", "desc": "+ Block 6", "keywords": ['block7', 'block6'], "lr": 2e-5, "epoci": 10},
    {"nume": "Faza_4_Deep", "desc": "+ Block 4 & 5", "keywords": ['block7', 'block6', 'block5', 'block4'], "lr": 1e-5, "epoci": 10},
    {"nume": "Faza_5_Tot", "desc": "+ Block 2 & 3", "keywords": ['block7', 'block6', 'block5', 'block4', 'block3', 'block2'], "lr": 5e-6, "epoci": 10}
]