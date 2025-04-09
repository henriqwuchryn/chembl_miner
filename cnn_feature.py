import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet import preprocess_input
import sys

filename = sys.argv[1]
fingerprint_df = pd.read_csv(f'datasets/{filename}', index_col='index')
fingerprint_df.replace([np.inf, -np.inf], np.nan, inplace=True)
fingerprint_df.dropna(inplace=True)
features_df = fingerprint_df.drop(
    ['molecule_chembl_id','neg_log_value','bioactivity_class'],axis=1)
input_shape = shape(features_df)

cnn = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
inputs = keras.Input(shape=input_shape)
x = preprocess_input(inputs)
x = cnn(x)
output = GlobalAveragePooling2D()(x)
model = Model(inputs, output)

x_cnn = model.predict(features_df)
x_cnn = pd.DataFrame(x_cnn)
x_cnn.to_csv('teste_cnn.csv',index=True,index_label='index')
