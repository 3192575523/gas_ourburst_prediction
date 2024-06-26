import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Dense, Conv1D, GlobalAveragePooling1D, Dropout, Input
from tensorflow.keras.models import Model
# transformer for timeseries
class Transformer:
    def __init__(self, head_size, num_heads, ff_dim, num_trans_blocks, mlp_units, dropout=0, mlp_dropout=0, attention_axes=None, epsilon=1e-6, kernel_size=1):
        self.head_size = head_size
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_trans_blocks = num_trans_blocks
        self.mlp_units = mlp_units
        self.dropout = dropout
        self.mlp_dropout = mlp_dropout
        self.attention_axes = attention_axes
        self.epsilon = epsilon
        self.kernel_size = kernel_size
        self.model = self._build_model()

    def _build_model(self):
        n_timesteps, n_features, n_outputs = 60, 1, 1
        inputs = Input(shape=(n_timesteps, n_features))
        x = inputs
        for _ in range(self.num_trans_blocks):
            x = self._transformer_encoder(x)
        
        x = GlobalAveragePooling1D(data_format="channels_first")(x)
        for dim in self.mlp_units:
            x = Dense(dim, activation="relu")(x)
            x = Dropout(self.mlp_dropout)(x)
        
        outputs = Dense(n_outputs)(x)
        return Model(inputs, outputs)

    def _transformer_encoder(self, inputs):
        x = LayerNormalization(epsilon=self.epsilon)(inputs)
        x = MultiHeadAttention(key_dim=self.head_size, num_heads=self.num_heads, dropout=self.dropout, attention_axes=self.attention_axes)(x, x)
        x = Dropout(self.dropout)(x)
        res = x + inputs
        
        # Feed Forward Part
        x = LayerNormalization(epsilon=self.epsilon)(res)
        x = Conv1D(filters=self.ff_dim, kernel_size=self.kernel_size, activation="relu")(x)
        x = Dropout(self.dropout)(x)
        x = Conv1D(filters=inputs.shape[-1], kernel_size=self.kernel_size)(x)
        return x + res

    def compile(self, optimizer=tf.keras.optimizers.Adam(0.001), loss='mean_squared_error'):
        self.model.compile(optimizer=optimizer, loss=loss)

    def fit(self, x_train, y_train, batch_size=64, epochs=50, validation_data=None, validation_freq=1, callbacks=None):
        history = self.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=validation_data, validation_freq=validation_freq, callbacks=callbacks)
        self.history = history
        return history

    def predict(self, x_test):
        return self.model.predict(x_test)
# Optuna using in transformer
import optuna
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dropout, Dense
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from optuna.integration.tfkeras import TFKerasPruningCallback

def objective(trial):
    head_size = trial.suggest_int('head_size', 32, 128)
    num_heads = trial.suggest_int('num_heads', 2, 6)
    ff_dim = trial.suggest_int('ff_dim', 128, 512)
    num_trans_blocks = trial.suggest_int('num_trans_blocks', 1, 4)
    num_mlp_layers = trial.suggest_int('num_mlp_layers', 1, 3)  # 添加这一行
    mlp_units = []

    for i in range(num_mlp_layers):
        mlp_units.append(trial.suggest_int(f'mlp_units_{i}', 128, 512))
    dropout = trial.suggest_float('dropout', 0, 0.5)
    mlp_dropout = trial.suggest_float('mlp_dropout', 0, 0.5)

    transformer_model = TransformerTimeSeriesModel(
        head_size=head_size,
        num_heads=num_heads,
        ff_dim=ff_dim,
        num_trans_blocks=num_trans_blocks,
        mlp_units=mlp_units,
        dropout=dropout,
        mlp_dropout=mlp_dropout
    )
    transformer_model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='mean_squared_error')

    pruning_callback = TFKerasPruningCallback(trial, 'val_loss')
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint("my_model.h5", save_best_only=True, monitor='val_loss', mode='min')
    EPOCHS = 50
    history = transformer_model.fit(
        x_train, 
        y_train, 
        batch_size=64, 
        epochs=EPOCHS, 
        validation_data=(x_test, y_test), 
        validation_freq=1, 
        callbacks=[pruning_callback, model_checkpoint]
    )

    predicted_stock_price = transformer_model.predict(x_test)
    real_stock_price = test_set[60:, 1]
    mse = mean_squared_error(predicted_stock_price, real_stock_price)

    return mse

from PyEMD import EEMD
import numpy as np

class EEMDProcessor:
    def __init__(self):
        self.eemd = EEMD()
    
    def find_k(self, signal):
        # 执行EEMD分解
        self.eIMFs = self.eemd(signal)
        self.rows, self.cols = self.eIMFs.shape
        # rows is the definition of dimension in vmd
        return self.rows
