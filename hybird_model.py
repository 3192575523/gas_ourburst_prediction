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

class AdaptiveNormalization:
    def __init__(self, window_length=2, MA_type='ema', k=1.0, q1=25, q3=75):
        self.window_length = window_length
        self.MA_type = MA_type
        self.k = k
        self.q1 = q1
        self.q3 = q3

    def calculate_adjustment_level(self, S, DSWs):
        adjustment_levels = []
        for i in range(len(S)):
            numerator_sum = 0
            denominator_sum = 0

            for l in range(len(DSWs)):
                num_ij = self.calculate_numerator(S[i], DSWs[l])
                den_ij = self.calculate_denominator(S[i], DSWs[l])

                if den_ij is not None and den_ij != 0:
                    numerator_sum += num_ij
                    denominator_sum += den_ij

            if denominator_sum != 0:
                adjustment_level_i = abs(numerator_sum / denominator_sum - 1) / len(DSWs)
                adjustment_levels.append(adjustment_level_i)

        return adjustment_levels

    def calculate_numerator(self, value, DSW):
        if self.MA_type == "SMA":
            return sum(DSW) / len(DSW)
        elif self.MA_type == "EMA":
            alpha = 2 / (self.k + 1)
            num = sum(value * alpha * (1 - alpha) ** i * DSW[-(i + 1)] for i in range(len(DSW)))
            return num

    def calculate_denominator(self, value, DSW):
        if self.MA_type == "SMA":
            return sum(DSW) / len(DSW)
        elif self.MA_type == "EMA":
            alpha = 2 / (self.k + 1)
            den = sum(alpha * (1 - alpha) ** i * DSW[-(i + 1)] for i in range(len(DSW)))
            return den if den != 0 else 1e-10  # Avoid division by zero

    def calculate_moving_average(self, sequence):
        if self.MA_type == 'sma':
            return np.convolve(sequence, np.ones(self.window_length)/self.window_length, mode='valid')
        elif self.MA_type == 'ema':
            alpha = 2 / (self.window_length + 1)
            weights = np.exp(np.linspace(-1.0, 0.0, self.window_length) * alpha)
            weights /= weights.sum()
            return np.convolve(sequence, weights, mode='valid')

    def select_best_moving_average(self, data):
        best_moving_average = np.mean(data)
        best_adjustment_level = float('inf')

        for order in range(1, self.window_length):
            moving_averages = self.calculate_moving_average(data)
            adjustment_levels = self.calculate_adjustment_level(data, moving_averages)

            if adjustment_levels and min(adjustment_levels) is not None and min(adjustment_levels) < best_adjustment_level:
                best_moving_average = moving_averages
                best_adjustment_level = min(adjustment_levels)

        return best_moving_average, best_adjustment_level

    def remove_outliers(self, data):
        if data is not None:
            q1 = np.percentile(data, self.q1)
            q3 = np.percentile(data, self.q3)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            return [value for value in data if lower_bound <= value <= upper_bound]
        else:
            return []

    def normalize_data(self, data):
        if data is not None and len(data) > 0:
            scaler = MinMaxScaler(feature_range=(-1, 1))
            normalized_data = scaler.fit_transform(np.array(data).reshape(-1, 1))
            return normalized_data, scaler
        else:
            return [], None

    def denormalize_values(self, normalized_values, scaler):
        return scaler.inverse_transform(normalized_values) if normalized_values is not None else None
