from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

def create_dqn(input_shape=(10,10,10), n_actions = 6):
    """
    input_shape = size of game
    n_actions = amount of possible actions
    """
    model = keras.Sequential([
        layers.Input(shape=(*input_shape, 1)),

        layers.Conv3D(
            filters = 32,
            kernel_size = 3,
            activation = 'relu',
            padding = 'same'
        ),

        layers.Conv3D(
            filters = 64,
            kernel_size = 3,
            activation='relu',
            padding='same'
        ),

        layers.Conv3D(
            filters = 64,
            kernel_size = 3,
            activation='relu',
            padding='same'
        ),

        layers.Flatten(),

        layers.Dense(
            units=512,
            activation = 'relu'
        ),

        layers.Dense(
            units = n_actions,
            activation = None,
        )
    ])

    model.compile(
            optimizer =keras.optimizers.Adam(learning_rate = 0.0001),
            loss = 'mse'
    )

    return model



if __name__ == '__main__':
    print("Creating DQN")
    model = create_dqn()
    model.summary()
    print("testing forward pass")
    fake_state = np.zeros((2,10,10,10,1))
    q_values = model.predict(fake_state, verbose = 0)
    print(f"input shape = {fake_state.shape}")
    print(f"output shape = {q_values.shape}")
    print(f"same q-vals = {q_values[0]}")
    print("checked")
    




