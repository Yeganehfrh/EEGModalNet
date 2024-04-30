import keras
from keras import ops, layers


class MultiInputModel(keras.Model):

    def __init__(self, input_shape):
        super().__init__()
        self.net = keras.Sequential([
            keras.Input(shape=input_shape, name='x'),
            layers.Flatten(),
            layers.Dense(32, activation='relu'),
            layers.Dense(2, activation='sigmoid')
        ])

    def call(self, inputs, training=None, mask=None):
        return self.net(inputs)


# OLD
# inputs
# input_x = layers.Input(shape=(128, 61,), name="x")
# input_sub_id = layers.Input(shape=(1,), name="sub_id")
# input_pos = layers.Input(shape=(61, 2,), name="pos")

# # heads
# head_x = keras.Sequential([input_x,
#                             layers.Flatten(),
#                             layers.Dense(64, activation="relu"),
#                             layers.Dense(2, activation='softmax')], name="head_x")(input_x)

# # model
# model = keras.Model(inputs=[input_x, input_sub_id, input_pos], outputs=head_x)
