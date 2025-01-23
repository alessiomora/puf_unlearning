import tensorflow as tf
from typing import Optional, Tuple, Union

BATCH_NORM_DECAY = 0.997
BATCH_NORM_EPSILON = 1e-5
LAYER_NORM_EPSILON = 1e-5
GROUP_NORM_EPSILON = 1e-5


class CNNModel(tf.keras.Model):
    def __init__(self, only_digits=True, seed=1):
        super().__init__()
        data_format = 'channels_last'
        self.conv1 = tf.keras.layers.Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                                            padding='same', activation=tf.nn.relu,
                                            kernel_initializer=tf.keras.initializers.HeNormal(
                                                seed=seed),
                                            )

        self.conv2 = tf.keras.layers.Conv2D(64, kernel_size=(5, 5), strides=(1, 1),
                                            padding='same', activation=tf.nn.relu,
                                            kernel_initializer=tf.keras.initializers.HeNormal(
                                                seed=seed),
                                            )
        self.maxpool1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='same',
                                                  data_format=data_format)
        self.maxpool2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='same',
                                                  data_format=data_format)
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(512, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(10 if only_digits else 47)

    def call(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x


def create_cnn_model(only_digits=True, seed=1):
    """The CNN model used in https://arxiv.org/abs/1602.05629.
    Args:
      only_digits: If True, uses a final layer with 10 outputs, for use with the
        digits only EMNIST dataset. If False, uses 62 outputs for the larger
        dataset.
    Returns:
      An uncompiled `tf.keras.Model`.
    """

    input_shape = [None, 28, 28, 1]
    model = CNNModel()
    model.build(input_shape)
    return model


def get_norm_layer(norm: str, channel_axis: int) -> tf.keras.layers.Layer:
    """Return the requested norm layer."""
    if norm == "batch":
        return tf.keras.layers.BatchNormalization(
            axis=channel_axis, momentum=BATCH_NORM_DECAY
        )

    if norm == "layer":
        return tf.keras.layers.LayerNormalization(
            axis=channel_axis, epsilon=LAYER_NORM_EPSILON
        )

    return tf.keras.layers.GroupNormalization(groups=2, epsilon=GROUP_NORM_EPSILON)


# pylint: disable=W0221
class ResBlock(tf.keras.Model):
    """Implement a ResBlock."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        filters: int,
        downsample: bool,
        norm: str = "group",
        l2_weight_decay: float = 1e-3,
        stride: int = 1,
        seed: Optional[Union[int, None]] = None,
    ):
        super().__init__()

        if tf.keras.backend.image_data_format() == "channels_last":
            channel_axis = 3
        else:
            channel_axis = 1

        self.conv1 = tf.keras.layers.Conv2D(
            filters,
            kernel_size=(3, 3),
            strides=stride,
            padding="same",
            use_bias=False,
            kernel_initializer=tf.keras.initializers.HeNormal(seed=seed),
            kernel_regularizer=tf.keras.regularizers.l2(l2_weight_decay),
        )

        if downsample:
            self.shortcut = tf.keras.Sequential(
                [
                    tf.keras.layers.Conv2D(
                        filters,
                        kernel_size=(1, 1),
                        strides=stride,
                        padding="valid",
                        use_bias=False,
                        kernel_initializer=tf.keras.initializers.HeNormal(seed=seed),
                        kernel_regularizer=tf.keras.regularizers.l2(l2_weight_decay),
                    ),
                    get_norm_layer(norm, channel_axis=channel_axis),
                ]
            )
        else:
            self.shortcut = tf.keras.Sequential()

        self.gn1 = get_norm_layer(norm, channel_axis=channel_axis)

        self.conv2 = tf.keras.layers.Conv2D(
            filters,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="same",
            use_bias=False,
            kernel_initializer=tf.keras.initializers.HeNormal(seed=seed),
            kernel_regularizer=tf.keras.regularizers.l2(l2_weight_decay),
        )
        self.gn2 = get_norm_layer(norm, channel_axis=channel_axis)

    def call(
        self,
        x: tf.Tensor,
    ):
        """Call the model on new inputs and returns the outputs as tensors."""
        shortcut = self.shortcut(x)

        x = self.conv1(x)
        x = self.gn1(x)
        x = tf.keras.layers.ReLU()(x)

        x = self.conv2(x)
        x = self.gn2(x)

        x = x + shortcut
        return tf.keras.layers.ReLU()(x)


# pylint: disable=W0221
class ResNet18(tf.keras.Model):
    """Implement a ResNet18 architecture as in FedMLB paper."""

    def __init__(
        self,
        outputs: int = 10,
        l2_weight_decay: float = 1e-3,
        norm: str = "",
        seed: Optional[Union[int, None]] = None,
    ):
        super().__init__()
        if seed is not None:
            tf.random.set_seed(seed)
        if tf.keras.backend.image_data_format() == "channels_last":
            channel_axis = 3
        else:
            channel_axis = 1

        self.layer0 = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(
                    64,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    padding="same",
                    use_bias=False,
                    kernel_initializer=tf.keras.initializers.HeNormal(seed=seed),
                    kernel_regularizer=tf.keras.regularizers.l2(l2_weight_decay),
                ),
                get_norm_layer(norm, channel_axis=channel_axis),
                tf.keras.layers.ReLU(),
            ],
            name="layer0",
        )

        self.layer1 = tf.keras.Sequential(
            [
                ResBlock(
                    64,
                    downsample=False,
                    l2_weight_decay=l2_weight_decay,
                    stride=1,
                    norm=norm,
                ),
                ResBlock(
                    64,
                    downsample=False,
                    l2_weight_decay=l2_weight_decay,
                    stride=1,
                    norm=norm,
                ),
            ],
            name="layer1",
        )

        self.layer2 = tf.keras.Sequential(
            [
                ResBlock(
                    128,
                    downsample=True,
                    l2_weight_decay=l2_weight_decay,
                    stride=2,
                    norm=norm,
                ),
                ResBlock(
                    128,
                    downsample=False,
                    l2_weight_decay=l2_weight_decay,
                    stride=1,
                    norm=norm,
                ),
            ],
            name="layer2",
        )

        self.layer3 = tf.keras.Sequential(
            [
                ResBlock(
                    256,
                    downsample=True,
                    l2_weight_decay=l2_weight_decay,
                    stride=2,
                    norm=norm,
                ),
                ResBlock(
                    256,
                    downsample=False,
                    l2_weight_decay=l2_weight_decay,
                    stride=1,
                    norm=norm,
                ),
            ],
            name="layer3",
        )

        self.layer4 = tf.keras.Sequential(
            [
                ResBlock(
                    512,
                    downsample=True,
                    l2_weight_decay=l2_weight_decay,
                    stride=2,
                    norm=norm,
                ),
                ResBlock(
                    512,
                    downsample=False,
                    l2_weight_decay=l2_weight_decay,
                    stride=1,
                    norm=norm,
                ),
            ],
            name="layer4",
        )

        self.gap = tf.keras.Sequential(
            [tf.keras.layers.GlobalAveragePooling2D()], name="gn_relu_gap"
        )
        self.fully_connected = tf.keras.layers.Dense(
            outputs,
            kernel_initializer=tf.keras.initializers.RandomNormal(
                stddev=0.01, seed=seed
            ),
            kernel_regularizer=tf.keras.regularizers.l2(l2_weight_decay),
            bias_regularizer=tf.keras.regularizers.l2(l2_weight_decay),
        )

    def call(self, x: tf.Tensor):
        """Call the model on new inputs and returns the outputs as tensors."""
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.gap(x)
        x = self.fully_connected(x)

        return x


def create_resnet18(
    num_classes: int = 100,
    input_shape: Tuple = (None, 32, 32, 3),
    norm: str = "group",
    l2_weight_decay: float = 0.0,
    seed: Optional[Union[int, None]] = None,
) -> tf.keras.Model:
    """Return a built ResNet model."""
    resnet18 = ResNet18(
        outputs=num_classes, l2_weight_decay=l2_weight_decay, seed=seed, norm=norm
    )
    resnet18.build(input_shape)
    return resnet18