"""Handle dataset loading and preprocessing utility."""
import os
from typing import Union

import keras_cv
import numpy as np
import tensorflow as tf


def get_add_unlearning_label_fn(dataset, unlearning_label):

    def add_unl_label_cifar100_fn(image, label):
        return image, label, unlearning_label

    if dataset in ["cifar100", "cifar20"]:
        return add_unl_label_cifar100_fn
    else:
        return 0


def get_preprocess_fn(dataset):
    if dataset in ["mnsit"]:
        preprocess_fn = normalize_img
    elif dataset in ["cifar10"]:
        preprocess_fn = element_norm_cifar10_train
    elif dataset in ["cifar100"]:
        preprocess_fn = element_norm_cifar100_train

    return preprocess_fn


def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255., label


def expand_dims(image, label):
    """"""
    return tf.expand_dims(image, axis=-1), label


def element_norm_cifar10(element):
    """Utility function to normalize input images."""
    norm_layer = tf.keras.layers.Normalization(mean=[0.4914, 0.4822, 0.4465],
                                               variance=[np.square(0.2470),
                                                         np.square(0.2435),
                                                         np.square(0.2616)])
    return norm_layer(tf.cast(element["image"], tf.float32) / 255.0), element["label"]


def element_norm_cifar100(element):
    """Utility function to normalize input images."""
    norm_layer = tf.keras.layers.Normalization(mean=[0.5071, 0.4865, 0.4409],
                                               variance=[np.square(0.2673),
                                                         np.square(0.2564),
                                                         np.square(0.2762)])
    return norm_layer(tf.cast(element["image"], tf.float32) / 255.0), element["label"]


def element_norm_cifar20(element):
    """Utility function to normalize input images."""
    norm_layer = tf.keras.layers.Normalization(mean=[0.5071, 0.4865, 0.4409],
                                               variance=[np.square(0.2673),
                                                         np.square(0.2564),
                                                         np.square(0.2762)])
    return norm_layer(tf.cast(element["image"], tf.float32) / 255.0), element["coarse_label"]


def element_norm_cifar100_train(image, label):
    """Utility function to normalize input images."""
    norm_layer = tf.keras.layers.Normalization(mean=[0.5071, 0.4865, 0.4409],
                                               variance=[np.square(0.2673),
                                                         np.square(0.2564),
                                                         np.square(0.2762)])
    return norm_layer(tf.cast(image, tf.float32) / 255.0), tf.squeeze(label)


def element_norm_cifar10_train(image, label):
    """Utility function to normalize input images."""
    norm_layer = tf.keras.layers.Normalization(mean=[0.4914, 0.4822, 0.4465],
                                               variance=[np.square(0.2470),
                                                         np.square(0.2435),
                                                         np.square(0.2616)])
    return norm_layer(tf.cast(image, tf.float32) / 255.0), tf.squeeze(label)


# def element_norm_cifar20_train(element):
#     """Utility function to normalize input images."""
#     norm_layer = tf.keras.layers.Normalization(mean=[0.5071, 0.4865, 0.4409],
#                                                variance=[np.square(0.2673),
#                                                          np.square(0.2564),
#                                                          np.square(0.2762)])
#     return norm_layer(tf.cast(element["image"], tf.float32) / 255.0), element["coarse_label"]


class PaddedRandomCrop(keras_cv.layers.BaseImageAugmentationLayer):
    def __init__(self, seed=None, **kwargs):
        super().__init__(**kwargs)
        self.seed = seed

    def augment_image(self, image, transformation=None, **kwargs):
        # image is of shape (height, width, channels)
        image = tf.image.resize_with_crop_or_pad(image=image, target_height=32 + 4,
                                                     target_width=32 + 4)
        image = tf.image.random_crop(value=image, size=[32, 32, 3], seed=self.seed)
        return image


def get_string_distribution(alpha: float,):
    if alpha == 0:
        alpha_dirichlet_string = "one_label"
    elif alpha < 0:
        alpha_dirichlet_string = "iid"
    else:
        alpha_dirichlet_string = str(round(alpha, 2))
    return alpha_dirichlet_string


def load_selected_client_statistics(
        selected_client: int,
        alpha: float,
        dataset: str,
        total_clients: int,
):
    """Return the amount of local examples for the selected client.

    Clients are referenced with a client_id. Loads a numpy array saved on disk. This
    could be done directly by doing len(ds.to_list()) but it's more expensive at run
    time.
    """
    if dataset in ["cifar100-transformer"]:
        dataset = "cifar100"

    if dataset in ["cifar20"]:
        if selected_client == 0:
            return 500
        else:
            return int((50000 - 500) / 9)

    alpha_dirichlet_string = get_string_distribution(alpha)
    path = os.path.join(
        "federated_datasets",
        dataset,
        str(total_clients),
        alpha_dirichlet_string,
        "distribution_train.npy",
    )
    smpls_loaded = np.load(path)
    # tf.print(smpls_loaded, summarize=-1)
    local_examples_all_clients = np.sum(smpls_loaded, axis=1)
    return local_examples_all_clients[selected_client]


def load_selected_clients_statistics(selected_clients, alpha, dataset, total_clients):
    if dataset in ["cifar100-transformer"]:
        dataset = "cifar100"
    if dataset in ["cifar20"]:
        n = int((50000 - 500) / 9)
        local_examples_all_clients = np.array([500, n, n, n, n, n, n, n, n, n])
        return local_examples_all_clients[selected_clients.tolist()]

    alpha_dirichlet_string = get_string_distribution(alpha)
    path = os.path.join(
        "federated_datasets",
        dataset,
        str(total_clients),
        alpha_dirichlet_string,
        "distribution_train.npy",
    )
    smpls_loaded = np.load(path)
    local_examples_all_clients = np.sum(smpls_loaded, axis=1)
    if type(selected_clients) == list:
        return local_examples_all_clients[selected_clients]
    return local_examples_all_clients[selected_clients.tolist()]


def load_label_distribution(alpha, dataset, total_clients):
    if dataset in ["cifar100-transformer"]:
        dataset = "cifar100"
    if dataset in ["cifar20"]:
        n = int((50000 - 500) / 9)
        smpls_loaded = np.array([500, n, n, n, n, n, n, n, n, n])
        return smpls_loaded

    alpha_dirichlet_string = get_string_distribution(alpha)

    path = os.path.join(
        "federated_datasets",
        dataset,
        str(total_clients),
        alpha_dirichlet_string,
        "distribution_train.npy",
    )
    smpls_loaded = np.load(path)
    return smpls_loaded


def load_label_distribution_selected_client(selected_client, alpha, dataset, total_clients):
    smpls_loaded = load_label_distribution(alpha, dataset, total_clients)
    return smpls_loaded[selected_client]


def load_client_datasets_from_files(  # pylint: disable=too-many-arguments
        # pylint: disable=too-many-locals
        selected_client: int,
        dataset: str = "mnist",
        # batch_size: int = 32,
        total_clients: int = 100,
        alpha: float = 0.3,
        split: str = "train",
        cifar20_case: str = "rocket",
):
    """Load the partition of the dataset for the sampled client.

    Sampled client represented by its client_id.
    Returns a batched dataset.
    """
    if dataset in ["cifar100-transformer"]:
        dataset = "cifar100"

    if dataset in ["cifar20"]:
        path = os.path.join(
            "federated_datasets",
            dataset,
            str(total_clients),
            cifar20_case,
            split,
        )

        loaded_ds = tf.data.Dataset.load(
            path=os.path.join(path, str(selected_client)),
            element_spec=None,
            compression=None,
            reader_func=None,
        )
        return loaded_ds

    alpha_dirichlet_string = get_string_distribution(alpha)
    path = os.path.join(
        "federated_datasets",
        dataset,
        str(total_clients),
        alpha_dirichlet_string,
        split,
    )

    loaded_ds = tf.data.Dataset.load(
        path=os.path.join(path, str(selected_client)),
        element_spec=None,
        compression=None,
        reader_func=None,
    )
    return loaded_ds
