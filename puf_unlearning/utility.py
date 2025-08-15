import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from puf_unlearning.dataset import element_norm_cifar100, normalize_img, expand_dims, \
    element_norm_cifar10, PaddedRandomCrop, element_norm_cifar100_train, \
    element_norm_cifar20, element_norm_cifar10_train
from puf_unlearning.model import create_cnn_model, create_resnet18
from puf_unlearning.transformer_utility import get_transformer_model


def save_line_to_file(folder_path, file_name, line):
    """
    Save a line to a file in a specific folder. Creates the folder if it doesn't exist.

    Args:
        folder_path (str): Path to the folder where the file should be saved.
        file_name (str): Name of the file to save the line in.
        line (str): The line to save to the file.
    """
    # Ensure the folder exists
    os.makedirs(folder_path, exist_ok=True)

    # Full file path
    file_path = os.path.join(folder_path, file_name)

    # Append the line to the file
    with open(file_path, 'a') as file:
        file.write(line + '\n')


def draw_and_save_heatmap(np_array_data, alpha_string,
                          chart_folder=os.path.join("charts"), mode="single_class",
                          title="Accuracy After Unlearning"):
    print("---- Drawing and saving chart ----")
    plt.figure()

    g = sns.heatmap(np_array_data, annot=True, cmap="Reds", fmt='.4f')

    if mode == "all_class":
        x_label = ""
        ll = ["test", "train", "test", "train", "kl_div (o)", "kl_div (u)"]

        # Hide major tick labels
        # g.set_xticklabels('')
        g.set_xticks(np.arange(0.5, len(ll), 1).tolist())
        g.set_xticklabels(ll)


    else:
        x_label = "Class"
        # title = "Accuracy After Unlearning"

    filename = "heatmap_" + alpha_string + "_" + title.lower().replace(" ", "_") + ".pdf"
    g.set_ylabel('Client', fontsize=18)
    g.set_xlabel(x_label, fontsize=18)
    g.set_title(title, fontsize=19, pad=20)

    exist = os.path.exists(chart_folder)
    if not exist:
        os.makedirs(chart_folder)

    g.get_figure().savefig(os.path.join(chart_folder, filename),
                           format='pdf', bbox_inches='tight')
    plt.show()
    np.save(os.path.join(chart_folder, alpha_string + "_" + title.lower().replace(" ", "_")+".npy"), np_array_data)


def compute_kl_div(logits, num_classes):
    kl = tf.keras.losses.KLDivergence()
    pred_uniform_prob = tf.fill([tf.shape(logits)[0], num_classes], 1 / num_classes)
    kl_div = kl(pred_uniform_prob, tf.nn.softmax(logits))
    return kl_div


def compute_overlap_predictions(logit_1, logit_2):
    pred_1 = tf.nn.softmax(logit_1)
    pred_2 = tf.nn.softmax(logit_2)
    argmax_1 = tf.math.argmax(pred_1, axis=1)
    argmax_2 = tf.math.argmax(pred_2, axis=1)
    # print(argmax_1)
    # print(argmax_2)
    overlap = tf.reduce_sum(
        tf.cast(tf.equal(argmax_1, argmax_2), tf.float32)) / tf.cast(tf.size(argmax_1),
                                                                     tf.float32)
    return overlap



def create_model(dataset, total_classes, norm="group"):
    if dataset in ["mnist"]:
        model = create_cnn_model()
    elif dataset in ["cifar100", "cifar10", "cifar20"]:
        model = create_resnet18(
            num_classes=total_classes,
            # input_shape=input_shape,
            norm=norm,
            seed=1,
        )
    elif dataset in ["cifar100-transformer", "birds-transformer"]:
        model = get_transformer_model(
            model_name="mit-b0",
            # classifier_hidden_layers=classifier_hidden_layers,
            num_classes=total_classes,
            # random_seed=random_seed,
            load_pretrained_weights=True,
            # trainable_feature_extractor=trainable_feature_extractor,
            # trainable_blocks_fe=trainable_blocks_fe
        )
    return model


def get_test_dataset(dataset, take_n=10000):
    if dataset in ["mnist"]:
        ds_test = tfds.load(
                    'mnist',
                    split='test',
                    shuffle_files=True,
                    as_supervised=True,
                )

        ds_test = ds_test.map(
                    normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
        # ds_test = ds_test.map(expand_dims)

    elif dataset in ["cifar100"]:
        cifar100_tfds = tfds.load("cifar100")
        ds_test = cifar100_tfds["test"]
        ds_test = ds_test.map(element_norm_cifar100)
    elif dataset in ["cifar10"]:
        cifar10_tfds = tfds.load("cifar10")
        ds_test = cifar10_tfds["test"]
        ds_test = ds_test.map(element_norm_cifar10)
    elif dataset in ["cifar20"]:
        cifar100_tfds = tfds.load("cifar100", as_supervised=False)
        ds_test = cifar100_tfds["test"]
        ds_test = ds_test.map(element_norm_cifar20)
    elif dataset in ["cifar100-transformer"]:
        (_, _), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
        ds_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        ds_test = (
            ds_test.map(
                preprocess_dataset_for_transformers_models(is_training=False))
            .map(get_normalization_fn(model_name="mit-b0", dataset="cifar100"))
        )
    elif dataset in ["birds-transformer"]:
        ds_test = tfds.load("caltech_birds2011", split='test',
                            shuffle_files=False, as_supervised=True)
        ds_test = (
            ds_test.map(
                preprocess_dataset_for_birds_aircafts_cars(is_training=False))
            .map(get_normalization_fn("mit-b0", dataset="birds"))
            # .batch(TEST_BATCH_SIZE)
        )
        print(f"------- Take n {take_n}")


    ds_test = ds_test.take(take_n)
    ds_test = ds_test.batch(32)
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

    return ds_test


def preprocess_ds_test(ds, dataset="mnist", reshuffle_each_iteration=True):
    if dataset in ["mnist"]:
        ds = ds.shuffle(2048, reshuffle_each_iteration=reshuffle_each_iteration).map(normalize_img).map(expand_dims)
    elif dataset in ["cifar100"]:
        ds = ds.shuffle(1024, reshuffle_each_iteration=reshuffle_each_iteration).map(element_norm_cifar100_train)
    elif dataset in ["cifar10"]:
        ds = ds.shuffle(1024, reshuffle_each_iteration=reshuffle_each_iteration).map(element_norm_cifar10_train)
    elif dataset in ["cifar20"]:
        ds = ds.shuffle(1024, reshuffle_each_iteration=reshuffle_each_iteration).map(element_norm_cifar20)
    elif dataset in ["cifar100-transformer"]:
        ds = ds.shuffle(32, reshuffle_each_iteration=reshuffle_each_iteration).map(
                preprocess_dataset_for_transformers_models(is_training=False)).map(get_normalization_fn(model_name="mit-b0", dataset="cifar100"))
    elif dataset in ["birds-transformer"]:
        ds = ds.shuffle(32, reshuffle_each_iteration=reshuffle_each_iteration).map(
            preprocess_dataset_for_birds_aircafts_cars(is_training=False)).map(
            get_normalization_fn(model_name="mit-b0", dataset="birds"))
    return ds


def preprocess_ds(ds, dataset="mnist"):
    if dataset in ["mnist"]:
        ds = ds.shuffle(2048).map(normalize_img).map(expand_dims)
    elif dataset in ["cifar100", "cifar10"]:
        # transform images
        rotate = tf.keras.layers.RandomRotation(0.06, seed=1)
        flip = tf.keras.layers.RandomFlip(mode="horizontal", seed=1)
        crop = PaddedRandomCrop(seed=1)

        rotate_flip_crop = tf.keras.Sequential([
            rotate,
            crop,
            flip,
        ])

        def transform_data(image, img_label):
            return rotate_flip_crop(image), img_label

        if dataset in ["cifar100"]:
            ds = ds.shuffle(1024).map(element_norm_cifar100_train).map(transform_data)
        elif dataset in ["cifar10"]:
            ds = ds.shuffle(1024).map(element_norm_cifar10_train).map(transform_data)


    elif dataset in ["cifar100-transformer"]:
        ds = (ds.shuffle(256)
              .map(preprocess_dataset_for_transformers_models(is_training=True))
              .map(get_normalization_fn(model_name="mit-b0", dataset="cifar100")))
    elif dataset in ["birds-transformer"]:
        ds = (
            ds.shuffle(256)
            .map(preprocess_dataset_for_birds_aircafts_cars(is_training=True))
            .map(get_normalization_fn(model_name="mit-b0", dataset="birds"))

        )

    return ds


def preprocess_dataset_for_transformers_models(is_training=True, resolution=224):
    def resize_and_crop_fn(image, label):
        if is_training:
            # Resize to a bigger spatial resolution and take the random
            # crops.
            image = tf.image.resize(image, (resolution + 20, resolution + 20))
            image = tf.image.random_crop(image, (resolution, resolution, 3))
            image = tf.image.random_flip_left_right(image)
        else:
            image = tf.image.resize(image, (resolution, resolution))
        return image, label

    return resize_and_crop_fn



def get_normalization_fn(model_name="mit-b0", dataset="cifar100"):
    """Return the normalization function based on model family and dataset."""
    if model_name.startswith("mit"):
        transpose = True
    else:
        transpose = False

    if dataset in ["cifar100"]:
        mean = [0.5071, 0.4865, 0.4409]
        variance = [np.square(0.2673), np.square(0.2564), np.square(0.2762)]
    elif dataset in ["aircrafts"]:
        mean = [0.4862, 0.5179, 0.5420]
        variance = [np.square(0.1920), np.square(0.1899), np.square(0.2131)]
    elif dataset in ["cars"]:
        mean = [0.4668, 0.4599, 0.4505]
        variance = [np.square(0.2642), np.square(0.2636), np.square(0.2687)]
    else: # birds
        mean = [0.485, 0.456, 0.406]
        variance = [np.square(0.229), np.square(0.224), np.square(0.225)]

    def element_norm_fn(image, label):
        """Normalize cifar100 images."""
        norm_layer = tf.keras.layers.Normalization(
            mean=mean,
            variance=variance,
        )
        if transpose:
            return tf.transpose(norm_layer(tf.cast(image, tf.float32) / 255.0),
                                    (2, 0, 1)), label

        return norm_layer(tf.cast(image, tf.float32) / 255.0), label

    return element_norm_fn

def list_clients_to_string(unlearned_cid):
    s = ""
    for u in unlearned_cid:
        if s != "":
            s = s+"_"
        s = s + str(u)
    print(f"--- unlearning cid string {s} -----")
    return s


class RandomResizedCrop(tf.keras.layers.Layer):
    """Preprocessing birds, aircrafts, cars."""
    def __init__(self, size=(224, 224), scale=(0.08, 1.0), ratio=(0.75, 1.3333), **kwargs):
        super().__init__(**kwargs)
        self.crop_shape = size
        self.scale = scale
        self.log_ratio = (tf.math.log(ratio[0]), tf.math.log(ratio[1]))

    def call(self, inputs: tf.Tensor):
        """Call the layer on new inputs and returns the outputs as tensors."""

        inputs = tf.expand_dims(inputs, axis=0)
        batch_size = tf.shape(inputs)[0]
        # tf.print("batch_size ", batch_size)
        random_scales = tf.random.uniform(
            (batch_size,),
            self.scale[0],
            self.scale[1]
        )
        random_ratios = tf.exp(tf.random.uniform(
            (batch_size,),
            self.log_ratio[0],
            self.log_ratio[1]
        ))

        new_heights = tf.clip_by_value(
            tf.sqrt(random_scales / random_ratios),
            0,
            1,
        )
        new_widths = tf.clip_by_value(
            tf.sqrt(random_scales * random_ratios),
            0,
            1,
        )
        height_offsets = tf.random.uniform(
            (batch_size,),
            0,
            1 - new_heights,
        )
        width_offsets = tf.random.uniform(
            (batch_size,),
            0,
            1 - new_widths,
        )

        bounding_boxes = tf.stack(
            [
                height_offsets,
                width_offsets,
                height_offsets + new_heights,
                width_offsets + new_widths,
            ],
            axis=1,
        )
        images = tf.image.crop_and_resize(
            inputs,
            bounding_boxes,
            tf.range(batch_size),
            self.crop_shape,
        )
        image = tf.squeeze(images, axis=0)
        # tf.print("image ", tf.shape(image))
        return image


def preprocess_dataset_for_birds_aircafts_cars(is_training=True, resolution=224):
    def resize_and_crop_fn(image, label):
        if is_training:
            # Resize to a bigger spatial resolution and take the random
            # crops.
            random_resize_crop_fn = RandomResizedCrop(size=(resolution, resolution))
            image = random_resize_crop_fn(image)
            image = tf.image.random_flip_left_right(image)
            # image = tf.image.resize(image, (256, 256))
            # image = tf.keras.layers.CenterCrop(resolution, resolution)(image)
        else:
            # image = tf.image.resize(image, (256, 256))
            image = tf.keras.layers.CenterCrop(resolution, resolution)(image)
        return image, label

    return resize_and_crop_fn