"""Handle the dataset partitioning and (optionally) complex downloads.

Please add here all the necessary logic to either download, uncompress, pre/post-process
your dataset (or all of the above). If the desired way of running your baseline is to
first download the dataset and partition it and then run the experiments, please
uncomment the lines below and tell us in the README.md (see the "Running the Experiment"
block) that this file should be executed first.
"""
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import call, instantiate
from omegaconf import DictConfig, OmegaConf
import os
from collections import defaultdict

from basics_unlearning.dataset import get_string_distribution

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
import shutil
import sys
import ast
from PIL import Image

np.set_printoptions(threshold=sys.maxsize)

def read_fedquit_distribution(dataset, total_clients, alpha, data_folder="client_data"):
    file_path = os.path.join(data_folder, dataset, "unbalanced",
                             alpha + "_clients" + str(total_clients) + ".txt")

    # reading the data from the file
    with open(file_path) as f:
        data = f.read()

    # reconstructing the data as a dictionary
    data_mlb = ast.literal_eval(data)

    return data_mlb


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


@hydra.main(config_path="conf", config_name="base", version_base=None)
def download_and_preprocess(cfg: DictConfig) -> None:
    """Does everything needed to get the dataset.

    Parameters
    ----------
    cfg : DictConfig
        An omegaconf object that stores the hydra config.
    """

    ## print parsed config
    print(OmegaConf.to_yaml(cfg))

    dataset = cfg.dataset
    alpha = cfg.alpha
    alpha_dirichlet_string = get_string_distribution(alpha)

    total_clients = cfg.total_clients

    folder = "federated_datasets_sample_unl"
    if dataset in ["cifar100"]:
        num_classes = 100
    else:
        num_classes = 10

    # if the folder exist it is deleted and the ds partitions are re-created
    # if the folder does not exist, firstly the folder is created
    # and then the ds partitions are generated

    folder_path = os.path.join(folder, dataset, str(total_clients), alpha_dirichlet_string)
    exist = os.path.exists(folder_path)
    if not exist:
        os.makedirs(folder_path)
    else:
        shutil.rmtree(folder_path, ignore_errors=True)

    if dataset in ["cifar10"]:
        (x_train, y_train), (_, _) = tf.keras.datasets.cifar10.load_data()

    elif dataset in ["cifar100"]:
        (x_train, y_train), (_, _) = tf.keras.datasets.cifar100.load_data()

    # read the distribution of per-label examples for each client
    # from txt file
    data_mlb = read_fedquit_distribution(dataset, total_clients=total_clients, alpha=alpha_dirichlet_string)

    # Create output folders
    os.makedirs(os.path.join(folder_path, "train"), exist_ok=True)
    os.makedirs(os.path.join(folder_path, "forget"), exist_ok=True)

    dict_folder_path = os.path.join(
        "client_data_sample_unl",
        dataset,
        "unbalanced",
    )
    dict_path = os.path.join(
        dict_folder_path,
        alpha_dirichlet_string + "_clients" + str(total_clients) + ".txt"
    )

    exist = os.path.exists(dict_folder_path)
    if not exist:
        os.makedirs(dict_folder_path)

    # exist = os.path.exists(dict_path)
    # if exist:
    #     os.remove(dict_path)

    # Check if split file exists
    if os.path.exists(dict_path):
        with open(dict_path, "r") as f:
            split_dict = ast.literal_eval(f.read())
        exist = True
    else:
        split_dict = {}
        exist = False

    for client in data_mlb:
        # Check if split file exists
        if exist:
            idx_train = split_dict[f"{client}_train"]
            idx_forget = split_dict[f"{client}_forget"]
        else:
            list_indices = np.array(data_mlb[client])
            np.random.seed(3019 + int(client))
            np.random.shuffle(list_indices)

            mid = len(list_indices) // 2
            idx_train = list_indices[:mid].tolist()
            idx_forget = list_indices[mid:].tolist()

            split_dict.update({f"{client}_train": idx_train, f"{client}_forget": idx_forget})

        # Create datasets
        x_train_client = x_train[idx_train]
        y_train_client = y_train[idx_train]

        x_forget_client = x_train[idx_forget]
        y_forget_client = y_train[idx_forget]

        ds_train = tf.data.Dataset.from_tensor_slices(
            (x_train_client, y_train_client)).shuffle(4096)
        ds_forget = tf.data.Dataset.from_tensor_slices(
            (x_forget_client, y_forget_client)).shuffle(4096)

        path = os.path.join(folder_path, "train", str(client))
        print(f"save ds at {path}")
        tf.data.Dataset.save(ds_train,
                             path=os.path.join(folder_path, "train", str(client)))
        tf.data.Dataset.save(ds_forget,
                             path=os.path.join(folder_path, "forget", str(client)))

    if not exist:
        with open(dict_path, "w") as f:
            f.write(str(split_dict))
        print(f"save dict at {dict_path}")

        print("All clients processed with 50/50 train/forget splits stored or reused from file.")

    path_train = os.path.join(os.path.join(folder_path, "train"))
    path_forget = os.path.join(os.path.join(folder_path, "forget"))

    list_of_narrays_train = []
    list_of_narrays_forget = []
    for sampled_client in range(0, total_clients):
        loaded_ds_train = tf.data.Dataset.load(
            path=os.path.join(path_train, str(sampled_client)), element_spec=None, compression=None, reader_func=None
        )
        loaded_ds_forget = tf.data.Dataset.load(
            path=os.path.join(path_forget, str(sampled_client)), element_spec=None, compression=None, reader_func=None
        )

        print("[Client " + str(sampled_client) + "]")
        print("Cardinality: ", tf.data.experimental.cardinality(loaded_ds_train).numpy())
        print("[Client " + str(sampled_client) + "]")
        print("Cardinality: ", tf.data.experimental.cardinality(loaded_ds_forget).numpy())

        def count_class(counts, batch, num_classes=num_classes):
            _, labels = batch
            for i in range(num_classes):
                cc = tf.cast(labels == i, tf.int32)
                counts[i] += tf.reduce_sum(cc)
            return counts

        initial_state = dict((i, 0) for i in range(num_classes))
        counts = loaded_ds_train.reduce(initial_state=initial_state, reduce_func=count_class)

        # print([(k, v.numpy()) for k, v in counts.items()])
        new_dict = {k: v.numpy() for k, v in counts.items()}
        # print(new_dict)
        res = np.array([item for item in new_dict.values()])
        # print(res)
        list_of_narrays_train.append(res)

        initial_state = dict((i, 0) for i in range(num_classes))
        counts = loaded_ds_forget.reduce(initial_state=initial_state,
                                        reduce_func=count_class)

        # print([(k, v.numpy()) for k, v in counts.items()])
        new_dict = {k: v.numpy() for k, v in counts.items()}
        # print(new_dict)
        res = np.array([item for item in new_dict.values()])
        # print(res)
        list_of_narrays_forget.append(res)

    distribution_t = np.stack(list_of_narrays_train)
    print(distribution_t)

    distribution_f = np.stack(list_of_narrays_forget)
    print(distribution_f)
    # saving the distribution of per-label examples in a numpy file
    # this can be useful also to draw charts about the label distrib.
    path = os.path.join(folder_path, "distribution_train.npy")
    np.save(path, distribution_t)
    print(f"save .npy at {path}")

    path = os.path.join(folder_path, "distribution_forget.npy")
    np.save(path, distribution_f)
    print(f"save .npy at {path}")


if __name__ == "__main__":
    download_and_preprocess()