""" This script partions the CIFAR10/CIFAR100 dataset in a federated fashion.
The level of non-iidness is defined via the alpha parameter (alpha in the paper below as well)
for a dirichlet distribution, and rules the distribution of examples per label on clients.
This implementation is based on the paper: https://arxiv.org/abs/1909.06335
"""
import json
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_probability as tfp
import numpy as np
import os
import shutil


# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from dataset import get_string_distribution

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def generate_dirichlet_samples(num_of_classes, alpha, num_of_clients,
                               num_of_examples_per_label):
    """Generate samples from a dirichlet distribution based on alpha parameter.
    Samples will have the shape (num_of_clients, num_of_classes).
    Returns an int tensor with shape (num_of_clients, num_of_classes)."""
    for _ in range(0, 10):
        alpha_tensor = tf.fill(num_of_clients, alpha)
        # alpha_tensor = alpha * prior_distrib
        print(alpha_tensor)
        dist = tfp.distributions.Dirichlet(tf.cast(alpha_tensor, tf.float32))
        samples = dist.sample(num_of_classes)
        print(samples)
        # Cast to integer for an integer number of examples per label per client
        int_samples = tf.cast(tf.round(samples * num_of_examples_per_label), tf.int32)
        # int_samples = tf.cast(tf.math.ceil(samples * num_of_examples_per_label), tf.int32)
        int_samples_transpose = tf.transpose(int_samples, [1, 0])
        # print("reduce_sum", tf.reduce_sum(int_samples_transpose, axis=1))
        correctly_generated = tf.reduce_min(tf.reduce_sum(int_samples_transpose, axis=1))
        if tf.cast(correctly_generated, tf.float32) != tf.constant(0.0, tf.float32):
            break
        print("Generated some clients without any examples. Retrying..")

    return int_samples_transpose


def generate_one_class_distrib(num_of_classes, num_of_clients, total_samples):
    per_client_samples = total_samples//num_of_clients
    print("Samples for each client: ", per_client_samples)
    zeros_tensor = tf.zeros([num_of_clients, num_of_classes], tf.int32)

    # 10 client, 60,000 examples, 6,000 examples per class

    label = 0
    for c in range(0, num_of_clients):
        zeros_tensor = tf.tensor_scatter_nd_update(zeros_tensor, updates=[per_client_samples], indices=[[c, label]])

        label = label +1
        if label == num_of_classes:
            label = 0
    print(zeros_tensor)

    return zeros_tensor



def remove_list_from_list(orig_list, to_remove):
    """Remove to_remove list from the orig_list and returns a new list."""
    new_list = []
    for element in orig_list:
        if element not in to_remove:
            new_list.append(element)
    return new_list


def save_dic_as_txt(filename, dic):
    with open(filename, 'w') as file:
        file.write(json.dumps(dic))


if __name__ == '__main__':
    no_repetition = True
    alphas = [0.1]  # -1 generates a homogeneous distrib.
    datasets = ["birds"]  # dataset = ["cifar100", "birds", "cars", "aircrafts"]
    nums_of_clients = [10]
    table_dataset_classes = {"mnist": 10, "cifar10": 10, "cifar100": 100, "birds": 200, "cars": 196, "aircrafts": 100}
    table_num_of_examples_per_label = {"mnist": 5421, "cifar100": 500, "birds": 32, "cars": 41,
                                       "aircrafts": 70, "cifar10": 5000}
    # total example train, cars: 8,144, birds: 5,994, aircrafts: 6,667
    recover_dataset_name = {"birds": "caltech_birds2011"}
    print("Generating dirichlet partitions..")

    for dataset in datasets:
        for alpha in alphas:
            for num_of_clients in nums_of_clients:
                print(f"Generating alpha = {alpha} partitions for {dataset} with {num_of_clients} sites.")

                client_data_dict = {}
                # preparing folder
                folder = os.path.join(
                    "federated_datasets",
                    dataset)
                exist = os.path.exists(folder)

                if not exist:
                    os.makedirs(folder)

                alpha_string = get_string_distribution(alpha)
                folder_path = os.path.join(
                    folder,
                    str(num_of_clients),
                    alpha_string)
                exist = os.path.exists(folder_path)

                if not exist:
                    os.makedirs(folder_path)
                else:
                    shutil.rmtree(folder_path, ignore_errors=True)

                num_of_examples_per_label = table_num_of_examples_per_label[dataset]
                num_of_classes = table_dataset_classes[dataset]

                if alpha < 0:  # iid
                    smpls = generate_dirichlet_samples(num_of_classes=num_of_classes, alpha=10000000,
                                                   num_of_clients=num_of_clients,
                                                   num_of_examples_per_label=num_of_examples_per_label)
                elif alpha > 0:
                    smpls = generate_dirichlet_samples(num_of_classes=num_of_classes,
                                                       alpha=alpha,
                                                       num_of_clients=num_of_clients,
                                                       num_of_examples_per_label=num_of_examples_per_label)
                else:
                    smpls = generate_one_class_distrib(num_of_classes=num_of_classes,
                                                       num_of_clients=num_of_clients,
                                                       total_samples=num_of_examples_per_label*num_of_classes)

                if dataset in ["mnist"]:
                    (x_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()
                elif dataset in ["cifar100"]:
                    (x_train, y_train), (_, _) = tf.keras.datasets.cifar100.load_data()
                elif dataset in ["cifar10"]:
                    (x_train, y_train), (_, _) = tf.keras.datasets.cifar10.load_data()
                elif dataset in ["birds"]:
                    dataset_name = recover_dataset_name[dataset]
                    train_ds = tfds.load(dataset_name, split='train',
                                         shuffle_files=False, as_supervised=True)

                    y_train = train_ds.map(lambda x, y: y)
                    y_train_as_list = list(y_train)
                    y_train_as_list_of_np = [t.numpy() for t in y_train_as_list]
                    y_train = np.array(y_train_as_list_of_np)

                    x_train = train_ds.map(lambda x, y: x)
                    x_train_as_list = list(x_train)
                    x_train_as_numpy = [tf.image.resize(t, size=(224, 224)).numpy() for t in x_train_as_list]
                    x_train = np.array(x_train_as_numpy)

                indexes_of_labels = list([list([]) for _ in range(0, num_of_classes)])

                j = 0
                print(y_train)
                for label in y_train:
                    if dataset in ["mnist"]:
                        indexes_of_labels[label].append(j)
                    elif dataset in ["cifar100", "cifar10", "birds"]:
                        indexes_of_labels[label.item()].append(j)
                    j = j + 1

                for i in indexes_of_labels:
                    print(len(i))

                c = 0
                indexes_of_labels_backup = [element for element in indexes_of_labels]
                smpls = smpls.numpy()
                for per_client_sample in smpls:
                    print(f"[Client {c}] Generating dataset..")
                    label = 0

                    list_extracted_all_labels = []

                    for num_of_examples_per_label in per_client_sample:
                        if no_repetition:
                            if len(indexes_of_labels[label]) < num_of_examples_per_label:
                                print(f"label {label} ended")
                                extracted = np.random.choice(indexes_of_labels[label], len(indexes_of_labels[label]), replace=False)
                                smpls[c, label] = smpls[c, label] - len(indexes_of_labels[label])
                            else:
                                extracted = np.random.choice(indexes_of_labels[label], num_of_examples_per_label, replace=False)
                        else:
                            if len(indexes_of_labels[label]) < num_of_examples_per_label:
                                print("[WARNING] Repeated examples!")
                                remained = len(indexes_of_labels[label])
                                extracted_1 = np.random.choice(indexes_of_labels[label], remained, replace=False)
                                indexes_of_labels[label] = indexes_of_labels_backup[label]
                                extracted_2 = np.random.choice(indexes_of_labels[label], num_of_examples_per_label - remained,
                                                               replace=False)
                                extracted = np.concatenate((extracted_1, extracted_2), axis=0)
                            else:
                                extracted = np.random.choice(indexes_of_labels[label], num_of_examples_per_label, replace=False)

                        indexes_of_labels[label] = remove_list_from_list(indexes_of_labels[label], extracted.tolist())

                        for ee in extracted.tolist():
                            list_extracted_all_labels.append(ee)

                        label = label + 1

                    list_extracted_all_labels = list(map(int, list_extracted_all_labels))

                    if dataset not in ["aircrafts"]:
                        numpy_dataset_y = tf.convert_to_tensor(
                            np.asarray(y_train[list_extracted_all_labels]),
                            dtype=tf.int64)
                        # print(type(numpy_dataset_y))
                        numpy_dataset_x = tf.convert_to_tensor(
                            np.asarray(x_train[list_extracted_all_labels]),
                            dtype=tf.uint8)
                        # print(type(numpy_dataset_x))
                        ds = tf.data.Dataset.from_tensor_slices((numpy_dataset_x, numpy_dataset_y))
                        ds = ds.shuffle(buffer_size=4096)

                        tf.data.Dataset.save(ds,
                                                  path=os.path.join(os.path.join(folder_path, "train"),
                                                                    str(c)))

                    # saving the list of image indexes in a dictionary for reproducibility
                    client_data_dict[c] = list_extracted_all_labels
                    c = c + 1

                path = os.path.join(folder_path, "distribution_train.npy")
                np.save(path, smpls)
                smpls_loaded = np.load(path)
                tf.print(smpls_loaded, summarize=-1)
                print("Reduce sum axis label", tf.reduce_sum(smpls_loaded, axis=1))
                print("Reduce sum axis client", tf.reduce_sum(smpls_loaded, axis=0))
                print("Reduce sum ", tf.reduce_sum(smpls_loaded))

                folder_path = os.path.join(
                    "client_data",
                    dataset,
                    "unbalanced",
                )
                alpha_dirichlet_string = get_string_distribution(alpha)
                file_path = os.path.join(
                        folder_path,
                        alpha_dirichlet_string+"_clients"+str(num_of_clients)+".txt"
                    )

                exist = os.path.exists(folder_path)
                if not exist:
                    os.makedirs(folder_path)

                exist = os.path.exists(file_path)
                if exist:
                    os.remove(file_path)

                save_dic_as_txt(file_path, client_data_dict)
