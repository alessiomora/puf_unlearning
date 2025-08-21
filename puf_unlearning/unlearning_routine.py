import os
import hydra
import warnings
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from omegaconf import DictConfig, OmegaConf
import matplotlib as plt
import seaborn as sns
import pandas as pd
from puf_unlearning.dataset import load_client_datasets_from_files, normalize_img, \
    get_string_distribution, expand_dims, load_selected_client_statistics, \
    get_add_unlearning_label_fn, load_label_distribution_selected_client
from puf_unlearning.main_fl_2 import find_last_checkpoint
from puf_unlearning.model import create_cnn_model
from puf_unlearning.model_kd_div import ModelKLDiv, ModelKLDivAdaptive, \
    ModelKLDivAdaptiveSoftmax, ModelCompetentIncompetentTeacher, ModelKLDivSoftmaxZero, \
    ModelKLDivLogitMin, ModelNoT
from puf_unlearning.model_projected_ga import DistanceEarlyStopping, get_distance, custom_train_loop
from puf_unlearning.utility import draw_and_save_heatmap, compute_kl_div, \
    create_model, get_test_dataset, preprocess_ds, preprocess_ds_test
from puf_unlearning.ModelSmoothie import FedSmoothieModel, ModelRandomPenalized, \
    UnlearningModelRandomLabel
import json

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
tf.get_logger().setLevel(5)


def save_dic_as_txt(filename, dic, what):
    with open(filename, 'a') as file:
        file.write(json.dumps(dic))
        file.write("\n")


class VirtualTeacher(tf.keras.Model):
    """ A virtual teacher that outputs fixed predictions, i.e.,
    1/num_classes for each class."""

    def __init__(self, num_classes, config="fixed", mean_per_class_prob=tf.zeros([])):
        super().__init__()
        self.config = config
        self.num_classes = num_classes
        self.mean_per_class_prob = mean_per_class_prob

    def call(self, inputs):
        x, y = inputs
        if self.config == "fixed":
            # tf.print(tf.shape(y))
            output = tf.fill([tf.shape(x)[0], self.num_classes],
                             1.0 / self.num_classes)
        elif self.config == "not-true-forgetting":
            output = tf.fill([tf.shape(x)[0], self.num_classes],
                             1.0 / (self.num_classes - 1))
            range_idx = tf.expand_dims(tf.range(0, tf.shape(y)[0]), -1)
            # tf.print(range_idx)
            # tf.print(y)
            idx = tf.concat([tf.cast(range_idx, tf.int64), tf.expand_dims(y, -1)], axis=1)
            output = tf.tensor_scatter_nd_update(output,
                                                 indices=idx,
                                                 updates=tf.zeros(tf.shape(x)[0])
                                                 )
        elif self.config == "mean_per_classs_prob":
            output = tf.gather(self.mean_per_class_prob, y)
            range_idx = tf.expand_dims(tf.range(0, tf.shape(y)[0]), -1)
            idx = tf.concat([tf.cast(range_idx, tf.int64), tf.expand_dims(y, -1)], axis=1)
            output = tf.tensor_scatter_nd_update(tf.cast(output, tf.float32),
                                                 indices=idx,
                                                 updates=tf.zeros(tf.shape(x)[0])
                                                 )
            output = tf.nn.softmax(output, axis=1)
        return output


@hydra.main(config_path="conf", config_name="unlearning", version_base=None)
def main(cfg: DictConfig) -> None:
    print("[Start Simulation]")
    # Print parsed config
    print(OmegaConf.to_yaml(cfg))

    dataset = cfg.dataset
    alpha = cfg.alpha
    algorithm = cfg.algorithm
    seed  = cfg.seed
    model = cfg.model
    alpha_dirichlet_string = get_string_distribution(alpha)

    if dataset in ["mnist", "cifar10"]:
        total_classes = 10
    elif dataset in ["cifar100"]:
        total_classes = 100
    else:
        total_classes = 200

    local_batch_size = cfg.local_batch_size
    total_clients = cfg.total_clients
    alpha = cfg.alpha
    active_clients = cfg.active_clients
    local_epochs = cfg.local_epochs
    frozen_layers = cfg.frozen_layers
    epochs_unlearning = cfg.unlearning_epochs
    learning_rate_unlearning = cfg.learning_rate_unlearning
    early_stopping_threshold_pga = cfg.projected_ga.early_stopping_threshold
    if dataset in ["cifar100", "birds"] and model in ["MitB0"]:
        dataset = f"{dataset}-transformer"

    model_string = model

    config_dir = os.path.join(f"{dataset}_{alpha_dirichlet_string}",
                              f"{model_string}_K{total_clients}_C{active_clients}_epochs{local_epochs}_seed{seed}"
                              )
    last_checkpoint_retrained = find_last_checkpoint(
        os.path.join("model_checkpoints", config_dir, "checkpoints"))

    model_checkpoint_dir = os.path.join("model_checkpoints", config_dir, "checkpoints", f"R_{last_checkpoint_retrained}")

    def load_reference_model_for_pga(global_model, unlearning_client, config_dir, dataset, total_classes, total_clients):
        last_checkpoint_retrained = find_last_checkpoint(
            os.path.join("model_checkpoints", config_dir, "checkpoints"))
        location = os.path.join("model_checkpoints", config_dir, f"client_models_R{last_checkpoint_retrained}",
                                f"client{unlearning_client}")
        unl_client_model = create_model(dataset=dataset, total_classes=total_classes)
        unl_client_model.load_weights(location)
        unl_client_weights = unl_client_model.get_weights()
        global_weights = global_model.get_weights()
        n = total_clients

        pga_ref_model_weights = tf.nest.map_structure(lambda a, b: 1/(n-1) * (n*a - b),
                                                   global_weights,
                                                   unl_client_weights)

        pga_ref_model = create_model(dataset=dataset, total_classes=total_classes)
        pga_ref_model.set_weights(pga_ref_model_weights)
        return pga_ref_model, unl_client_model

    # for MIA
    # if dataset in ["birds-transformer", "birds"]:
    #     n = 5794
    # else:
    #     n = 10000

    ds_test = get_test_dataset(dataset)

    server_model = create_model(dataset=dataset, total_classes=total_classes)
    if dataset in ["mnist"]:
        saved_checkpoint = tf.keras.saving.load_model(model_checkpoint_dir)
        original_weights = saved_checkpoint.get_weights()
        # server_model = create_cnn_model()
    else:
        server_model.load_weights(model_checkpoint_dir)
        original_weights = server_model.get_weights()

    # original_model = create_cnn_model()
    original_model = create_model(dataset=dataset, total_classes=total_classes)
    original_model.set_weights(original_weights)
    original_model.compile(optimizer=tf.keras.optimizers.experimental.SGD(),
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(
                               from_logits=True),
                           metrics=['accuracy'])
    list_dict =[]
    for cid in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
    # for cid in [0, 1, 3, 4, 5, 6, 7, 8, 9]:
    # for cid in [0, 1, 2]:
    # for cid in [4]:
        print(f"-------- [Client {cid}] --------")
        # loading client u's data
        client_train_ds = load_client_datasets_from_files(
            selected_client=int(cid),
            dataset=dataset,
            total_clients=total_clients,
            alpha=alpha,
        )
        client_label_distribution = load_label_distribution_selected_client(
            selected_client=int(cid),
            dataset=dataset,
            total_clients=total_clients,
            alpha=alpha,
        )
        print(client_label_distribution)

        client_train_ds = preprocess_ds_test(client_train_ds, dataset)
        client_train_ds = client_train_ds.batch(local_batch_size,
                                                          drop_remainder=False)
        client_train_ds = client_train_ds.prefetch(tf.data.AUTOTUNE)

        print("[Test Server Model]")

        server_model.set_weights(original_weights)

        # original_model.evaluate(ds_test_batched)
        original_model.evaluate(ds_test, verbose=2)

        # unlearning routine with kl divergence
        # unlearning_model = ModelKLDiv(server_model, virtual_teacher)
        if algorithm == "natural":
            # just to use unlearning_model.model
            # we dont train this
            unlearning_model = ModelKLDivAdaptive(server_model, original_model)
        elif algorithm == "not":
            unlearning_model = ModelNoT(server_model)

            # Get model weights
            layer_weights = unlearning_model.get_weights()
            # print("Before:", layer_weights[0][:5])  # Print first few values
            # print("Before:", layer_weights[1][:5])  # Print first few values


            # Negate the first layer's weights
            layer_weights[0] = -layer_weights[0]
            print("Shape first layer", tf.shape(layer_weights[0]))
            # bias
            # layer_weights[1] = -layer_weights[1]
            # print("Shape first layer", tf.shape(layer_weights[1]))

            # Set the modified weights back to the model
            unlearning_model.set_weights(layer_weights)
        elif algorithm == "random_penalized":
            unlearning_model = ModelRandomPenalized(server_model)
        elif algorithm == "random":
            unlearning_model = UnlearningModelRandomLabel(server_model)
        elif algorithm == "logit":
            unlearning_model = ModelKLDivAdaptive(server_model, original_model, model_type=model_string, dataset=dataset)
        elif algorithm == "logit_min":
            unlearning_model = ModelKLDivLogitMin(server_model, original_model)
        elif algorithm == "softmax":
            unlearning_model = ModelKLDivAdaptiveSoftmax(server_model, original_model, model_type=model_string)
        elif algorithm == "softmax_zero":
            unlearning_model = ModelKLDivSoftmaxZero(server_model, original_model)
        elif algorithm == "projected_ga":
            model_ref, unl_client_model = load_reference_model_for_pga(server_model, cid, config_dir, dataset, total_classes, total_clients)

            dist_ref_random_lst = []
            for _ in range(10):
                FLNet = create_model(dataset=dataset, total_classes=total_classes)
                dist_ref_random_lst.append(get_distance(model_ref, FLNet))

            print(
                f'Mean distance of Reference Model to random: {np.mean(dist_ref_random_lst)}')
            threshold = np.mean(dist_ref_random_lst) / 3.0
            print(f'Radius for model_ref: {threshold}')

            unlearning_model = create_model(dataset=dataset, total_classes=total_classes)
            unlearning_model.set_weights(model_ref.get_weights())

        elif algorithm in ["ci", "ci_balanced"]:
            virtual_teacher = VirtualTeacher(num_classes=total_classes, config="fixed")
            unlearning_model = ModelCompetentIncompetentTeacher(server_model, original_model, virtual_teacher)
        else:  # just equi-distributed probability over classes
            virtual_teacher = VirtualTeacher(num_classes=total_classes, config="fixed")
            unlearning_model = ModelKLDiv(server_model, virtual_teacher)

        # print("---- Unlearning ----")
        # if algorithm not in ["projected_ga"]:
        #     if dataset in ["mnist"]:
        #         unlearning_model.model.conv1.trainable = False
        #         unlearning_model.model.conv2.trainable = False
        #         unlearning_model.model.dense1.trainable = False
        #         unlearning_model.model.dense2.trainable = True
        #     else:
        #         if frozen_layers >= 1:
        #             unlearning_model.model.layer0.trainable = False
        #         if frozen_layers >= 2:
        #             unlearning_model.model.layer1.trainable = False
        #         if frozen_layers >= 3:
        #             unlearning_model.model.layer2.trainable = False
        #         if frozen_layers >= 4:
        #             unlearning_model.model.layer3.trainable = False
        #         if frozen_layers >= 5:
        #             unlearning_model.model.layer4.trainable = False
        #             unlearning_model.model.gap.trainable = True
        #         unlearning_model.model.fully_connected.trainable = True

        if algorithm not in ["projected_ga"]:
            if model in ["ResNet18"]:
                optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_unlearning)
            elif model in ["MitB0"]:
                optimizer = tf.keras.optimizers.experimental.AdamW(learning_rate=learning_rate_unlearning)
            early_stopping_callback = tf.keras.callbacks.EarlyStopping(
                monitor='loss',
                restore_best_weights=True,
                patience=3)
            unlearning_model.compile(optimizer=optimizer,
                                     loss=tf.keras.losses.KLDivergence(),
                                     metrics=['accuracy'])
        else:
            clip_grad = 5.0  # as in the original paper
            distance_threshold_early_stopping = early_stopping_threshold_pga

            if model in ["ResNet18"]:
                optimizer = tf.keras.optimizers.SGD(
                    learning_rate=learning_rate_unlearning,
                    momentum=0.9,
                    clipnorm=clip_grad)

            elif model in ["MitB0"]:
                optimizer = tf.keras.optimizers.experimental.AdamW(
                    learning_rate=learning_rate_unlearning)

            unlearning_model.compile(optimizer=optimizer,
                                     loss=tf.keras.losses.SparseCategoricalCrossentropy(
                                         from_logits=True),
                                     metrics=['accuracy'])


        if algorithm in ["logit", "softmax", "incorrect", "softmax_zero", "logit_min", "random_penalized", "random"]:
            unlearning_model.model.summary(show_trainable=True)
            unlearning_model.fit(client_train_ds,
                                 epochs=epochs_unlearning,
                                 callbacks=[early_stopping_callback])
        if algorithm in ["projected_ga"]:
            custom_train_loop(unlearning_model, unl_client_model = unl_client_model, epochs=epochs_unlearning, optimizer=optimizer, train_dataset=client_train_ds, threshold=threshold, distance_early_stop=distance_threshold_early_stopping, model_ref=model_ref)
        elif algorithm in ["natural", "not"]:
            print("Do nothing..")

        elif algorithm in ["ci", "ci_balanced"]:
            # optimizer = tf.keras.optimizers.experimental.SGD(
            #     learning_rate=learning_rate_unlearning)  # try 0.01 for ci
            # early_stopping_callback = tf.keras.callbacks.EarlyStopping(
            #     monitor='loss',
            #     restore_best_weights=True,
            #     patience=3)
            # unlearning_model.compile(optimizer=optimizer,
            #                          loss=tf.keras.losses.KLDivergence(),
            #                          metrics=['accuracy'])
            client_train_ds = load_client_datasets_from_files(
                selected_client=int(cid),
                dataset=dataset,
                total_clients=total_clients,
                alpha=alpha,
            )
            train_ds_cardinality = load_selected_client_statistics(
                selected_client=int(cid),
                dataset=dataset,
                total_clients=total_clients,
                alpha=alpha,
            )
            print(f"[Competent-Incompetent] Taking {train_ds_cardinality} samples...")
            client_train_ds = preprocess_ds_test(client_train_ds, dataset)
            client_train_ds = client_train_ds.map(
                get_add_unlearning_label_fn(dataset, 1)
            )

            first_time = True
            for i in range(total_clients):
                if i != cid:
                    ds = load_client_datasets_from_files(
                        selected_client=i,
                        dataset=dataset,
                        total_clients=total_clients,
                        alpha=alpha,
                    )
                    if first_time:
                        ds_retain = ds
                        first_time = False
                    else:
                        ds_retain = ds.concatenate(ds_retain)

            if algorithm not in ["ci_balanced"]:
                train_ds_cardinality = int((50000 - train_ds_cardinality) *0.3)  # taking 30% as in the original paper

            ds_retain = ds_retain.shuffle(60000).take(train_ds_cardinality)
            ds_retain = preprocess_ds_test(ds_retain, dataset)
            ds_retain = ds_retain.map(
                get_add_unlearning_label_fn(dataset, 0)
            )

            ci_ds = client_train_ds.concatenate(ds_retain)
            ci_ds = ci_ds.shuffle(60000)

            ci_ds = ci_ds.batch(local_batch_size, drop_remainder=False)
            ci_ds = ci_ds.prefetch(tf.data.AUTOTUNE)

            unlearning_model.fit(ci_ds,
                                 epochs=epochs_unlearning,
                                 callbacks=[early_stopping_callback])
        print("--------------------")
        client_train_ds = load_client_datasets_from_files(
            selected_client=int(cid),
            dataset=dataset,
            total_clients=total_clients,
            alpha=alpha,
        )

        client_train_ds = preprocess_ds_test(client_train_ds, dataset)
        client_train_ds = client_train_ds.batch(local_batch_size,
                                                          drop_remainder=False)
        client_train_ds = client_train_ds.prefetch(tf.data.AUTOTUNE)

        print("[Original - Test]")
        _, acc_o_test = original_model.evaluate(ds_test, verbose=2)
        # results_dict[] = acc_o_test
        print("[Original - Train]")
        _, acc_o_train = original_model.evaluate(client_train_ds, verbose=2)
        # results_dict[] = acc_o_train

        print("[Unlearned - Test]")
        if algorithm not in ["natural", "not"]:
            _, acc_u_test = unlearning_model.evaluate(ds_test, verbose=2)
        else:
            acc_u_test = unlearning_model.evaluate(ds_test, verbose=2)
        results_dict = {}
        results_dict["test_acc"] = acc_u_test
        print("[Unlearned - Train]")
        if algorithm not in ["natural", "not"]:
            _, acc_u_train = unlearning_model.evaluate(client_train_ds, verbose=2)
        else:
            acc_u_train = unlearning_model.evaluate(client_train_ds, verbose=2)
        results_dict["train_acc"] = acc_u_train
        if algorithm not in ["projected_ga"]:
            name = f"{algorithm}_fl_{frozen_layers}_lr{learning_rate_unlearning}_e_{epochs_unlearning}"
        else:
            name = f"{algorithm}_fl_{frozen_layers}_lr{learning_rate_unlearning}_e_{epochs_unlearning}_threshold_{early_stopping_threshold_pga}"
        results_dict["name"] = name
        results_dict["client"] = cid
        list_dict.append(results_dict)
        # pred = tf.nn.softmax(original_model.predict(client_train_ds), axis=1)
        # kl_div = compute_kl_div(pred, total_classes)
        # print(f"kl_div (original): {kl_div}")
        # results_dict[] = kl_div

        # pred = tf.nn.softmax(unlearning_model.model.predict(client_train_ds), axis=1)
        # kl_div = compute_kl_div(pred, total_classes)
        # print(f"kl_div (unlearned): {kl_div}")
        # results_dict[]] = kl_div

        print("[Server] Saving checkpoint... ")
        if algorithm not in ["projected_ga"]:
            unlearning_config = f"fl_{frozen_layers}_lr{learning_rate_unlearning}_e_{epochs_unlearning}"
        else:
            unlearning_config = f"fl_{frozen_layers}_lr{learning_rate_unlearning}_e_{epochs_unlearning}_threshold_{early_stopping_threshold_pga}"
        model_checkpoint_dir = os.path.join("model_checkpoints", config_dir, algorithm,
                                            unlearning_config,
                                            f"R_{last_checkpoint_retrained}_unlearned_client_{cid}")
        # model_to_save = create_cnn_model()
        # model_to_save.set_weights(unlearning_model.get_weights())
        #
        # model_to_save.save(model_checkpoint_dir)
        if algorithm not in ["projected_ga"]:
            unlearning_model.model.save(model_checkpoint_dir)
        else:
            unlearning_model.save(model_checkpoint_dir)

    df = pd.DataFrame(list_dict)
    print(df)
    filename = f'results_unlearning_routine.csv'
    path_to_save = os.path.join("results_csv", dataset)
    exist = os.path.exists(path_to_save)
    if not exist:
        os.makedirs(path_to_save)
    df.to_csv(os.path.join(path_to_save, filename), mode='a', header=not os.path.join(path_to_save, filename))


    # epochs_string = " E" + str(unlearning_epochs)
    # # difference_as_npa = np.asarray(difference, dtype=np.float32)
    # draw_and_save_heatmap(acc_unlearned_grid, alpha_dirichlet_string, title="Accuracy (per class)" + epochs_string)
    # draw_and_save_heatmap(acc_original_grid, alpha_dirichlet_string, title="Original Accuracy (per class)" + epochs_string)
    # draw_and_save_heatmap(difference, alpha_dirichlet_string, title="Accuracy Delta Before-After Unlearning (per class)" + epochs_string)
    # draw_and_save_heatmap(results_dict[_dirichlet_string, title="Accuracy" + epochs_string, mode="all_class")

if __name__ == "__main__":
    main()
