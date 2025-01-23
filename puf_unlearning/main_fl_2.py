import warnings
import os
import shutil
import hydra
import numpy as np

from basics_unlearning.generate_csv_results import compute_yeom_mia
from basics_unlearning.mia_svc import SVC_MIA

import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import tensorflow_datasets as tfds
from omegaconf import DictConfig, OmegaConf

from basics_unlearning.dataset import (
    load_client_datasets_from_files,
    load_selected_client_statistics,
    get_string_distribution, load_selected_clients_statistics,
    normalize_img, expand_dims, element_norm_cifar100, PaddedRandomCrop,
    get_preprocess_fn
)
from basics_unlearning.utility import create_model, get_test_dataset, preprocess_ds, \
    compute_kl_div, preprocess_ds_test, list_clients_to_string, save_line_to_file

physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    assert tf.config.experimental.get_memory_growth(physical_devices[0])
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass

def find_last_checkpoint(dir):
    exist = os.path.exists(dir)
    if not exist:
        return -1
    else:
        filenames = os.listdir(dir)  # get all files' and folders' names in the current directory

    dirnames = []
    for filename in filenames:  # loop through all the files and folders
        if os.path.isdir(os.path.join(dir, filename)):  # check whether the current object is a folder or not
            filename = int(filename.replace("R_", ""))
            dirnames.append(filename)
    if not dirnames:
        return -1
    last_round_in_checkpoints = max(dirnames)
    print(f"Last checkpoint found in {dir} is from round {last_round_in_checkpoints}")
    return last_round_in_checkpoints


@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:
    print("[Start Simulation]")
    # Print parsed config
    print(OmegaConf.to_yaml(cfg))
    SAVE_ROUND_CLIENTS = 200
    dataset = cfg.dataset
    model = cfg.model
    alpha = cfg.alpha
    local_batch_size = cfg.local_batch_size
    total_clients = cfg.total_clients
    total_rounds = cfg.total_rounds
    active_clients = cfg.active_clients
    local_epochs = cfg.local_epochs
    lr_decay = cfg.lr_decay
    learning_rate = cfg.learning_rate
    resume_training = cfg.resume_training  # resume training after unlearning
    retraining = cfg.retraining  # retrain baseline
    restart_training = cfg.restart_training  # restart training from checkpoint
    seed  = cfg.seed
    if dataset in ["cifar100"] and model in ["MitB0"]:
        dataset = "cifar100-transformer"
        SAVE_ROUND_CLIENTS = 50
    if dataset in ["mnist", "cifar10"]:
        total_classes = 10
    elif dataset in ["cifar100", "cifar100-transformer"]:
        total_classes = 100
    else:
        total_classes = 20

    resumed_round = 0
    unlearned_cid = list(cfg.unlearned_cid)
    unlearned_cid_string = list_clients_to_string(unlearned_cid)
    save_checkpoint = "save_all"  ## "save_last"
    first_time = True
    checkpoint_frequency = 1

    alpha_dirichlet_string = get_string_distribution(alpha)

    # loading test dataset
    ds_test = get_test_dataset(dataset)

    # server model
    server_model = create_model(dataset=dataset, total_classes=total_classes)
    model_string = model
    config_dir = os.path.join(f"{dataset}_{alpha_dirichlet_string}",
                              f"{model_string}_K{total_clients}_C{active_clients}_epochs{local_epochs}_seed{seed}"
                              )
    model_checkpoint_base_dir = os.path.join("model_checkpoints", config_dir,
                                             "checkpoints")

    best_round = find_last_checkpoint(model_checkpoint_base_dir)

    if resume_training:
        # creating config string for resume training
        algorithm = cfg.resuming_after_unlearning.algorithm
        frozen_layers = cfg.resuming_after_unlearning.frozen_layers
        learning_rate_unlearning = cfg.resuming_after_unlearning.unlearning_lr
        epochs_unlearning = cfg.resuming_after_unlearning.unlearning_epochs
        # unlearning_config = f"fl_{frozen_layers}_lr{learning_rate_unlearning}_e_{epochs_unlearning}"
        if algorithm not in ["projected_ga"]:
            unlearning_config = f"fl_{frozen_layers}_lr{learning_rate_unlearning}_e_{epochs_unlearning}"
        else:
            early_stopping_threshold_pga = cfg.resuming_after_unlearning.early_stopping_threshold
            unlearning_config = f"fl_{frozen_layers}_lr{learning_rate_unlearning}_e_{epochs_unlearning}_threshold_{early_stopping_threshold_pga}"

    if restart_training:
        if resume_training:
            model_checkpoint_base_dir = os.path.join("model_checkpoints_resumed",
                                                config_dir,
                                                algorithm,
                                                unlearning_config,
                                                "client" + unlearned_cid_string,
                                                "checkpoints")
            print(f"[Server] Loading checkpoint at {model_checkpoint_base_dir} ")

            last_round = find_last_checkpoint(model_checkpoint_base_dir)
            if last_round > 0:
                model_checkpoint_dir = os.path.join(model_checkpoint_base_dir,
                                                    f"R_{last_round}")
                server_model.load_weights(model_checkpoint_dir)
                resumed_round = last_round
            else:
                if algorithm in ["pseudo_gradient_ascent", "pseudo_gradient_ascent_single"]:
                    model_checkpoint_dir = os.path.join("model_checkpoints", config_dir, "checkpoints",
                                 f"R_{best_round}")
                else:
                    model_checkpoint_dir = os.path.join("model_checkpoints",
                                                        config_dir,
                                                        algorithm,
                                                        unlearning_config,
                                                        f"R_{best_round}_unlearned_client_{unlearned_cid_string}")
                server_model.load_weights(model_checkpoint_dir)
                resumed_round = best_round
        elif retraining:
            model_checkpoint_base_dir = os.path.join("model_checkpoints_retrained",
                                                config_dir,
                                                "client" + unlearned_cid_string,
                                                "checkpoints")
            print(f"[Server] Loading checkpoint at {model_checkpoint_base_dir} ")
            last_round = find_last_checkpoint(model_checkpoint_base_dir)
            if last_round > 0:
                model_checkpoint_dir = os.path.join(model_checkpoint_base_dir,
                                                    f"R_{last_round}")
                server_model.load_weights(model_checkpoint_dir)
                resumed_round = last_round
            else:
                print("Checkpoint not found. Start from round 0.")
        else:  # continue original training
            model_checkpoint_base_dir = os.path.join("model_checkpoints", config_dir,
                                                     "checkpoints")
            print(f"[Server] Loading checkpoint at {model_checkpoint_base_dir} ")
            last_round = find_last_checkpoint(model_checkpoint_base_dir)
            if last_round > 0:
                model_checkpoint_dir = os.path.join(model_checkpoint_base_dir,
                                                    f"R_{last_round}")
                server_model.load_weights(model_checkpoint_dir)
                resumed_round = last_round
            else:
                print("Checkpoint not found. Start from round 0.")

    server_model.summary()
    if model in ["ResNet18"]:
        server_model.compile(optimizer='sgd',
                             loss=tf.keras.losses.SparseCategoricalCrossentropy(
                                 from_logits=True),
                             metrics=['accuracy'])
    elif model in ["MitB0"]:
        optimizer = tf.keras.optimizers.experimental.AdamW()
        server_model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True,
            ),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
        )


    def client_update(cid, local_model, model, t=1, verbose=2, epochs=1, retraining=False):
        amount_of_local_examples = load_selected_client_statistics(
            int(cid),
            total_clients=total_clients,
            alpha=alpha,
            dataset=dataset,
        )

        ds_train_client = load_client_datasets_from_files(
            selected_client=int(cid),
            dataset=dataset,
            total_clients=total_clients,
            alpha=alpha,
        )
        ds_train_client_unbatched = preprocess_ds(ds_train_client, dataset)
        ds_train_client = ds_train_client_unbatched.batch(local_batch_size,
                                                          drop_remainder=False)
        ds_train_client = ds_train_client.prefetch(tf.data.AUTOTUNE)
        if model in ["ResNet18"]:
            optimizer = tf.keras.optimizers.experimental.SGD(
                learning_rate=learning_rate * (lr_decay ** (t - 1)), # lr=0.01, 0.1 (mnist, cifar)
                weight_decay=None if dataset in ["mnist"] else 1e-3)
            local_model.compile(optimizer=optimizer,
                                loss=tf.keras.losses.SparseCategoricalCrossentropy(
                                    from_logits=True),
                                metrics=['accuracy'])
        elif model in ["MitB0"]:
            clipnorm = None
            l2_weight_decay = 1e-3
            optimizer = tf.keras.optimizers.experimental.AdamW(
                learning_rate=learning_rate * (lr_decay ** (t - 1)),
                clipnorm=clipnorm,
                weight_decay=l2_weight_decay,
            )
            local_model.compile(
                optimizer=optimizer,
                loss=tf.keras.losses.SparseCategoricalCrossentropy(
                    from_logits=True,
                ),
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
            )

        # Local training
        print(f"[Client {cid}] Local training..")
        local_model.fit(
            ds_train_client,
            epochs=epochs,
            validation_data=ds_test,
            verbose=verbose
        )
        # collect per-class mean output

        per_class_mean_output = np.zeros([total_classes, total_classes])

        # for projected ga
        if t == SAVE_ROUND_CLIENTS:
            if not retraining:
                print("Saving client models for PGA")
                location = os.path.join(model_checkpoint_dir, f"client_models_R{t}", f"client{cid}")
                print(f"[Client {cid}] Saving model checkpoint at {location}")
                exist = os.path.exists(location)
                if not exist:
                    os.makedirs(location)

                local_model.save(location)

        return local_model.get_weights(), amount_of_local_examples, per_class_mean_output

    config_dir = os.path.join(f"{dataset}_{alpha_dirichlet_string}",
                              f"{model_string}_K{total_clients}_C{active_clients}_epochs{local_epochs}_seed{seed}")
    if retraining:
        log_dir = os.path.join("logs_retrained", config_dir)
        log_dir_accuracy = os.path.join(log_dir, "client"+unlearned_cid_string, "accuracy")
        log_dir_loss = os.path.join(log_dir, "client"+unlearned_cid_string, "loss")
        test_summary_writer_acc = tf.summary.create_file_writer(log_dir_accuracy)
        test_summary_writer_loss = tf.summary.create_file_writer(log_dir_loss)

        log_dir_kl_div = os.path.join(log_dir, "client"+unlearned_cid_string, "kl_train")
        log_dir_acc_train = os.path.join(log_dir, "client"+unlearned_cid_string, "acc_train")
        log_dir_loss_train = os.path.join(log_dir, "client"+unlearned_cid_string, "loss_train")
        train_summary_writer_loss = tf.summary.create_file_writer(log_dir_loss_train)
        train_summary_writer_kl_div = tf.summary.create_file_writer(log_dir_kl_div)
        train_summary_writer_acc = tf.summary.create_file_writer(log_dir_acc_train)
        model_checkpoint_dir = os.path.join("model_checkpoints_retrained", config_dir, "client"+unlearned_cid_string)

    elif resume_training:
        # log_dir = os.path.join("logs_resumed", config_dir, algorithm)
        log_dir = os.path.join("logs_resumed", config_dir, algorithm, unlearning_config)
        log_dir_accuracy = os.path.join(log_dir, "client"+unlearned_cid_string, "accuracy")
        log_dir_loss = os.path.join(log_dir, "client"+unlearned_cid_string, "loss")
        test_summary_writer_acc = tf.summary.create_file_writer(log_dir_accuracy)
        test_summary_writer_loss = tf.summary.create_file_writer(log_dir_loss)

        log_dir_kl_div = os.path.join(log_dir, "client"+unlearned_cid_string, "kl_train")
        log_dir_acc_train = os.path.join(log_dir, "client"+unlearned_cid_string, "acc_train")
        log_dir_loss_train = os.path.join(log_dir, "client"+unlearned_cid_string, "loss_train")
        train_summary_writer_loss = tf.summary.create_file_writer(log_dir_loss_train)
        train_summary_writer_kl_div = tf.summary.create_file_writer(log_dir_kl_div)
        train_summary_writer_acc = tf.summary.create_file_writer(log_dir_acc_train)

        model_checkpoint_dir = os.path.join("model_checkpoints_resumed",
                                                 config_dir,
                                                 algorithm,
                                                 unlearning_config,
                                                 "client" + unlearned_cid_string)

    else:
        model_checkpoint_dir = os.path.join("model_checkpoints", config_dir)
        log_dir = os.path.join("logs_original", config_dir)
        log_dir_accuracy = os.path.join(log_dir, "accuracy")
        log_dir_loss = os.path.join(log_dir, "loss")
        test_summary_writer_acc = tf.summary.create_file_writer(log_dir_accuracy)
        test_summary_writer_loss = tf.summary.create_file_writer(log_dir_loss)

    max_accuracy = 0

    test_loss, test_acc = server_model.evaluate(ds_test, verbose=2)

    early_stop_recovery = False
    if resume_training:
        # -- retrained acc
        if algorithm not in ["natural"]:
            client_dir_r = os.path.join(f"client{unlearned_cid_string}", "checkpoints")

            last_checkpoint_retrained = find_last_checkpoint(
                os.path.join("model_checkpoints_retrained", config_dir, client_dir_r))

            model_checkpoint_dir_retrained = os.path.join("model_checkpoints_retrained",
                                                          config_dir,
                                                          client_dir_r,
                                                          f"R_{last_checkpoint_retrained}")

            model_retrained = create_model(dataset=dataset,
                                           total_classes=total_classes)
            model_retrained.load_weights(model_checkpoint_dir_retrained)
            if dataset not in ["cifar100-transformer"]:
                model_retrained.compile(optimizer='sgd',
                                        loss=tf.keras.losses.SparseCategoricalCrossentropy(
                                            from_logits=True),
                                        metrics=['accuracy'])
            else:
                optimizer = tf.keras.optimizers.experimental.AdamW()
                model_retrained.compile(
                    optimizer=optimizer,
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(
                        from_logits=True,
                    ),
                    metrics=[tf.keras.metrics.SparseCategoricalAccuracy(
                        name="accuracy")],
                )
            print("----- Retrained model -----")
            print("Test")
            _, test_acc_retrained = model_retrained.evaluate(ds_test, verbose=2)
            # retrained_computed = False
        # ----------------

        if algorithm not in ["natural"] and test_acc >= test_acc_retrained:
            early_stop_recovery = True
            if resumed_round == SAVE_ROUND_CLIENTS:
                if resume_training:
                    if algorithm not in ["pseudo_gradient_ascent_single", "pseudo_gradient_ascent"]:
                        dir = os.path.join(model_checkpoint_dir, "checkpoints")
                        shutil.rmtree(dir, ignore_errors=True)
                        print("Saving checkpoint global model......")
                        server_model.save(
                            os.path.join(model_checkpoint_dir, "checkpoints", f"R_{resumed_round}"))
                        print("[Info] Already recovered. Not performing any round.")
                    else:
                        early_stop_recovery = False

    if not early_stop_recovery:
        # retrained_computed = False
        for r in range(resumed_round + 1, resumed_round + total_rounds + 1):
            delta_w_aggregated = tf.nest.map_structure(lambda a, b: a - b,
                                                       server_model.get_weights(),
                                                       server_model.get_weights())

            if resume_training:
                if active_clients == 1.0:
                    m = max(total_clients * active_clients, 1) - len(unlearned_cid)
                elif r == SAVE_ROUND_CLIENTS+1 and algorithm in ["pseudo_gradient_ascent"]:
                    m = max(total_clients * active_clients, 1) - len(unlearned_cid)
                else:
                    m = max(total_clients * active_clients, 1)
            elif retraining:
                if active_clients == 1.0:
                    m = max(total_clients * active_clients, 1) - len(unlearned_cid)
                else:
                    m = max(total_clients * active_clients, 1)
            else:
                m = max(total_clients * active_clients, 1)

            client_list = list(range(total_clients))
            if resume_training or retraining:
                for u in unlearned_cid:
                    client_list.remove(u)

            print(client_list)
            sampled_clients = np.random.choice(
                np.asarray(client_list, np.int32),
                size=int(m),
                replace=False)

            print(f"[Server] Round {r} -- Selected clients: {sampled_clients}")

            selected_client_examples = load_selected_clients_statistics(
                selected_clients=sampled_clients, alpha=alpha, dataset=dataset,
                total_clients=total_clients)

            unl_client_examples = load_selected_clients_statistics(
                        selected_clients=unlearned_cid, alpha=alpha, dataset=dataset,
                        total_clients=total_clients)

            if r == SAVE_ROUND_CLIENTS+1:
                if algorithm in ["pseudo_gradient_ascent"]:
                    total_examples = np.sum(selected_client_examples) + np.sum(unl_client_examples)
                elif algorithm in ["pseudo_gradient_ascent_single"]:
                    total_examples = np.sum(unl_client_examples)
                else:
                    total_examples = np.sum(selected_client_examples)
            else:
                total_examples = np.sum(selected_client_examples)

            print("Total examples ", total_examples)
            print("Local examples selected clients ", selected_client_examples)

            global_weights = server_model.get_weights()
            # aggregated_mean_output = np.zeros([total_classes, total_classes], np.float32)
            if r != SAVE_ROUND_CLIENTS+1 or algorithm not in ["pseudo_gradient_ascent_single"]:
                for k in sampled_clients:
                    client_model = server_model
                    client_model.set_weights(global_weights)
                    client_weights, local_samples, pc_mean_output = client_update(k,
                                                                                  client_model,
                                                                                  model=model,
                                                                                  t=r,
                                                                                  retraining=retraining)

                    # FedAvg aggregation
                    delta_w_local = tf.nest.map_structure(lambda a, b: a - b,
                                                          client_model.get_weights(),
                                                          global_weights,
                                                          )
                    delta_w_aggregated = tf.nest.map_structure(
                        lambda a, b: a + b * (local_samples / total_examples),
                        delta_w_aggregated,
                        delta_w_local)

            if r == SAVE_ROUND_CLIENTS+1 and algorithm in ["pseudo_gradient_ascent", "pseudo_gradient_ascent_single"]:
                print("... Pseudo gradient ascent")
                for u in unlearned_cid:
                    client_model = server_model
                    client_model.set_weights(global_weights)
                    client_weights, local_samples, pc_mean_output = client_update(
                            u,
                            client_model,
                            model=model,
                            t=r,
                            epochs=epochs_unlearning,
                            retraining=retraining)
                    # FedAvg aggregation
                    delta_w_local = tf.nest.map_structure(lambda a, b: a - b,
                                                              client_model.get_weights(),
                                                              global_weights,
                                                              )
                    delta_w_local = tf.nest.map_structure(lambda a: -(learning_rate_unlearning) * a,
                                                              delta_w_local,
                                                              )
                    # print("delta_w_local ", len(delta_w_local))
                    # print("local_samples ", local_samples)
                    # print("total_examples ", total_examples)
                    # print("delta_w_local ", len(delta_w_local))
                    delta_w_aggregated = tf.nest.map_structure(
                        lambda a, b: a + b * (local_samples / total_examples),
                        delta_w_aggregated,
                        delta_w_local)
                    # print("delta_w_aggregated ", len(delta_w_aggregated))


            # apply the aggregated updates
            # --> sgd with 1.0 lr
            new_global_weights = tf.nest.map_structure(lambda a, b: a + b,
                                                       global_weights,
                                                       delta_w_aggregated)
            server_model.set_weights(new_global_weights)

            # logging global model performance
            test_loss, test_acc = server_model.evaluate(ds_test, verbose=2)
            print(f'[Server] Round {r} -- Test accuracy: {test_acc}')

            if not retraining and not resume_training:
                print("Logging selected clients for this round to file...")
                config = f"{dataset}_{alpha_dirichlet_string}_{model_string}_K{total_clients}_C{active_clients}_epochs{local_epochs}_seed{seed}"
                file_name = "original_model_" + config
                list_selected_clients = ','.join(map(str, sampled_clients.tolist()))
                save_line_to_file("logging", file_name+".txt",
                                f"R_{r}:{list_selected_clients}")

            if save_checkpoint == "save_last":
                dir = os.path.join(model_checkpoint_dir, "last_checkpoint")
                if r == (resumed_round + total_rounds):
                    exist = os.path.exists(dir)
                    if not exist:
                        os.makedirs(dir)
                    else:
                        shutil.rmtree(dir, ignore_errors=True)

                    server_model.save(os.path.join(model_checkpoint_dir, "last_checkpoint", f"R_{r}"))
            elif save_checkpoint == "save_all":
                print("Saving checkpoint...")
                dir = os.path.join(model_checkpoint_dir, "checkpoints")
                if resume_training:  # need for all the checkpoints for the analysis
                    checkpoint_frequency = 1
                if r % checkpoint_frequency == 0:
                    if first_time:
                        exist = os.path.exists(dir)
                        if not exist:
                            os.makedirs(dir)
                        else:
                            if not resume_training:
                                shutil.rmtree(dir, ignore_errors=True)
                        first_time = False
                    if resume_training and algorithm not in ["natural"]:
                        shutil.rmtree(dir, ignore_errors=True)
                    print("Saving checkpoint global model......")
                    server_model.save(os.path.join(model_checkpoint_dir, "checkpoints", f"R_{r}"))

            if retraining:
                first_time_ds = True
                for u in unlearned_cid:
                    ds = load_client_datasets_from_files(
                        selected_client=int(u),
                        dataset=dataset,
                        total_clients=total_clients,
                        alpha=alpha,
                    )
                    if first_time_ds:
                        ds_train_client = ds
                    ds_train_client = ds_train_client.concatenate(ds)
                    first_time_ds = False


                ds_train_client = preprocess_ds_test(ds_train_client, dataset)
                ds_train_client = ds_train_client.batch(local_batch_size,
                                                                  drop_remainder=False)
                ds_train_client = ds_train_client.prefetch(tf.data.AUTOTUNE)

                print("[Server] Train acc ")
                loss, acc = server_model.evaluate(ds_train_client, verbose=2)
            elif resume_training:
                first_time_ds = True
                for u in unlearned_cid:
                    ds = load_client_datasets_from_files(
                        selected_client=int(u),
                        dataset=dataset,
                        total_clients=total_clients,
                        alpha=alpha,
                    )
                    if first_time_ds:
                        ds_train_client = ds
                    ds_train_client = ds_train_client.concatenate(ds)
                    first_time_ds = False

                ds_train_client = preprocess_ds_test(ds_train_client, dataset)
                ds_train_client = ds_train_client.batch(local_batch_size,
                                                        drop_remainder=False)
                ds_train_client = ds_train_client.prefetch(tf.data.AUTOTUNE)
                print("[Retrained] Train")
                _, train_acc_retrained = model_retrained.evaluate(ds_train_client, verbose=2)
                print("[Server - Resumed] Train acc ")
                loss, acc = server_model.evaluate(ds_train_client, verbose=2)

                if algorithm not in ["natural"] and test_acc >= test_acc_retrained:
                    # print("----- Reached test acc of retrained model -----")
                    # print("Evaluating MIA success rates...")
                    # n = 10000
                    #
                    # first_time_r = True
                    # for i in range(total_clients):
                    #     if i not in unlearned_cid:
                    #         ds = load_client_datasets_from_files(
                    #             selected_client=i,
                    #             dataset=dataset,
                    #             total_clients=total_clients,
                    #             alpha=alpha,
                    #         )
                    #         if first_time_r:
                    #             ds_retain = ds
                    #             first_time_r = False
                    #         else:
                    #             ds_retain = ds.concatenate(ds_retain)
                    #
                    # ds_retain = ds_retain.shuffle(60000).take(n)
                    # ds_retain = preprocess_ds_test(ds_retain, dataset)
                    # ds_retain = ds_retain.batch(128, drop_remainder=False)
                    # ds_retain = ds_retain.cache()
                    # ds_retain = ds_retain.prefetch(tf.data.AUTOTUNE)
                    #
                    # # --- MIA Yeom et al.
                    # yeom_mia_retrained = compute_yeom_mia(model_retrained,
                    #                             train_data=ds_retain,
                    #                             forget_data=ds_train_client)
                    #
                    # # --- MIA Efficiency
                    # results_mia = SVC_MIA(shadow_train=ds_retain,
                    #                       shadow_test=ds_test,
                    #                       target_train=ds_train_client,
                    #                       target_test=None,
                    #                       model=model_retrained)
                    #
                    # mia_retrained = results_mia["confidence"] * 100
                    #
                    # # --- MIA Yeom et al. (approximated)
                    # yeom_mia = compute_yeom_mia(server_model,
                    #                             train_data=ds_retain,
                    #                             forget_data=ds_train_client)
                    #
                    # # --- MIA Efficiency
                    # results_mia = SVC_MIA(shadow_train=ds_retain,
                    #                       shadow_test=ds_test,
                    #                       target_train=ds_train_client,
                    #                       target_test=None,
                    #                       model=server_model)
                    #
                    # mia = results_mia["confidence"] * 100
                    # print(f"[Retrained] yeom_mia: {yeom_mia_retrained}, song_mia: {mia_retrained}")
                    # print(f"[Unlearned] yeom_mia: {yeom_mia}, song_mia: {mia}")
                    break


if __name__ == "__main__":
    main()
