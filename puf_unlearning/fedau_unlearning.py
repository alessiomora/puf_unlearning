import os
import hydra
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf
from puf_unlearning.dataset import get_string_distribution, \
    load_client_datasets_from_files
from puf_unlearning.utility import get_test_dataset, create_model, preprocess_ds, \
    preprocess_ds_test, list_clients_to_string


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


# --- MAIN ---

@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    #  build base config
    dataset = cfg.dataset
    model = cfg.model
    alpha = cfg.alpha
    local_batch_size = cfg.local_batch_size
    total_clients = cfg.total_clients
    active_clients = cfg.active_clients
    local_epochs = cfg.local_epochs
    lr_decay = cfg.lr_decay
    sample_unlearning = cfg.sample_unlearning
    sample_unl = "_sample_unl" if sample_unlearning else ""
    learning_rate = cfg.fedau.learning_rate
    seed = cfg.seed
    epochs_to_train_w_a = cfg.fedau.epochs
    alpha_fed_au = 0.04
    l2_weight_decay = 1e-3
    # cids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # cids = [[0], [1], [2], ]
    cids = [[1]]
    # cids = [[0, 1], [2, 3], [4, 5]]
    model_string = model

    if dataset in ["cifar100", "birds"] and model_string in ["MitB0"]:
        dataset = f"{dataset}-transformer"
    if dataset in ["mnist", "cifar10"]:
        total_classes = 10
    elif dataset in ["cifar100", "cifar100-transformer"]:
        total_classes = 100
    else:
        total_classes = 200

    for cid in cids:
        alpha_dirichlet_string = get_string_distribution(alpha)
        unlearned_cid = list(cid)
        unlearned_cid_string = list_clients_to_string(unlearned_cid)
        print(f"[Unlearning Clients: {unlearned_cid_string}]")
        config_dir = os.path.join(f"{dataset}_{alpha_dirichlet_string}",
                                  f"{model_string}_K{total_clients}_C{active_clients}_epochs{local_epochs}_seed{seed}"
                                  )

        ds_test_batched = get_test_dataset(dataset)

        # LOAD GLOBAL MODEL BEFORE UNLEARNING
        model_checkpoint_base_dir = os.path.join("model_checkpoints", config_dir,
                                                 "checkpoints")
        print(f"[Server] Loading checkpoint at {model_checkpoint_base_dir} ")
        last_round = find_last_checkpoint(model_checkpoint_base_dir)
        original_model = create_model(dataset=dataset,
                             total_classes=total_classes)
        if last_round > 0:
            model_checkpoint_dir = os.path.join(model_checkpoint_base_dir,
                                                f"R_{last_round}")
            original_model.load_weights(model_checkpoint_dir)
            t = last_round
        original_model.compile(optimizer='sgd',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(
                          from_logits=True),
                      metrics=['accuracy'])

        original_model.evaluate(ds_test_batched,)

        # RETRAIN MODEL
        # unlearned_cid_string = f"{cid}"
        model_checkpoint_base_dir = os.path.join(f"model_checkpoints_retrained{sample_unl}",
                                                 config_dir,
                                                 "client" + unlearned_cid_string,
                                                 "checkpoints")
        print(f"[Retrain] Loading checkpoint at {model_checkpoint_base_dir} ")
        last_round = find_last_checkpoint(model_checkpoint_base_dir)
        retrained_model = create_model(dataset=dataset,
                                      total_classes=total_classes)
        if last_round > 0:
            model_checkpoint_dir = os.path.join(model_checkpoint_base_dir,
                                                f"R_{last_round}")
            retrained_model.load_weights(model_checkpoint_dir)

        retrained_model.compile(optimizer='sgd',
                               loss=tf.keras.losses.SparseCategoricalCrossentropy(
                                   from_logits=True),
                               metrics=['accuracy'])
        if type(cid) is not list:
            # -----------------------------------------------------------
            # LOAD CLIENT DS
            ds_train_client = load_client_datasets_from_files(
                selected_client=int(cid),
                dataset=dataset,
                total_clients=total_clients,
                alpha=alpha,
                sample_unl=sample_unlearning,
            )

            def randomize_label(x, y):
                new_y = tf.random.uniform(shape=(), minval=0, maxval=total_classes,
                                          dtype=tf.int64)
                return x, new_y

            if sample_unlearning:
                ds_forget_client = load_client_datasets_from_files(
                    selected_client=int(cid),
                    dataset=dataset,
                    total_clients=total_clients,
                    alpha=alpha,
                    split="forget",
                    sample_unl=sample_unlearning,
                )

                ds_train_client_for_test = preprocess_ds_test(ds_forget_client, dataset,
                                                              reshuffle_each_iteration=False)

                ds_train_client_for_test = ds_train_client_for_test.batch(local_batch_size,
                                                                          drop_remainder=False)
                ds_train_client_for_test = ds_train_client_for_test.prefetch(
                    tf.data.AUTOTUNE)

                # forget data with random labels
                print("[Sample Unlearning] Randomizing labels for forget data")
                ds_forget_client = preprocess_ds_test(ds_forget_client, dataset)
                ds_forget_client_randomized = ds_forget_client.map(randomize_label)

                # retain data
                print("[Sample Unlearning] Using retain data as is")
                ds_retain_client = preprocess_ds_test(ds_train_client, dataset)

                # concatenate
                ds_unlearning_unbatched = ds_retain_client.concatenate(ds_forget_client_randomized)
                print("[Sample Unlearning] Shuffling")
                ds_unlearning_unbatched = ds_unlearning_unbatched.shuffle(5200)

            else:
                ds_train_client_for_test = preprocess_ds_test(ds_train_client, dataset,
                                                              reshuffle_each_iteration=False)

                ds_train_client_for_test = ds_train_client_for_test.batch(local_batch_size,
                                                                          drop_remainder=False)
                ds_train_client_for_test = ds_train_client_for_test.prefetch(
                    tf.data.AUTOTUNE)

                ds_unlearning_unbatched = preprocess_ds_test(ds_train_client, dataset)
                ds_unlearning_unbatched = ds_unlearning_unbatched.map(randomize_label)

            ds_unlearning = ds_unlearning_unbatched.batch(local_batch_size,
                                                              drop_remainder=False)
            ds_unlearning = ds_unlearning.prefetch(tf.data.AUTOTUNE)

            # model
            local_model = create_model(dataset=dataset,
                                 total_classes=total_classes)
            local_model.set_weights(original_model.get_weights())
            local_model.summary()

            # Build new classifier with same input size
            if model in ["ResNet18"]:
                existing_weights = local_model.fully_connected.get_weights()  # [kernel, bias]
                input_dim = existing_weights[0].shape[0]  # shape: [input_dim, num_classes]
                new_classifier = tf.keras.layers.Dense(
                    total_classes,
                    kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01,
                                                                          seed=seed),
                    kernel_regularizer=tf.keras.regularizers.l2(l2_weight_decay),
                    bias_regularizer=tf.keras.regularizers.l2(l2_weight_decay),
                )
                new_classifier.build(input_shape=(None, input_dim))
                local_model.fully_connected.set_weights(new_classifier.get_weights())

                local_model.layer0.trainable = False
                local_model.layer1.trainable = False
                local_model.layer2.trainable = False
                local_model.layer3.trainable = False
                local_model.layer4.trainable = False
                local_model.gap.trainable = False
                local_model.fully_connected.trainable = True
            else:
                new_classifier_weights = create_model(dataset=dataset,
                                                      total_classes=total_classes).classifier.get_weights()
                local_model.classifier.set_weights(new_classifier_weights)
                local_model.segformer.trainable = False
                local_model.classifier.trainable = True
                local_model.summary(show_trainable=True)


            if model in ["ResNet18"]:
                optimizer = tf.keras.optimizers.experimental.SGD(
                    learning_rate=learning_rate * (lr_decay ** (t - 1)),
                    # lr=0.01, 0.1 (mnist, cifar)
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
                ds_unlearning,
                epochs=epochs_to_train_w_a,
                # validation_data=ds_test,
                verbose=2
            )

            print("----------------------------------")
            print("[Test acc] Local model")
            _, test_acc = local_model.evaluate(ds_test_batched, verbose=2)
            print("[Train acc] Local model")
            _, train_acc = local_model.evaluate(ds_train_client_for_test, verbose=2)
            print("----------------------------------")

            unlearned_model = create_model(dataset=dataset,
                                 total_classes=total_classes)
            unlearned_model.set_weights(original_model.get_weights())

            if model in ["ResNet18"]:
                w_l = original_model.fully_connected.get_weights()
                w_a = local_model.fully_connected.get_weights()
            else:
                w_l = original_model.classifier.get_weights()
                w_a = local_model.classifier.get_weights()

            # Combine them element-wise
            new_weights = [
                alpha_fed_au * w_l[0] + (1 - alpha_fed_au) * w_a[0],  # kernel
                alpha_fed_au * w_l[1] + (1 - alpha_fed_au) * w_a[1],  # bias
            ]

        else:  # ---------------- Multiple unlearning -------------------
            # -----------------------------------------------------------
            # LOAD CLIENT DS
            w_a_list = list()

            for ucid in cid:
                print(f"[Client {ucid}] Loading dataset..")
                ds_train_client = load_client_datasets_from_files(
                    selected_client=int(ucid),
                    dataset=dataset,
                    total_clients=total_clients,
                    alpha=alpha,
                )

                def randomize_label(x, y):
                    new_y = tf.random.uniform(shape=(), minval=0, maxval=total_classes,
                                              dtype=tf.int64)
                    return x, new_y

                ds_train_client_for_test = preprocess_ds_test(ds_train_client, dataset,
                                                              reshuffle_each_iteration=False)

                ds_train_client_for_test = ds_train_client_for_test.batch(local_batch_size,
                                                                          drop_remainder=False)
                ds_train_client_for_test = ds_train_client_for_test.prefetch(
                    tf.data.AUTOTUNE)

                ds_unlearning_unbatched = preprocess_ds_test(ds_train_client, dataset)
                ds_unlearning_unbatched = ds_unlearning_unbatched.map(randomize_label)

                ds_unlearning = ds_unlearning_unbatched.batch(local_batch_size,
                                                              drop_remainder=False)
                ds_unlearning = ds_unlearning.prefetch(tf.data.AUTOTUNE)

                # model
                local_model = create_model(dataset=dataset,
                                           total_classes=total_classes)
                local_model.set_weights(original_model.get_weights())
                local_model.summary()

                # Build new classifier with same input size
                if model in ["ResNet18"]:
                    existing_weights = local_model.fully_connected.get_weights()  # [kernel, bias]
                    input_dim = existing_weights[0].shape[0]  # shape: [input_dim, num_classes]
                    new_classifier = tf.keras.layers.Dense(
                        total_classes,
                        kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01,
                                                                              seed=seed),
                        kernel_regularizer=tf.keras.regularizers.l2(l2_weight_decay),
                        bias_regularizer=tf.keras.regularizers.l2(l2_weight_decay),
                    )
                    new_classifier.build(input_shape=(None, input_dim))
                    local_model.fully_connected.set_weights(new_classifier.get_weights())

                    local_model.layer0.trainable = False
                    local_model.layer1.trainable = False
                    local_model.layer2.trainable = False
                    local_model.layer3.trainable = False
                    local_model.layer4.trainable = False
                    local_model.gap.trainable = False
                    local_model.fully_connected.trainable = True
                else:
                    new_classifier_weights = create_model(dataset=dataset,
                                                          total_classes=total_classes).classifier.get_weights()
                    local_model.classifier.set_weights(new_classifier_weights)
                    local_model.segformer.trainable = False
                    local_model.classifier.trainable = True
                    local_model.summary(show_trainable=True)

                if model in ["ResNet18"]:
                    optimizer = tf.keras.optimizers.experimental.SGD(
                        learning_rate=learning_rate * (lr_decay ** (t - 1)),
                        # lr=0.01, 0.1 (mnist, cifar)
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
                print(f"[Client {ucid}] Local training..")
                local_model.fit(
                        ds_unlearning,
                        epochs=epochs_to_train_w_a,
                        # validation_data=ds_test,
                        verbose=2
                )

                print("----------------------------------")
                print("[Test acc] Local model")
                _, test_acc = local_model.evaluate(ds_test_batched, verbose=2)
                print("[Train acc] Local model")
                _, train_acc = local_model.evaluate(ds_train_client_for_test, verbose=2)
                print("----------------------------------")

                unlearned_model = create_model(dataset=dataset,
                                                   total_classes=total_classes)
                unlearned_model.set_weights(original_model.get_weights())

                if model in ["ResNet18"]:
                    w_l = original_model.fully_connected.get_weights()
                    w_a = local_model.fully_connected.get_weights()
                else:
                    w_l = original_model.classifier.get_weights()
                    w_a = local_model.classifier.get_weights()

                w_a_list.append(w_a)

            # Aggregate w_a of multiple clients via average
            print("Aggregation of w_a")
            kernel_stack = np.stack([wa[0] for wa in w_a_list], axis=0)  # [num_clients, in_dim, out_dim]
            bias_stack = np.stack([wa[1] for wa in w_a_list], axis=0)  # [num_clients, out_dim]

            w_a_aggregated = [
                kernel_stack.mean(axis=0),
                bias_stack.mean(axis=0),
            ]

            # Combine them element-wise
            new_weights = [
               alpha_fed_au * w_l[0] + (1 - alpha_fed_au) * w_a_aggregated[0],  # kernel
               alpha_fed_au * w_l[1] + (1 - alpha_fed_au) * w_a_aggregated[1],  # bias
            ]
        #--- END IF ----
        #---------------

        if model in ["ResNet18"]:
            # Set the combined weights in the new model
            unlearned_model.fully_connected.set_weights(new_weights)

            unlearned_model.compile(optimizer='sgd',
                                    loss=tf.keras.losses.SparseCategoricalCrossentropy(
                                        from_logits=True),
                                    metrics=['accuracy'])
        else:
            unlearned_model.classifier.set_weights(new_weights)
            clipnorm = None
            l2_weight_decay = 1e-3
            optimizer = tf.keras.optimizers.experimental.AdamW(
                learning_rate=learning_rate * (lr_decay ** (t - 1)),
                clipnorm=clipnorm,
                weight_decay=l2_weight_decay,
            )
            unlearned_model.compile(
                optimizer=optimizer,
                loss=tf.keras.losses.SparseCategoricalCrossentropy(
                    from_logits=True,
                ),
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
            )

        if type(cids) is not list:
            print("----------------------------------")
            print("[Test acc] Original model")
            _, test_acc = original_model.evaluate(ds_test_batched, verbose=2)
            print("[Train acc] Original model")
            _, train_acc = original_model.evaluate(ds_train_client_for_test, verbose=2)
            print("----------------------------------")
            print("[Test acc] Retrained model")
            _, test_acc = retrained_model.evaluate(ds_test_batched, verbose=2)
            print("[Train acc] Retrained model")
            _, train_acc = retrained_model.evaluate(ds_train_client_for_test, verbose=2)
            print("----------------------------------")
            print("[Test acc] Unlearned model")
            _, test_acc = unlearned_model.evaluate(ds_test_batched, verbose=2)
            print("[Train acc] Unlearned model")
            _, train_acc = unlearned_model.evaluate(ds_train_client_for_test, verbose=2)
            print("----------------------------------")
        else:
            first_time_ds = True
            for u in cid:
                ds = load_client_datasets_from_files(
                    selected_client=int(u),
                    dataset=dataset,
                    total_clients=total_clients,
                    alpha=alpha,
                )
                if first_time_ds:
                    ds_forget_client = ds
                    first_time_ds = False
                else:
                    ds_forget_client = ds_forget_client.concatenate(ds)

            ds_forget_client = preprocess_ds_test(ds_forget_client, dataset)
            ds_forget_client = ds_forget_client.batch(local_batch_size,
                                                      drop_remainder=False)
            ds_forget_client = ds_forget_client.prefetch(tf.data.AUTOTUNE)
            print("----------------------------------")
            print("[Test acc] Original model")
            _, test_acc = original_model.evaluate(ds_test_batched, verbose=2)
            print("[Train acc] Original model")
            _, train_acc = original_model.evaluate(ds_forget_client, verbose=2)
            print("----------------------------------")
            print("[Test acc] Retrained model")
            _, test_acc = retrained_model.evaluate(ds_test_batched, verbose=2)
            print("[Train acc] Retrained model")
            _, train_acc = retrained_model.evaluate(ds_forget_client, verbose=2)
            print("----------------------------------")
            print("[Test acc] Unlearned model")
            _, test_acc = unlearned_model.evaluate(ds_test_batched, verbose=2)
            print("[Train acc] Unlearned model")
            _, train_acc = unlearned_model.evaluate(ds_forget_client, verbose=2)
            print("----------------------------------")

        print("[Server] Saving checkpoint... ")
        unlearning_config = f"alpha_{alpha_fed_au}_e_{epochs_to_train_w_a}"

        model_checkpoint_dir = os.path.join(f"model_checkpoints{sample_unl}", config_dir, "fedau",
                                            unlearning_config,
                                            f"R_{last_round}_unlearned_client_{unlearned_cid_string}")
        unlearned_model.save(model_checkpoint_dir)



if __name__ == "__main__":
    main()

