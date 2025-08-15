import tensorflow as tf
from puf_unlearning.dataset import load_client_datasets_from_files, \
    load_selected_client_statistics
from puf_unlearning.utility import create_model, preprocess_ds


def client_update(cid, local_model, model, dataset, total_clients, alpha, lr_decay, learning_rate, from_logits, local_batch_size=32, t=1, epochs=1):
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
            learning_rate=learning_rate * (lr_decay ** (t - 1)),
            # lr=0.01, 0.1 (mnist, cifar)
            weight_decay=None if dataset in ["mnist"] else 1e-3)
        local_model.compile(optimizer=optimizer,
                            loss=tf.keras.losses.SparseCategoricalCrossentropy(
                                from_logits=from_logits),
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
                from_logits=from_logits,
            ),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
        )

    # Local training
    print(f"[Client {cid}] Local training..")
    local_model.fit(
        ds_train_client,
        epochs=epochs,
        verbose=2
    )
    return local_model.get_weights(), amount_of_local_examples


class ModelModeGuidance(tf.keras.Model):
    """
    """

    def __init__(
        self,
        model: tf.keras.Model,
        global_model: tf.keras.Model,
        model_type: str = "ResNet18",
    ):
        super().__init__()
        self.model = model
        self.global_model = global_model
        self.model_type = model_type

    def train_step(self, data):
        """Implement logic for one training step.

        This method can be overridden to support custom training logic.
        """
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        global_output = self.global_model(x, training=False)
        if self.model_type in ["MitB0"]:
            global_output = global_output.logits
        pseudo_labels = tf.argmax(global_output, axis=1)

        with tf.GradientTape() as tape:
            local_output = self.model(x, training=True)  # Forward pass
            if self.model_type in ["MitB0"]:
                local_output = local_output.logits
            local_output = tf.nn.softmax(local_output, axis=1)

            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(
                pseudo_labels,
                local_output,  # above softmaxed
                regularization_losses=self.model.losses
            )


        # Compute gradients
        trainable_vars = self.model.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, local_output)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        """Implement logic for one evaluation step.

        This method can be overridden to support custom evaluation logic.
        """
        x, y = data
        y_pred = self.model(x, training=False)  # Forward pass
        # self.compiled_loss(y, y_pred, regularization_losses=self.local_model.losses)
        self.compiled_metrics.update_state(y, y_pred)
        # self.compiled_metrics
        return {m.name: m.result() for m in self.metrics}

    def get_weights(self):
        """Return the weights of the local model."""
        return self.model.get_weights()

    def set_weights(self, weights):
        """Return the weights of the local model."""
        return self.model.set_weights(weights)




# def mode_unlearning_rounds(clients, client_cid, global_model, model_fn, r_max=10, r_de=5, lamdba=0.95):
#     """Here commments"""
#     model_de = create_model(dataset=dataset, total_classes=total_classes)  # degradation model (random init)
#
#     for r in range(r_max):
#         if r < r_de:
#             # 1. Train Mde on remaining clients
#             local_models_de = []
#             for i, (dataset, is_target) in enumerate(clients):
#                 if not is_target:
#                     local_model = model_fn()
#                     local_model.set_weights(model_de.get_weights())
#                     local_models_de.append(local_train(local_model, dataset))
#             model_de = federated_average(local_models_de)
#
#             # 2. Apply MoDe update
#             new_weights = [
#                 lamdba * w + (1 - lamdba) * w_de
#                 for w, w_de in zip(global_model.get_weights(), model_de.get_weights())
#             ]
#             global_model.set_weights(new_weights)
#
#         # 3. Memory Guidance — all clients participate
#         local_models_M = []
#         for i, (dataset, is_target) in enumerate(clients):
#             local_model = model_fn()
#             local_model.set_weights(global_model.get_weights())
#             if is_target:
#                 pseudo_dataset = []
#                 for x_batch, _ in dataset:
#                     pseudo_labels = tf.argmax(model_de(x_batch), axis=1)
#                     pseudo_dataset.append((x_batch, pseudo_labels))
#                 local_model = local_train(local_model, pseudo_dataset)
#             else:
#                 local_model = local_train(local_model, dataset)
#             local_models_M.append(local_model)
#
#         global_model = federated_average(local_models_M)
#
#     return M


def mode_unlearning_round(
        current_round,
        max_rounds,
        degradation_rounds,
        degradation_model,
        server_model,
        model,
        total_clients,
        unlearned_cid,
        dataset,
        total_classes,
        alpha,
        remaining_client_examples,
        total_examples,
        lr_decay,
        learning_rates,
        model_type="ResNet18",
        lamdba=0.95):
    """
    Integrate MoDe algorithm with memory guidance into existing FL simulation.
    """
    print(f"[MoDe] Round {current_round}, Degradation Rounds {degradation_rounds}, Max Rounds {max_rounds}")
    print(f"remaining_client_examples: {remaining_client_examples}")
    print(f"total_examples: {total_examples}")
    degradation_model_weights = degradation_model.get_weights()
    global_weights = server_model.get_weights()
    if current_round <= degradation_rounds:
        delta_w_aggregated = tf.nest.map_structure(lambda a, b: a - b,
                                              degradation_model_weights,
                                              degradation_model_weights,
                                              )

        for cid in range(total_clients):
            if cid in unlearned_cid:
                continue

            client_model = create_model(dataset=dataset, total_classes=total_classes)
            client_model.set_weights(degradation_model_weights)
            client_weights, local_samples = client_update(
                cid,
                client_model,
                dataset=dataset,
                total_clients=total_clients,
                alpha=alpha,
                lr_decay=1.0,
                learning_rate=learning_rates[0],
                model=model,
                t=current_round,
                from_logits=True)
            # FedAvg aggregation
            delta_w_de = tf.nest.map_structure(lambda a, b: a - b,
                                                  client_weights,
                                                  degradation_model_weights,
                                                  )

            delta_w_aggregated = tf.nest.map_structure(
                lambda a, b: a + b * (local_samples / remaining_client_examples),
                delta_w_aggregated,
                delta_w_de)

        new_model_de_weights = tf.nest.map_structure(lambda a, b: a + b,
                                                   degradation_model_weights,
                                                   delta_w_aggregated)
        degradation_model.set_weights(new_model_de_weights)

        # Apply MoDe update to global model
        print("[MoDE Momentum Update]")
        server_model.set_weights(
                [lamdba * wg + (1 - lamdba) * wd for wg, wd in zip(global_weights, new_model_de_weights)])


    # --- Memory Guidance ---
    print("[MoDE Memory Guidance]")
    #init
    delta_w_mem = tf.nest.map_structure(lambda a, b: a - b,
                                              degradation_model_weights,
                                              degradation_model_weights,
                                              )
    for cid in range(total_clients):
        client_model = create_model(dataset=dataset, total_classes=total_classes)
        client_model.set_weights(server_model.get_weights())

        if cid in unlearned_cid:
            print("Unlearning client mem guidance")
            mode_model = ModelModeGuidance(client_model, degradation_model, model_type=model_type)
            client_weights, local_samples = client_update(
                cid,
                mode_model,
                dataset=dataset,
                total_clients=total_clients,
                alpha=alpha,
                lr_decay=1.0,
                learning_rate=learning_rates[0],
                model=model,
                t=current_round,
                from_logits=False)
        else:
            # normal training
            client_weights, local_samples = client_update(
                cid,
                client_model,
                dataset=dataset,
                total_clients=total_clients,
                alpha=alpha,
                lr_decay=1.0,
                learning_rate=learning_rates[1],
                model=model,
                t=current_round,
                from_logits=True)

        delta_w = tf.nest.map_structure(lambda a, b: a - b,
                                                   client_weights,
                                                   server_model.get_weights(),
                                                   )

        delta_w_mem = tf.nest.map_structure(
                    lambda a, b: a + b * (local_samples / total_examples),
                    delta_w_mem,
                    delta_w)

    new_model_weights = tf.nest.map_structure(lambda a, b: a + b,
                                                         server_model.get_weights(),
                                                         delta_w_mem)
    server_model.set_weights(new_model_weights)

    return server_model, degradation_model

