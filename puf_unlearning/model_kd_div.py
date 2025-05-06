"""Implement FedAvg+KD as tf.keras.Model."""

import tensorflow as tf


# pylint: disable=W0223
class ModelKLDiv(tf.keras.Model):
    """
    """

    def __init__(
        self,
        model: tf.keras.Model,
        virtual_model: tf.keras.Model,
    ):
        super().__init__()
        self.model = model
        self.virtual_teacher = virtual_model

    def train_step(self, data):
        """Implement logic for one training step.

        This method can be overridden to support custom training logic.
        """
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        virtual_output = self.virtual_teacher(data, training=True)

        with tf.GradientTape() as tape:
            local_output = self.model(x, training=True)  # Forward pass

            # Compute the loss value
            # (the loss function is configured in `compile()`)
            kd_loss = self.compiled_loss(
                # tf.nn.softmax(virtual_output, axis=1),
                virtual_output,
                tf.nn.softmax(local_output, axis=1),
                regularization_losses=self.model.losses
            )


        # Compute gradients
        trainable_vars = self.model.trainable_variables
        gradients = tape.gradient(kd_loss, trainable_vars)
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



class ModelNoT(tf.keras.Model):
    """
    """

    def __init__(
        self,
        model: tf.keras.Model,
    ):
        super().__init__()
        self.model = model

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


class ModelKLDivAdaptive(tf.keras.Model):
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
        range_idx = tf.expand_dims(tf.range(0, tf.shape(y)[0]), -1)

        if self.model_type in ["MitB0"]:
            idx = tf.concat([tf.cast(range_idx, tf.int64), y], axis=1)
        else:
            idx = tf.concat([tf.cast(range_idx, tf.int64), tf.expand_dims(y, -1)],
                            axis=1)
        global_output = tf.tensor_scatter_nd_update(tf.cast(global_output, tf.float32),
                                             indices=idx,
                                             updates=tf.zeros(tf.shape(x)[0])
                                             )
        global_output = tf.nn.softmax(global_output, axis=1)

        with tf.GradientTape() as tape:
            local_output = self.model(x, training=True)  # Forward pass
            if self.model_type in ["MitB0"]:
                local_output = local_output.logits
            # range_idx = tf.expand_dims(tf.range(0, tf.shape(y)[0]), -1)
            # idx = tf.concat([tf.cast(range_idx, tf.int64), tf.expand_dims(y, -1)],
            #                 axis=1)
            # local_output = tf.tensor_scatter_nd_update(tf.cast(local_output, tf.float32),
            #                                      indices=idx,
            #                                      updates=tf.zeros(tf.shape(x)[0])
            #                                      )
            local_output = tf.nn.softmax(local_output, axis=1)


            # Compute the loss value
            # (the loss function is configured in `compile()`)
            kd_loss = self.compiled_loss(
                # tf.nn.softmax(virtual_output, axis=1),
                global_output,
                local_output,  # above softmaxed
                regularization_losses=self.model.losses
            )


        # Compute gradients
        trainable_vars = self.model.trainable_variables
        gradients = tape.gradient(kd_loss, trainable_vars)
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



class ModelKLDivLogitMin(tf.keras.Model):
    """
    """

    def __init__(
        self,
        model: tf.keras.Model,
        global_model: tf.keras.Model,
    ):
        super().__init__()
        self.model = model
        self.global_model = global_model

    def train_step(self, data):
        """Implement logic for one training step.

        This method can be overridden to support custom training logic.
        """
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        global_output = self.global_model(x, training=False)
        min_tensor = tf.math.reduce_min(global_output, axis=1)
        range_idx = tf.expand_dims(tf.range(0, tf.shape(y)[0]), -1)
        idx = tf.concat([tf.cast(range_idx, tf.int64), tf.expand_dims(y, -1)], axis=1)
        global_output = tf.tensor_scatter_nd_update(tf.cast(global_output, tf.float32),
                                             indices=idx,
                                             updates=min_tensor
                                             )
        global_output = tf.nn.softmax(global_output, axis=1)
        # find minimum in global output

        with tf.GradientTape() as tape:
            local_output = self.model(x, training=True)  # Forward pass
            local_output = tf.nn.softmax(local_output, axis=1)


            # Compute the loss value
            # (the loss function is configured in `compile()`)
            kd_loss = self.compiled_loss(
                # tf.nn.softmax(virtual_output, axis=1),
                global_output,
                local_output,  # above softmaxed
                regularization_losses=self.model.losses
            )


        # Compute gradients
        trainable_vars = self.model.trainable_variables
        gradients = tape.gradient(kd_loss, trainable_vars)
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


class ModelKLDivAdaptiveSoftmax(tf.keras.Model):
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
        range_idx = tf.expand_dims(tf.range(0, tf.shape(y)[0]), -1)

        if self.model_type in ["MitB0"]:
            idx = tf.concat([tf.cast(range_idx, tf.int64), y], axis=1)
        else:
            idx = tf.concat([tf.cast(range_idx, tf.int64), tf.expand_dims(y, -1)],
                            axis=1)

        global_output = tf.nn.softmax(global_output, axis=1)
        num_classes = tf.cast(tf.shape(global_output)[1], tf.float32)
        predicted_true_probability = tf.gather_nd(global_output, indices=idx)
        a = 1 / num_classes
        delta = (predicted_true_probability - a) / (num_classes - 1)
        delta = tf.repeat(tf.expand_dims(delta, -1),
                          repeats=tf.cast(num_classes, tf.int32), axis=1)
        global_output = global_output + delta
        # tf.print(tf.shape(global_output))
        # tf.print(tf.shape(idx))
        if self.model_type in ["MitB0"]:
            global_output = tf.tensor_scatter_nd_update(global_output, indices=idx,
                                                    updates=tf.fill(tf.shape(tf.squeeze(y)), a))
        else:
            global_output = tf.tensor_scatter_nd_update(global_output, indices=idx,
                                                    updates=tf.fill(tf.shape(y), a))

        with tf.GradientTape() as tape:
            local_output = self.model(x, training=True)  # Forward pass
            if self.model_type in ["MitB0"]:
                local_output = local_output.logits
            local_output = tf.nn.softmax(local_output, axis=1)

            # (the loss function is configured in `compile()`)
            kd_loss = self.compiled_loss(
                # tf.nn.softmax(virtual_output, axis=1),
                global_output,
                local_output,  # above softmaxed
                regularization_losses=self.model.losses
            )


        # Compute gradients
        trainable_vars = self.model.trainable_variables
        gradients = tape.gradient(kd_loss, trainable_vars)
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


class ModelCompetentIncompetentTeacher(tf.keras.Model):
    """
    """

    def __init__(
        self,
        model: tf.keras.Model,
        global_model: tf.keras.Model,
        virtual_model: tf.keras.Model,
    ):
        super().__init__()
        self.model = model
        self.global_model = global_model
        self.virtual_teacher = virtual_model


    def train_step(self, data):
        """Implement logic for one training step.

        This method can be overridden to support custom training logic.
        """
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y, unlearning_label = data  # unlearning_label 1 if the sample belongs to Df and 0 if it belongs to Dr.
        unlearning_label = tf.cast(unlearning_label, tf.float32)
        # tf.print(unlearning_label)
        global_output = self.global_model(x, training=False)
        global_output = tf.nn.softmax(global_output, axis=1)

        # kl = tf.keras.losses.KLDivergence()

        virtual_output = self.virtual_teacher((x, y), training=False)
        # tf.print(tf.shape(y))

        with tf.GradientTape() as tape:
            local_output = self.model(x, training=True)  # Forward pass
            local_output = tf.nn.softmax(local_output, axis=1)

            # overall_teacher_out = labels * u_teacher_out + (1 - labels) * f_teacher_out
            # ones = tf.fill(tf.shape(unlearning_label), 1.0)
            # tf.print(tf.shape(unlearning_label))

            # overall_teacher_out = tf.subtract(1, unlearning_label) * global_output + unlearning_label * virtual_output
            # overall_teacher_out = tf.subtract(1, unlearning_label) * global_output + unlearning_label * virtual_output
            overall_teacher_out = tf.expand_dims(tf.subtract(1.0, unlearning_label),
                                                 -1) * global_output + tf.expand_dims(
                unlearning_label, -1) * virtual_output
            kd_loss = self.compiled_loss(
                overall_teacher_out,
                local_output,
            )

            # kd_loss_competent = kl(
            #     global_output,
            #     local_output,
            #     # regularization_losses=self.model.losses
            # )
            #
            # kd_loss_incompetent = kl(
            #     virtual_output,
            #     local_output,
            #     # regularization_losses=self.model.losses
            # )
            # tf.print("----------------")
            # tf.print(kd_loss_competent, summarize=-1)
            # tf.print("----------------")
            # tf.print(kd_loss_incompetent, summarize=-1)
            # tf.print("----------------")
            # mixed_loss = (1-unlearning_label) * kd_loss_competent + unlearning_label * kd_loss_incompetent
            # tf.print("----------------")
            # tf.print(mixed_loss, summarize=-1)
        # Compute gradients
        trainable_vars = self.model.trainable_variables
        gradients = tape.gradient(kd_loss, trainable_vars)
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


class ModelKLDivSoftmaxZero(tf.keras.Model):
    """
    """

    def __init__(
        self,
        model: tf.keras.Model,
        global_model: tf.keras.Model,
    ):
        super().__init__()
        self.model = model
        self.global_model = global_model

    def train_step(self, data):
        """Implement logic for one training step.

        This method can be overridden to support custom training logic.
        """
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        global_output = self.global_model(x, training=False)

        range_idx = tf.expand_dims(tf.range(0, tf.shape(y)[0]), -1)
        idx = tf.concat([tf.cast(range_idx, tf.int64), tf.expand_dims(y, -1)], axis=1)

        global_output = tf.nn.softmax(global_output, axis=1)
        num_classes = tf.cast(tf.shape(global_output)[1], tf.float32)
        predicted_true_probability = tf.gather_nd(global_output, indices=idx)
        # a = 1 / num_classes
        a = 0.0
        delta = (predicted_true_probability - a) / (num_classes - 1)
        delta = tf.repeat(tf.expand_dims(delta, -1),
                          repeats=tf.cast(num_classes, tf.int32), axis=1)
        global_output = global_output + delta 
        global_output = tf.tensor_scatter_nd_update(global_output, indices=idx,
                                                    updates=tf.fill(tf.shape(y), a))

        with tf.GradientTape() as tape:
            local_output = self.model(x, training=True)  # Forward pass
            local_output = tf.nn.softmax(local_output, axis=1)

            # (the loss function is configured in `compile()`)
            kd_loss = self.compiled_loss(
                # tf.nn.softmax(virtual_output, axis=1),
                global_output,
                local_output,  # above softmaxed
                regularization_losses=self.model.losses
            )


        # Compute gradients
        trainable_vars = self.model.trainable_variables
        gradients = tape.gradient(kd_loss, trainable_vars)
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
