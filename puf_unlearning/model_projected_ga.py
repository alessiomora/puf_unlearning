import tensorflow as tf
import numpy as np


# Calculate distance between two sets of weights (L2 distance)
def get_distance(model1, model2):
    weights1 = tf.concat([tf.reshape(w, [-1]) for w in model1.get_weights()],
                         axis=0)
    weights2 = tf.concat([tf.reshape(w, [-1]) for w in model2.get_weights()],
                         axis=0)
    return tf.norm(weights1 - weights2, ord=2)

def get_distance_ww(model1, model2):
    weights1 = tf.concat([tf.reshape(w, [-1]) for w in model1],
                         axis=0)
    weights2 = tf.concat([tf.reshape(w, [-1]) for w in model2],
                         axis=0)
    return tf.norm(weights1 - weights2, ord=2)

# class ModelProjectedGA(tf.keras.Model):
#     def __init__(self, original_model, ref_model, threshold):
#         super(ModelProjectedGA, self).__init__()
#         self.model = original_model
#         self.model_ref = ref_model
#         self.threshold = threshold  # Projection threshold
#
#     # Project the weights back to the norm-ball
#     def project_weights(self):
#         current_weights = tf.concat(
#             [tf.reshape(w, [-1]) for w in self.model.get_weights()], axis=0)
#         ref_weights = tf.concat(
#             [tf.reshape(w, [-1]) for w in self.model_ref.get_weights()], axis=0)
#         dist_vec = current_weights - ref_weights
#         dist_norm = tf.norm(dist_vec, ord=2)
#
#         dist_vec = dist_vec / dist_norm * np.sqrt(self.threshold)
#         proj_vec = ref_weights + dist_vec
#
#         # Reshape the vector back into the original shape
#         new_weights = []
#         start_idx = 0
#         for w in self.model.get_weights():
#             weight_size = tf.size(w).numpy()
#             new_weights.append(
#                     tf.reshape(proj_vec[start_idx:start_idx + weight_size], w.shape))
#             start_idx += weight_size
#
#         self.model.set_weights(new_weights)
#
#     # Custom training step with EarlyStopping based on L2 distance
#     # def train_step(self, data):
#     #     x_batch, y_batch = data
#     #
#     #     # Gradient computation and ascent (negative of the loss)
#     #     with tf.GradientTape() as tape:
#     #         outputs = self.model(x_batch, training=True)
#     #         # predictions = tf.nn.softmax(outputs)
#     #         loss = self.compiled_loss(y_batch, outputs)
#     #         loss_joint = -loss  # Negate loss for gradient ascent
#     #
#     #     # Compute gradients
#     #     gradients = tape.gradient(loss_joint, self.model.trainable_variables)
#     #
#     #     # Apply gradients to model parameters
#     #     self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
#     #
#     #     # Project weights onto the L2-norm ball
#     #     distance = get_distance_ww(self.get_weights(), self.model_ref)
#     #     if distance > self.threshold:
#     #         self.project_weights()
#     #
#     #     # Return metrics to be monitored
#     #     return {"loss": loss_joint, "distance": distance}
#
#     def get_weights(self):
#         """Return the weights of the local model."""
#         return self.model.get_weights()
#
#     def set_weights(self, weights):
#         """Return the weights of the local model."""
#         return self.model.set_weights(weights)


class DistanceEarlyStopping(tf.keras.callbacks.Callback):
    def __init__(self, distance_threshold, reference_model):
        super(DistanceEarlyStopping, self).__init__()
        self.distance_threshold = distance_threshold
        self.model_ref = reference_model

    # Check the distance at the end of every batch
    def on_batch_end(self, batch, logs=None):
        distance = get_distance(self.model, self.model_ref)
        print(f"Batch {batch}: L2 distance = {distance.numpy()}")

        if distance > self.distance_threshold:
            print(f"Stopping early! Distance {distance.numpy()} exceeded threshold {self.threshold}")
            self.model.stop_training = True  # Stops the training loop


def project_weights(model, model_ref, threshold):
        current_weights = tf.concat(
            [tf.reshape(w, [-1]) for w in model.get_weights()], axis=0)
        ref_weights = tf.concat(
            [tf.reshape(w, [-1]) for w in model_ref.get_weights()], axis=0)
        dist_vec = current_weights - ref_weights
        dist_norm = tf.norm(dist_vec, ord=2)

        dist_vec = dist_vec / dist_norm * np.sqrt(threshold)
        proj_vec = ref_weights + dist_vec

        # Reshape the vector back into the original shape
        new_weights = []
        start_idx = 0
        for w in model.get_weights():
            weight_size = tf.size(w).numpy()
            new_weights.append(
                    tf.reshape(proj_vec[start_idx:start_idx + weight_size], w.shape))
            start_idx += weight_size

        model.set_weights(new_weights)

def custom_train_loop(model, model_ref, unl_client_model, threshold, optimizer, epochs, train_dataset, distance_early_stop):
    stop_training = False
    for epoch in range(epochs):
        print("Start of epoch %d" % (epoch,))
        if stop_training:
            break
        for step, data in enumerate(train_dataset):
            # with tf.GradientTape() as tape:
            #     reconstructed = model(x_batch_train)
            #     loss = mse_loss_fn(x_batch_train, reconstructed)
            #     loss += sum(model.losses)  # Add KLD regularization loss
            x_batch, y_batch = data
            with tf.GradientTape() as tape:
                outputs = model(x_batch, training=True)
                # predictions = tf.nn.softmax(outputs)
                loss = model.compiled_loss(y_batch, outputs)
                loss_joint = -loss

            gradients = tape.gradient(loss_joint, model.trainable_variables)
            # Apply gradients to model parameters
            model.optimizer.apply_gradients(
                zip(gradients, model.trainable_variables))

            # Project weights onto the L2-norm ball
            distance = get_distance(model, model_ref)
            if distance > threshold:
                print("Project weights!")
                project_weights(model, model_ref, threshold)

            distance = get_distance(model, unl_client_model)
            if distance > distance_early_stop:
                print(
                    f"Stopping early! Distance {distance} exceeded threshold {distance_early_stop}")
                stop_training = True  # Stops the training loop
                break

            # Return metrics to be monitored
            print(f"loss: {loss_joint} distance: {distance}")
