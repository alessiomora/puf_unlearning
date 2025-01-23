"""
Code adapted from
MIA implementation from https://arxiv.org/abs/2304.04934
Model Sparsity Can Simplify Machine Unlearning
Jinghan Jia, Jiancheng Liu, Parikshit Ram, Yuguang Yao, Gaowen Liu, Yang Liu, Pranay Sharma, Sijia Liu
"""
import os

import numpy as np
# import torch
# import torch.nn.functional as F
import tensorflow as tf
import tensorflow_datasets as tfds
from omegaconf import DictConfig, OmegaConf
import hydra
import os

from basics_unlearning.dataset import normalize_img, load_client_datasets_from_files, \
    load_selected_client_statistics, get_string_distribution, expand_dims
from basics_unlearning.model import create_cnn_model


class black_box_benchmarks(object):
    def __init__(
        self,
        shadow_train_performance,
        shadow_test_performance,
        target_train_performance,
        target_test_performance,
        num_classes,
    ):
        """
        each input contains both model predictions (shape: num_data*num_classes) and ground-truth labels.
        """
        self.num_classes = num_classes

        self.s_tr_outputs, self.s_tr_labels = shadow_train_performance
        self.s_te_outputs, self.s_te_labels = shadow_test_performance
        self.t_tr_outputs, self.t_tr_labels = target_train_performance
        self.t_te_outputs, self.t_te_labels = target_test_performance

        self.s_tr_corr = (
            np.argmax(self.s_tr_outputs, axis=1) == self.s_tr_labels
        ).astype(int)
        self.s_te_corr = (
            np.argmax(self.s_te_outputs, axis=1) == self.s_te_labels
        ).astype(int)
        self.t_tr_corr = (
            np.argmax(self.t_tr_outputs, axis=1) == self.t_tr_labels
        ).astype(int)
        self.t_te_corr = (
            np.argmax(self.t_te_outputs, axis=1) == self.t_te_labels
        ).astype(int)

        self.s_tr_conf = np.take_along_axis(
            self.s_tr_outputs, self.s_tr_labels[:, None], axis=1
        )
        self.s_te_conf = np.take_along_axis(
            self.s_te_outputs, self.s_te_labels[:, None], axis=1
        )
        self.t_tr_conf = np.take_along_axis(
            self.t_tr_outputs, self.t_tr_labels[:, None], axis=1
        )
        self.t_te_conf = np.take_along_axis(
            self.t_te_outputs, self.t_te_labels[:, None], axis=1
        )

        self.s_tr_entr = self._entr_comp(self.s_tr_outputs)
        self.s_te_entr = self._entr_comp(self.s_te_outputs)
        self.t_tr_entr = self._entr_comp(self.t_tr_outputs)
        self.t_te_entr = self._entr_comp(self.t_te_outputs)

        self.s_tr_m_entr = self._m_entr_comp(self.s_tr_outputs, self.s_tr_labels)
        self.s_te_m_entr = self._m_entr_comp(self.s_te_outputs, self.s_te_labels)
        self.t_tr_m_entr = self._m_entr_comp(self.t_tr_outputs, self.t_tr_labels)
        self.t_te_m_entr = self._m_entr_comp(self.t_te_outputs, self.t_te_labels)

    def _log_value(self, probs, eps=1e-30):
        return -np.log(np.maximum(probs, eps))

    def _entr_comp(self, probs):
        return np.sum(np.multiply(probs, self._log_value(probs)), axis=1)

    def _m_entr_comp(self, probs, true_labels):
        log_probs = self._log_value(probs)
        reverse_probs = 1 - probs
        log_reverse_probs = self._log_value(reverse_probs)
        modified_probs = np.copy(probs)
        modified_probs[range(true_labels.size), true_labels] = reverse_probs[
            range(true_labels.size), true_labels
        ]
        modified_log_probs = np.copy(log_reverse_probs)
        modified_log_probs[range(true_labels.size), true_labels] = log_probs[
            range(true_labels.size), true_labels
        ]
        return np.sum(np.multiply(modified_probs, modified_log_probs), axis=1)

    def _thre_setting(self, tr_values, te_values):
        value_list = np.concatenate((tr_values, te_values))
        thre, max_acc = 0, 0
        for value in value_list:
            tr_ratio = np.sum(tr_values >= value) / (len(tr_values) + 0.0)
            te_ratio = np.sum(te_values < value) / (len(te_values) + 0.0)
            acc = 0.5 * (tr_ratio + te_ratio)
            if acc > max_acc:
                thre, max_acc = value, acc
        return thre

    def _mem_inf_via_corr(self):
        # perform membership inference attack based on whether the input is correctly classified or not
        t_tr_acc = np.sum(self.t_tr_corr) / (len(self.t_tr_corr) + 0.0)
        t_te_acc = 1 - np.sum(self.t_te_corr) / (len(self.t_te_corr) + 0.0)
        mem_inf_acc = 0.5 * (t_tr_acc + t_te_acc)
        print(
            "For membership inference attack via correctness, the attack acc is {acc1:.3f}, with train acc {acc2:.3f} and test acc {acc3:.3f}".format(
                acc1=mem_inf_acc, acc2=t_tr_acc, acc3=t_te_acc
            )
        )
        return t_tr_acc, t_te_acc

    def _mem_inf_thre(self, v_name, s_tr_values, s_te_values, t_tr_values, t_te_values):
        # perform membership inference attack by thresholding feature values: the feature can be prediction confidence,
        # (negative) prediction entropy, and (negative) modified entropy
        t_tr_mem, t_te_non_mem = 0, 0
        for num in range(self.num_classes):
            thre = self._thre_setting(
                s_tr_values[self.s_tr_labels == num],
                s_te_values[self.s_te_labels == num],
            )
            t_tr_mem += np.sum(t_tr_values[self.t_tr_labels == num] >= thre)
            t_te_non_mem += np.sum(t_te_values[self.t_te_labels == num] < thre)
        t_tr_acc = t_tr_mem / (len(self.t_tr_labels) + 0.0)
        t_te_acc = t_te_non_mem / (len(self.t_te_labels) + 0.0)
        mem_inf_acc = 0.5 * (t_tr_acc + t_te_acc)
        print(
            "For membership inference attack via {n}, the attack acc is {acc1:.3f}, with train acc {acc2:.3f} and test acc {acc3:.3f}".format(
                n=v_name, acc1=mem_inf_acc, acc2=t_tr_acc, acc3=t_te_acc
            )
        )
        return t_tr_acc, t_te_acc

    def _mem_inf_benchmarks(self, all_methods=True, benchmark_methods=[]):
        ret = {}
        if (all_methods) or ("correctness" in benchmark_methods):
            ret["correctness"] = self._mem_inf_via_corr()
        if (all_methods) or ("confidence" in benchmark_methods):
            ret["confidence"] = self._mem_inf_thre(
                "confidence",
                self.s_tr_conf,
                self.s_te_conf,
                self.t_tr_conf,
                self.t_te_conf,
            )
        if (all_methods) or ("entropy" in benchmark_methods):
            ret["entropy"] = self._mem_inf_thre(
                "entropy",
                -self.s_tr_entr,
                -self.s_te_entr,
                -self.t_tr_entr,
                -self.t_te_entr,
            )
        if (all_methods) or ("modified entropy" in benchmark_methods):
            ret["m_entropy"] = self._mem_inf_thre(
                "modified entropy",
                -self.s_tr_m_entr,
                -self.s_te_m_entr,
                -self.t_tr_m_entr,
                -self.t_te_m_entr,
            )

        return ret


# def collect_performance(data_loader, model, device):
#     probs = []
#     labels = []
#     model.eval()
#
#     for data, target in data_loader:
#         data = data.to(device)
#         target = target.to(device)
#         with torch.no_grad():
#             output = model(data)
#             prob = F.softmax(output, dim=-1)
#
#         probs.append(prob)
#         labels.append(target)
#
#     return torch.cat(probs).cpu().numpy(), torch.cat(labels).cpu().numpy()

def collect_performance(dataset, model):
    print("Collection performance..")
    probs = []
    # labels = []

    prob = tf.nn.softmax(
        model.predict(dataset),
        axis=1
    )

    probs.append(prob)
    labels = np.concatenate([y for _, y in dataset], axis=0)
    print(labels)
    return tf.concat(prob, axis=0).numpy(), labels


def MIA(
    retain_loader_train, retain_loader_test, forget_loader, test_loader, model
):
    shadow_train_performance = collect_performance(retain_loader_train, model)
    shadow_test_performance = collect_performance(test_loader, model)
    target_train_performance = collect_performance(retain_loader_test, model)
    target_test_performance = collect_performance(forget_loader, model)

    # shadow_train_performance, retain_loader_train
    # shadow_test_performance, test_loader
    # target_train_performance, retain_loader_test
    # target_test_performance, forget_loader
    BBB = black_box_benchmarks(
        shadow_train_performance,
        shadow_test_performance,
        target_train_performance,
        target_test_performance,
        num_classes=10,
    )
    return BBB._mem_inf_benchmarks()


@hydra.main(config_path="conf", config_name="generate_tables", version_base=None)
def main(cfg: DictConfig) -> None:
    print("[Start Simulation]")
    # Print parsed config
    print(OmegaConf.to_yaml(cfg))


    #  build base config
    dataset = cfg.dataset
    alpha = cfg.alpha
    alpha_dirichlet_string = get_string_distribution(alpha)
    local_batch_size = cfg.local_batch_size
    total_clients = cfg.total_clients
    clients_to_analyse = cfg.clients_to_analyse
    algorithm = cfg.algorithm
    active_clients = cfg.active_clients
    local_epochs = cfg.local_epochs
    best_round = cfg.best_round
    model_string = "LeNet" if dataset in ["mnist"] else "ResNet18"
    total_classes = 100 if dataset == "cifar100" else 10

    # test the code above
    ds_test = tfds.load(
        'mnist',
        split='test',
        shuffle_files=True,
        as_supervised=True,
    )


    cid=6
    # amount_of_local_examples = load_selected_client_statistics(
    #     cid,
    #     dataset="mnist",
    #     total_clients=10,
    #     alpha=0.1,
    # )
    first_time = True
    for i in range(total_clients):
        if i != cid:
            ds = load_client_datasets_from_files(
                selected_client=i,
                dataset="mnist",
                total_clients=10,
                alpha=0.1,
            )
            if first_time:
                ds_retain = ds
                first_time = False
            else:
                ds_retain = ds.concatenate(ds_retain)

    n = 10000
    print("number of examples: ", n)

    ds_retain = ds_retain.shuffle(60000)  #.take(n)
    ds_retain = ds_retain.map(
        normalize_img, num_parallel_calls=tf.data.AUTOTUNE).map(expand_dims)
    ds_retain = ds_retain.batch(128, drop_remainder=False)
    ds_retain = ds_retain.cache()
    ds_retain = ds_retain.prefetch(tf.data.AUTOTUNE)

    ds_test = ds_test.map(
        normalize_img, num_parallel_calls=tf.data.AUTOTUNE)

    test_ds_1 = ds_test  #.take(n)
    test_ds_2 = ds_test  #.skip(n)

    test_ds_1 = test_ds_1.batch(128, drop_remainder=False)
    test_ds_1 = test_ds_1.cache()
    test_ds_1 = test_ds_1.prefetch(tf.data.AUTOTUNE)

    test_ds_2 = test_ds_2.batch(128, drop_remainder=False)
    test_ds_2 = test_ds_2.cache()
    test_ds_2 = test_ds_2.prefetch(tf.data.AUTOTUNE)


    client_train_ds = load_client_datasets_from_files(
        selected_client=cid,
        dataset="mnist",
        total_clients=10,
        alpha=0.1,
    )
    # client_train_ds = client_train_ds.take(n)
    client_train_ds = client_train_ds
    client_train_ds = (
        client_train_ds.shuffle(512, reshuffle_each_iteration=False)
            .map(normalize_img)
            .map(expand_dims)
            .batch(128, drop_remainder=False)
    )
    client_train_ds = client_train_ds.prefetch(tf.data.AUTOTUNE)

    forgetting_ds = client_train_ds

    simple_cnn = create_cnn_model()
    config_dir = os.path.join(f"{dataset}_{alpha_dirichlet_string}",
                              f"LeNet_K{total_clients}_C{active_clients}_epochs{local_epochs}"
                              )
    model_checkpoint_dir = os.path.join("model_checkpoints", config_dir, "best",
                                        f"R_{best_round}")

    model_checkpoint_retrained_dir = os.path.join("model_checkpoints_retrained", config_dir, "client"+str(cid), "last_checkpoint",
                                        f"R_25")

    saved_checkpoint = tf.keras.saving.load_model(model_checkpoint_dir)
    original_weights = saved_checkpoint.get_weights()
    simple_cnn.set_weights(original_weights)
    # prob, label = collect_performance(dataset=ds_test_batched, model=simple_cnn)
    # print(tf.shape(prob))
    # print(tf.shape(label))

    # retain_loader_train, retain_loader_test, forget_loader=forgetting_ds, test_loader
    # MIA(retain_loader_train=shadow_train, retain_loader_test=target_train, forget_loader=forgetting_ds, test_loader=shadow_test, model=simple_cnn)

    # shadow_train_performance, retain_loader_train
    # shadow_test_performance, test_loader
    # target_train_performance, retain_loader_test
    # target_test_performance, forget_loader
    MIA(retain_loader_train=ds_retain,  # shadow
        retain_loader_test=test_ds_2,
        forget_loader=forgetting_ds,
        test_loader=test_ds_1,  # shadow
        model=simple_cnn)
    simple_cnn.compile(optimizer='sgd',
                         loss=tf.keras.losses.SparseCategoricalCrossentropy(
                             from_logits=True),
                         metrics=['accuracy'])


    saved_checkpoint = tf.keras.saving.load_model(model_checkpoint_retrained_dir)
    retrained_weights = saved_checkpoint.get_weights()
    simple_cnn.set_weights(retrained_weights)
    simple_cnn.compile(optimizer='sgd',
                         loss=tf.keras.losses.SparseCategoricalCrossentropy(
                             from_logits=True),
                         metrics=['accuracy'])
    MIA(retain_loader_train=ds_retain,  # shadow
        retain_loader_test=test_ds_2,
        forget_loader=forgetting_ds,
        test_loader=test_ds_1,  # shadow
        model=simple_cnn)

    print("retain")
    simple_cnn.evaluate(ds_retain)
    print("test2")
    simple_cnn.evaluate(test_ds_2)
    print("forgetting")
    simple_cnn.evaluate(forgetting_ds)
    print("test1")
    simple_cnn.evaluate(test_ds_1)
    # simple_cnn = create_cnn_model()
    # MIA(retain_loader_train=ds_retain,
    #     retain_loader_test=test_ds_2,
    #     forget_loader=forgetting_ds,
    #     test_loader=test_ds_1,
    #     model=simple_cnn)


if __name__ == "__main__":
    main()