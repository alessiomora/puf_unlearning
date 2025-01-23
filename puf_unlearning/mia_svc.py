import numpy as np
import torch
import torch.nn.functional as F
from sklearn.svm import SVC
import tensorflow as tf
import tensorflow_datasets as tfds
from omegaconf import DictConfig, OmegaConf
import hydra
import os

# from imagenet import get_x_y_from_data_dict
from basics_unlearning.dataset import get_string_distribution, \
    load_client_datasets_from_files, normalize_img, expand_dims
from basics_unlearning.model import create_cnn_model


def entropy(p, dim=-1, keepdim=False):
    return -torch.where(p > 0, p * p.log(), p.new([0.0])).sum(dim=dim, keepdim=keepdim)


def m_entropy(p, labels, dim=-1, keepdim=False):
    log_prob = torch.where(p > 0, p.log(), torch.tensor(1e-30).to(p.device).log())
    reverse_prob = 1 - p
    log_reverse_prob = torch.where(
        p > 0, p.log(), torch.tensor(1e-30).to(p.device).log()
    )
    modified_probs = p.clone()
    modified_probs[:, labels] = reverse_prob[:, labels]
    modified_log_probs = log_reverse_prob.clone()
    modified_log_probs[:, labels] = log_prob[:, labels]
    return -torch.sum(modified_probs * modified_log_probs, dim=dim, keepdim=keepdim)


def collect_prob(dataset, model, model_name="ResNet18"):
    print("Collecting performance..")

    if dataset is None:
        return np.zeros([0, 10]), np.zeros([0])

    if dataset is not None:
        probs = []
        # labels = []

        output = model.predict(dataset, verbose=2)
        if model_name in ["MitB0"]:
            output = output.logits

        prob = tf.nn.softmax(
            output,
            axis=1
        )

        probs.append(prob)
        if model_name not in ["MitB0"]:
            labels = np.concatenate([y for _, y in dataset], axis=0)
        else:
            labels = np.concatenate([np.squeeze(y) for _, y in dataset], axis=0)
        numpy_prob = tf.concat(prob, axis=0).numpy()
        # print(np.shape(labels))
        # print(np.shape(numpy_prob))
        return numpy_prob, labels
    else:
        return None, None


def SVC_fit_predict(shadow_train, shadow_test, target_train, target_test):
    n_shadow_train = shadow_train.shape[0]
    n_shadow_test = shadow_test.shape[0]
    n_target_train = target_train.shape[0]
    n_target_test = target_test.shape[0]

    X_shadow = (
        torch.cat([shadow_train, shadow_test])
        .cpu()
        .numpy()
        .reshape(n_shadow_train + n_shadow_test, -1)
    )
    Y_shadow = np.concatenate([np.ones(n_shadow_train), np.zeros(n_shadow_test)])

    clf = SVC(C=3, gamma="auto", kernel="rbf")
    clf.fit(X_shadow, Y_shadow)

    # MIA-efficacy: the number of forgetting data examples classified as non-training.

    accs = []

    if n_target_train > 0:
        X_target_train = target_train.cpu().numpy().reshape(n_target_train, -1)
        acc_train = clf.predict(X_target_train).mean()
        accs.append(acc_train)
        # target_train --> original train, retrained/resumed non-train
        # train 1
        # non train 0
        # low accuracy means most of the data are predicted to be non-training
        # original: should be high
        # retrained/resumed: should be low

    if n_target_test > 0:
        X_target_test = target_test.cpu().numpy().reshape(n_target_test, -1)
        acc_test = 1 - clf.predict(X_target_test).mean()
        accs.append(acc_test)

        # target_train --> original train, retrained/resumed non-train
        # train 1
        # non train 0
        # low accuracy means most of the data are predicted to be training
        # original: should be low
        # retrained/resumed: should be high

    return np.mean(accs)


def SVC_MIA(shadow_train, target_train, target_test, shadow_test, model, model_name):
    shadow_train_prob, shadow_train_labels = collect_prob(shadow_train, model, model_name)
    shadow_test_prob, shadow_test_labels = collect_prob(shadow_test, model, model_name)

    target_train_prob, target_train_labels = collect_prob(target_train, model, model_name)
    target_test_prob, target_test_labels = collect_prob(target_test, model, model_name)

    shadow_train_prob = torch.from_numpy(shadow_train_prob)
    shadow_train_labels = torch.from_numpy(shadow_train_labels)
    shadow_test_prob = torch.from_numpy(shadow_test_prob)
    shadow_test_labels = torch.from_numpy(shadow_test_labels)
    target_train_prob = torch.from_numpy(target_train_prob)
    target_train_labels = torch.from_numpy(target_train_labels)

    target_test_prob = torch.from_numpy(target_test_prob)
    target_test_labels = torch.from_numpy(target_test_labels)

    # shadow_train_corr = (
    #     torch.argmax(shadow_train_prob, axis=1) == shadow_train_labels
    # ).int()
    # shadow_test_corr = (
    #     torch.argmax(shadow_test_prob, axis=1) == shadow_test_labels
    # ).int()
    # target_train_corr = (
    #         torch.argmax(target_train_prob, axis=1) == target_train_labels
    #     ).int()
    # target_test_corr = (
    #     torch.argmax(target_test_prob, axis=1) == target_test_labels
    # ).int()

    shadow_train_conf = torch.gather(shadow_train_prob, 1, shadow_train_labels[:, None])
    shadow_test_conf = torch.gather(shadow_test_prob, 1, shadow_test_labels[:, None])
    target_train_conf = torch.gather(target_train_prob, 1, target_train_labels[:, None])
    target_test_conf = torch.gather(target_test_prob, 1, target_test_labels[:, None])

    # shadow_train_entr = entropy(shadow_train_prob)
    # shadow_test_entr = entropy(shadow_test_prob)

    # target_train_entr = entropy(target_train_prob)
    # target_test_entr = entropy(target_test_prob)

    # shadow_train_m_entr = m_entropy(shadow_train_prob, shadow_train_labels)
    # shadow_test_m_entr = m_entropy(shadow_test_prob, shadow_test_labels)
    # if target_train is not None:
    #     target_train_m_entr = m_entropy(target_train_prob, target_train_labels)
    # else:
    #     target_train_m_entr = target_train_entr
    # if target_test is not None:
    #     target_test_m_entr = m_entropy(target_test_prob, target_test_labels)
    # else:
    #     target_test_m_entr = target_test_entr

    # acc_corr = SVC_fit_predict(
    #     shadow_train_corr, shadow_test_corr, target_train_corr, target_test_corr
    # )
    acc_corr = 0.0
    acc_conf = SVC_fit_predict(
        shadow_train_conf, shadow_test_conf, target_train_conf, target_test_conf
    )
    # acc_entr = SVC_fit_predict(
    #     shadow_train_entr, shadow_test_entr, target_train_entr, target_test_entr
    # )
    acc_entr = 0.0
    # acc_m_entr = SVC_fit_predict(
    #     shadow_train_m_entr, shadow_test_m_entr, target_train_m_entr, target_test_m_entr
    # )
    acc_m_entr = 0.0
    # acc_prob = SVC_fit_predict(
    #     shadow_train_prob, shadow_test_prob, target_train_prob, target_test_prob
    # )
    acc_prob = 0.0
    m = {
        "correctness": acc_corr,
        "confidence": acc_conf,
        "entropy": acc_entr,
        "m_entropy": acc_m_entr,
        "prob": acc_prob,
    }
    print(m)
    return m


def JSDiv(p, q):
    p = torch.from_numpy(p)
    q = torch.from_numpy(q)
    m = (p + q) / 2
    js_div_value = 0.5 * F.kl_div(torch.log(p), m) + 0.5 * F.kl_div(torch.log(q), m)
    return js_div_value.item()


# ZRF/UnLearningScore
def UnLearningScore(tmodel, gold_model, forget_dl):
    # model_preds = []
    # gold_model_preds = []
    # with torch.no_grad():
    #     for batch in forget_dl:
    #         x, y, cy = batch
    #         x = x.to(device)
    #         model_output = tmodel(x)
    #         gold_model_output = gold_model(x)
    #         model_preds.append(F.softmax(model_output, dim=1).detach().cpu())
    #         gold_model_preds.append(F.softmax(gold_model_output, dim=1).detach().cpu())

    model_preds, _ = collect_prob(dataset=forget_dl, model=tmodel)
    gold_model_preds, _ = collect_prob(dataset=forget_dl, model=gold_model)

    # model_preds = torch.cat(model_preds, axis=0)
    # gold_model_preds = torch.cat(gold_model_preds, axis=0)
    return 1 - JSDiv(model_preds, gold_model_preds)


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

    ds_retain = ds_retain.shuffle(60000).take(n)
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

    print("-- Original --")
    SVC_MIA(shadow_train=ds_retain,
            shadow_test=test_ds_1,
            target_train=forgetting_ds,
            target_test=None,
            model=simple_cnn)
    simple_cnn.compile(optimizer='sgd',
                         loss=tf.keras.losses.SparseCategoricalCrossentropy(
                             from_logits=True),
                         metrics=['accuracy'])
    print("-- Retrained --")
    saved_checkpoint = tf.keras.saving.load_model(model_checkpoint_retrained_dir)
    retrained_weights = saved_checkpoint.get_weights()
    simple_cnn.set_weights(retrained_weights)
    simple_cnn.compile(optimizer='sgd',
                       loss=tf.keras.losses.SparseCategoricalCrossentropy(
                           from_logits=True),
                       metrics=['accuracy'])
    SVC_MIA(shadow_train=ds_retain,
            shadow_test=test_ds_1,
            target_train=forgetting_ds,
            target_test=None,
            model=simple_cnn)

    for i in range(25, 45):
        print(f"-- Resumed {i} --")
        model_checkpoint_resumed_dir = os.path.join("model_checkpoints_resumed",
                                                    config_dir, algorithm,
                                                    f"client{cid}",
                                                    "checkpoints",
                                                    f"R_{i}")
        saved_checkpoint = tf.keras.saving.load_model(model_checkpoint_resumed_dir)
        resumed_weights = saved_checkpoint.get_weights()
        simple_cnn.set_weights(resumed_weights)
        simple_cnn.compile(optimizer='sgd',
                             loss=tf.keras.losses.SparseCategoricalCrossentropy(
                                 from_logits=True),
                             metrics=['accuracy'])
        SVC_MIA(shadow_train=ds_retain,
            shadow_test=test_ds_1,
            target_train=forgetting_ds,
            target_test=None,
            model=simple_cnn)


if __name__ == "__main__":
    main()