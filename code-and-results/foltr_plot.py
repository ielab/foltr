
import json
import numpy as np
import seaborn as sns
from matplotlib.pylab import plt
# from util import smoothen_trajectory
from pylab import rcParams

from foltr.client.click_simulate import PbmClickModel
from foltr.client.metrics import PrivatizedMetric

PERFECT_MODEL = PbmClickModel(click_relevance={0: 0.0, 1: 0.2, 2: 0.4, 3: 0.8, 4: 1.0},
                              stop_relevance={0: 0.0, 1: 0.0, 2: 0.0}, name="Perfect", depth=10)
NAVIGATIONAL_MODEL = PbmClickModel(click_relevance={0: 0.0, 1: 0.2, 2: 0.4, 3: 0.8, 4: 1.0},
                                   stop_relevance={0: 0.2, 1: 0.5, 2: 0.9}, name="Navigational", depth=10)
INFORMATIONAL_MODEL = PbmClickModel(click_relevance={0: 0.0, 1: 0.2, 2: 0.4, 3: 0.8, 4: 1.0},
                                    stop_relevance={0: 0.1, 1: 0.3, 2: 0.5}, name="Informational", depth=10)

# NOISY_MODEL = PbmClickModel(click_relevance={0: 0.4, 1: 0.6, 2: 0.7, 3: 0.8, 4: 0.9},
#                               stop_relevance={0: 0.0, 1: 0.0, 2: 0.0}, name="Noisy", depth=10)
seed = 7
common_params = dict(online_metric=PrivatizedMetric(1.0),
                     n_clients=2000,
                     sessions_budget=1000000,
                     seed=seed,
                     sessions_per_feedback=4,
                     antithetic=True,
                     lr=1e-3,
                     noise_std=1e-2)


# a = np.load("./results/mq2007_test.npy")
a = np.load("./results/mq2007_DCG_2000clients_p1.0.npy", allow_pickle=True)
click_model2sessions2trajectory = a.tolist()
linear, two_layer = click_model2sessions2trajectory
m = common_params['sessions_per_feedback'] * common_params['n_clients']


### load baseline results
# with open("baselines_ndcg.json", "r") as f:
#     baselines_ndcg = json.loads(f.read())
#
# with open("baselines_mrr.json", "r") as f:
#     baselines_mrr = json.loads(f.read())

### server ranker evaluation plot - ndcg
sns.set(style="darkgrid")
plt.close("all")
rcParams["figure.figsize"] = 10, 10
f, ax = plt.subplots(nrows=2, sharex=True)


for row, model in enumerate([NOISY_MODEL, PERFECT_MODEL]):
    metric_linear = dict()
    metric_two_layer = dict()
    l = []
    a = ax[row]
    for fold_id, fold_trajectories in linear[model.name].items():
        ys_linear = fold_trajectories.ndcg_server
        metric_linear[fold_id] = ys_linear
        l.append(len(ys_linear))
        ys_two_layer = two_layer[model.name][fold_id].ndcg_server
        metric_two_layer[fold_id] = ys_two_layer
        l.append(len(ys_two_layer))

    n = max(l)
    for key in metric_linear.keys():
        ys = metric_linear[key]
        if len(ys) < n:
            ys = ys + [0] * (n - len(ys))
            metric_linear[key] = ys
    for key in metric_two_layer.keys():
        ys = metric_two_layer[key]
        if len(ys) < n:
            ys = ys + [0] * (n - len(ys))
            metric_two_layer[key] = ys
    y_linear_avg = []
    y_two_layer_avg = []
    for i in range(0, n):
        temp_1 = 0
        for key in metric_linear.keys():
            temp_1 += metric_linear[key][i]
        y_linear_avg.append(temp_1 / 5.0)

        temp_2 = 0
        for key in metric_two_layer.keys():
            temp_2 += metric_two_layer[key][i]
        y_two_layer_avg.append(temp_2 / 5.0)

    # ys = smoothen_trajectory(ys, group_size=4)
    xs = np.linspace(0, common_params['sessions_budget'], len(y_linear_avg)) * 1e-3
    # xs = np.array(range(0, common_params['sessions_budget'], len(y_linear_avg)-1)) * 1e-3
    a.plot(xs, y_linear_avg, label=f"1-Layer")
    a.plot(xs, y_two_layer_avg, label=f"2-Layer")

    #         #ys = two_layer[model.name][fold_id].batch_metrics
    #         ys = two_layer[model.name][fold_id].ndcg
    #         #ys = smoothen_trajectory(ys, group_size=4)
    #         xs = np.array(range(len(ys))) * m * 1e-3
    #         a.plot(xs, ys, label=f"2-Layer")
    lsq = 0
    # for fold_id, fold_trajectories in linear[model.name].items():
    #     lsq += baselines_ndcg['lsq']['mq2007']['test'][model.name.lower()][fold_id]
    lsq = lsq / 5.0
    ys = np.array([lsq for _ in xs])
    # a.plot(xs, ys, label=f"MSE")

    #         svm_rank = baselines['svmrank']['mq2007']['train'][model.name][fold_id]
    #         ys = np.array([svm_rank for _ in xs])
    #         a.plot(xs, ys, label=f"SVM Rank")

    ax[row].set_title(f"{model.name} model")
ax[1].set_ylabel("ndcg_server")
ax[1].set_xlabel("# interactions, thousands")
ax[1].legend(loc='lower center')  # , bbox_to_anchor=(4.5, -0.6), ncol=1)

plt.show()
# f.savefig(f'./results/mq2007-server-ndcg.png')

### server ranker evaluation plot - mrr
sns.set(style="darkgrid")
plt.close("all")
rcParams["figure.figsize"] = 10, 10
f, ax = plt.subplots(nrows=2, sharex=True)

for row, model in enumerate([NOISY_MODEL, PERFECT_MODEL]):
    metric_linear = dict()
    metric_two_layer = dict()
    l = []
    a = ax[row]
    for fold_id, fold_trajectories in linear[model.name].items():
        ys_linear = fold_trajectories.mrr_server
        metric_linear[fold_id] = ys_linear
        l.append(len(ys_linear))
        ys_two_layer = two_layer[model.name][fold_id].mrr_server
        metric_two_layer[fold_id] = ys_two_layer
        l.append(len(ys_two_layer))

    n = max(l)
    for key in metric_linear.keys():
        ys = metric_linear[key]
        if len(ys) < n:
            ys = ys + [0] * (n - len(ys))
            metric_linear[key] = ys
    for key in metric_two_layer.keys():
        ys = metric_two_layer[key]
        if len(ys) < n:
            ys = ys + [0] * (n - len(ys))
            metric_two_layer[key] = ys
    y_linear_avg = []
    y_two_layer_avg = []
    for i in range(0, n):
        temp_1 = 0
        for key in metric_linear.keys():
            temp_1 += metric_linear[key][i]
        y_linear_avg.append(temp_1 / 5.0)

        temp_2 = 0
        for key in metric_two_layer.keys():
            temp_2 += metric_two_layer[key][i]
        y_two_layer_avg.append(temp_2 / 5.0)

    # ys = smoothen_trajectory(ys, group_size=4)
    xs = np.linspace(0, common_params['sessions_budget'], len(y_linear_avg)) * 1e-3
    a.plot(xs, y_linear_avg, label=f"1-Layer")
    a.plot(xs, y_two_layer_avg, label=f"2-Layer")

    #         #ys = two_layer[model.name][fold_id].batch_metrics
    #         ys = two_layer[model.name][fold_id].ndcg
    #         #ys = smoothen_trajectory(ys, group_size=4)
    #         xs = np.array(range(len(ys))) * m * 1e-3
    #         a.plot(xs, ys, label=f"2-Layer")
    lsq = 0
    # for fold_id, fold_trajectories in linear[model.name].items():
    #     lsq += baselines_mrr['lsq']['mq2007']['test'][model.name.lower()][fold_id]
    lsq = lsq / 5.0
    ys = np.array([lsq for _ in xs])
    # a.plot(xs, ys, label=f"MSE")

    #         svm_rank = baselines['svmrank']['mq2007']['train'][model.name][fold_id]
    #         ys = np.array([svm_rank for _ in xs])
    #         a.plot(xs, ys, label=f"SVM Rank")

    ax[row].set_title(f"{model.name} model")
ax[1].set_ylabel("mrr_server")
ax[1].set_xlabel("# interactions, thousands")
ax[1].legend(loc='lower center')  # , bbox_to_anchor=(4.5, -0.6), ncol=1)

plt.show()
# f.savefig(f'./results/mq2007-server-mrr.png')



### clients ranker evaluation plot - ndcg
sns.set(style="darkgrid")
plt.close("all")
rcParams["figure.figsize"] = 10, 10
f, ax = plt.subplots(nrows=2, sharex=True)

for row, model in enumerate([NOISY_MODEL, PERFECT_MODEL]):
    metric_linear = dict()
    metric_two_layer = dict()
    l = []
    a = ax[row]
    for fold_id, fold_trajectories in linear[model.name].items():
        ys_linear = fold_trajectories.ndcg_clients
        metric_linear[fold_id] = ys_linear
        l.append(len(ys_linear))
        ys_two_layer = two_layer[model.name][fold_id].ndcg_clients
        metric_two_layer[fold_id] = ys_two_layer
        l.append(len(ys_two_layer))

    n = max(l)
    for key in metric_linear.keys():
        ys = metric_linear[key]
        if len(ys) < n:
            ys = ys + [0] * (n - len(ys))
            metric_linear[key] = ys
    for key in metric_two_layer.keys():
        ys = metric_two_layer[key]
        if len(ys) < n:
            ys = ys + [0] * (n - len(ys))
            metric_two_layer[key] = ys
    y_linear_avg = []
    y_two_layer_avg = []
    for i in range(0, n):
        temp_1 = 0
        for key in metric_linear.keys():
            temp_1 += metric_linear[key][i]
        y_linear_avg.append(temp_1 / 5.0 / common_params['n_clients'])

        temp_2 = 0
        for key in metric_two_layer.keys():
            temp_2 += metric_two_layer[key][i]
        y_two_layer_avg.append(temp_2 / 5.0 / common_params['n_clients'])

    # ys = smoothen_trajectory(ys, group_size=4)
    xs = np.linspace(0, common_params['sessions_budget'], len(y_linear_avg)) * 1e-3
    a.plot(xs, y_linear_avg, label=f"1-Layer")
    a.plot(xs, y_two_layer_avg, label=f"2-Layer")

    #         #ys = two_layer[model.name][fold_id].batch_metrics
    #         ys = two_layer[model.name][fold_id].ndcg
    #         #ys = smoothen_trajectory(ys, group_size=4)
    #         xs = np.array(range(len(ys))) * m * 1e-3
    #         a.plot(xs, ys, label=f"2-Layer")
    lsq = 0
    # for fold_id, fold_trajectories in linear[model.name].items():
    #     lsq += baselines_ndcg['lsq']['mq2007']['test'][model.name.lower()][fold_id]
    lsq = lsq / 5.0
    ys = np.array([lsq for _ in xs])
    # a.plot(xs, ys, label=f"MSE")

    #         svm_rank = baselines['svmrank']['mq2007']['train'][model.name][fold_id]
    #         ys = np.array([svm_rank for _ in xs])
    #         a.plot(xs, ys, label=f"SVM Rank")

    ax[row].set_title(f"{model.name} model")
ax[1].set_ylabel("ndcg_clients")
ax[1].set_xlabel("# interactions, thousands")
ax[1].legend(loc='lower center')  # , bbox_to_anchor=(4.5, -0.6), ncol=1)

plt.show()
# f.savefig(f'./results/mq2007-clients-ndcg.png')

### clients ranker evaluation plot - mrr
sns.set(style="darkgrid")
plt.close("all")
rcParams["figure.figsize"] = 10, 10
f, ax = plt.subplots(nrows=2, sharex=True)

for row, model in enumerate([NOISY_MODEL, PERFECT_MODEL]):
    metric_linear = dict()
    metric_two_layer = dict()
    l = []
    a = ax[row]
    for fold_id, fold_trajectories in linear[model.name].items():
        ys_linear = fold_trajectories.mrr_clients
        metric_linear[fold_id] = ys_linear
        l.append(len(ys_linear))
        ys_two_layer = two_layer[model.name][fold_id].mrr_clients
        metric_two_layer[fold_id] = ys_two_layer
        l.append(len(ys_two_layer))

    n = max(l)
    for key in metric_linear.keys():
        ys = metric_linear[key]
        if len(ys) < n:
            ys = ys + [0] * (n - len(ys))
            metric_linear[key] = ys
    for key in metric_two_layer.keys():
        ys = metric_two_layer[key]
        if len(ys) < n:
            ys = ys + [0] * (n - len(ys))
            metric_two_layer[key] = ys
    y_linear_avg = []
    y_two_layer_avg = []
    for i in range(0, n):
        temp_1 = 0
        for key in metric_linear.keys():
            temp_1 += metric_linear[key][i]
        y_linear_avg.append(temp_1 / 5.0 / common_params['n_clients'])

        temp_2 = 0
        for key in metric_two_layer.keys():
            temp_2 += metric_two_layer[key][i]
        y_two_layer_avg.append(temp_2 / 5.0 / common_params['n_clients'])

    # ys = smoothen_trajectory(ys, group_size=4)
    xs = np.linspace(0, common_params['sessions_budget'], len(y_linear_avg)) * 1e-3
    a.plot(xs, y_linear_avg, label=f"1-Layer")
    a.plot(xs, y_two_layer_avg, label=f"2-Layer")

    #         #ys = two_layer[model.name][fold_id].batch_metrics
    #         ys = two_layer[model.name][fold_id].ndcg
    #         #ys = smoothen_trajectory(ys, group_size=4)
    #         xs = np.array(range(len(ys))) * m * 1e-3
    #         a.plot(xs, ys, label=f"2-Layer")
    lsq = 0
    # for fold_id, fold_trajectories in linear[model.name].items():
    #     lsq += baselines_mrr['lsq']['mq2007']['test'][model.name.lower()][fold_id]
    lsq = lsq / 5.0
    ys = np.array([lsq for _ in xs])
    # a.plot(xs, ys, label=f"MSE")

    #         svm_rank = baselines['svmrank']['mq2007']['train'][model.name][fold_id]
    #         ys = np.array([svm_rank for _ in xs])
    #         a.plot(xs, ys, label=f"SVM Rank")

    ax[row].set_title(f"{model.name} model")
ax[1].set_ylabel("mrr_clients")
ax[1].set_xlabel("# interactions, thousands")
ax[1].legend(loc='lower center')  # , bbox_to_anchor=(4.5, -0.6), ncol=1)

plt.show()
# f.savefig(f'./results/mq2007-clients-mrr.png')