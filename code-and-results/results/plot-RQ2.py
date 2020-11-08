from matplotlib.pylab import plt
import numpy as np
from foltr.client.click_simulate import CcmClickModel
from foltr.client.metrics import PrivatizedMetric, ExpectedMetric
import seaborn as sns
from util import smoothen_trajectory
from pylab import rcParams
seed=7
PERFECT_MODEL = CcmClickModel(click_relevance={0: 0.0, 1: 0.2, 2: 0.4, 3: 0.8, 4: 1.0},
                              stop_relevance={0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}, name="Perfect", depth=10)
NAVIGATIONAL_MODEL = CcmClickModel(click_relevance={0: 0.05, 1: 0.3, 2: 0.5, 3: 0.7, 4: 0.95},
                                   stop_relevance={0: 0.2, 1: 0.3, 2: 0.5, 3: 0.7, 4: 0.9}, name="Navigational", depth=10)
INFORMATIONAL_MODEL = CcmClickModel(click_relevance={0: 0.4, 1: 0.6, 2: 0.7, 3: 0.8, 4: 0.9},
                                    stop_relevance={0: 0.1, 1: 0.2, 2: 0.3, 3: 0.4, 4: 0.5}, name="Informational", depth=10)
# set parameters here
dataset = 'yahoo'
metric = "MRR"
n_clients = 2000
p = '0.9'
do_linear = False

if dataset == 'mq2007':
    if metric == "MRR":
        foltr_path1 = "./foltr-results/RQ2_mq2007_MaxRR_{}clients_p{}.npy".format(2000, p)
        foltr_path2 = "./foltr-results/RQ2_mq2007_MaxRR_{}clients_p{}.npy".format(1000, p)
        foltr_path3 = "./foltr-results/RQ2_mq2007_MaxRR_{}clients_p{}.npy".format(50, p)
        # foltr_path4 = "./foltr-results/RQ2_mq2007_MaxRR_{}clients_p{}.npy".format(4000, p)
elif dataset == 'mq2008':
    if metric == "MRR":
        foltr_path1 = "./foltr-results/RQ2_mq2008_MaxRR_{}clients_p{}.npy".format(2000, p)
        foltr_path2 = "./foltr-results/RQ2_mq2008_MaxRR_{}clients_p{}.npy".format(1000, p)
        foltr_path3 = "./foltr-results/RQ2_mq2008_MaxRR_{}clients_p{}.npy".format(50, p)
elif dataset == 'mslr10k':
    if metric == "MRR":
        foltr_path1 = "./foltr-results/RQ2_mslr10k_MaxRR_{}clients_p{}.npy".format(2000, p)
        foltr_path2 = "./foltr-results/RQ2_mslr10k_MaxRR_{}clients_p{}.npy".format(1000, p)
        foltr_path3 = "./foltr-results/RQ2_mslr10k_MaxRR_{}clients_p{}.npy".format(50, p)
elif dataset == 'yahoo':
    if metric == "MRR":
        foltr_path1 = "./foltr-results/RQ2_yahoo_MaxRR_{}clients_p{}.npy".format(2000, p)
        foltr_path2 = "./foltr-results/RQ2_yahoo_MaxRR_{}clients_p{}.npy".format(1000, p)
        foltr_path3 = "./foltr-results/RQ2_yahoo_MaxRR_{}clients_p{}.npy".format(50, p)


foltr1 = np.load(foltr_path1, allow_pickle=True)
foltr2 = np.load(foltr_path2, allow_pickle=True)
foltr3 = np.load(foltr_path3, allow_pickle=True)
common_params = dict(online_metric=ExpectedMetric(1.0),
                     n_clients=n_clients,
                     sessions_budget=2000000,
                     seed=seed,
                     sessions_per_feedback=4,
                     antithetic=True,
                     lr=1e-3,
                     noise_std=1e-2)

click_model2sessions2trajectory1 = foltr1.tolist()
click_model2sessions2trajectory2 = foltr2.tolist()
click_model2sessions2trajectory3 = foltr3.tolist()

sns.set(style="darkgrid")
plt.close('all')
rcParams['figure.figsize'] = 22, 4
f, ax = plt.subplots(nrows=1, ncols=3, sharex=True)

linear1, two_layer1 = click_model2sessions2trajectory1
linear2, two_layer2 = click_model2sessions2trajectory2
linear3, two_layer3 = click_model2sessions2trajectory3

m = common_params['sessions_per_feedback'] * common_params['n_clients']
mid = 0
for row, model in enumerate([PERFECT_MODEL, NAVIGATIONAL_MODEL, INFORMATIONAL_MODEL]):
    Linear_ys1 = np.zeros(250)
    Linear_ys2 = np.zeros(500)
    Linear_ys3 = np.zeros(10000)

    Neural_ys1 = np.zeros(250)
    Neural_ys2 = np.zeros(500)
    Neural_ys3 = np.zeros(10000)

    for fold_id1, fold_trajectories1 in linear1[model.name].items():

        if model.name == "Informational":
            click_model = "informational"
        if model.name == "Navigational":
            click_model = "navigational"
        if model.name == "Perfect":
            click_model = "perfect"
            # data = [sum(group) / 40 for group in zip(*[iter(data)] * 40)]

        if metric == "DCG":
            linear_ys1 = np.array(fold_trajectories1.ndcg_clients) / n_clients

        else:
            linear_ys1 = np.array(fold_trajectories1.batch_metrics)
            neural_ys1 = np.array(two_layer1[model.name][fold_id1].batch_metrics)

        Linear_ys1 += linear_ys1
        Neural_ys1 += neural_ys1

    for fold_id2, fold_trajectories2 in linear2[model.name].items():

        if model.name == "Informational":
            click_model = "informational"
        if model.name == "Navigational":
            click_model = "navigational"
        if model.name == "Perfect":
            click_model = "perfect"
            # data = [sum(group) / 40 for group in zip(*[iter(data)] * 40)]

        if metric == "DCG":
            linear_ys2 = np.array(fold_trajectories2.ndcg_clients) / n_clients

        else:
            linear_ys2 = np.array(fold_trajectories2.batch_metrics)
            neural_ys2 = np.array(two_layer2[model.name][fold_id1].batch_metrics)

        Linear_ys2 += linear_ys2
        Neural_ys2 += neural_ys2

    for fold_id3, fold_trajectories3 in linear3[model.name].items():

        if model.name == "Informational":
            click_model = "informational"
        if model.name == "Navigational":
            click_model = "navigational"
        if model.name == "Perfect":
            click_model = "perfect"
            # data = [sum(group) / 40 for group in zip(*[iter(data)] * 40)]

        if metric == "DCG":
            linear_ys3 = np.array(fold_trajectories3.ndcg_clients) / n_clients

        else:
            linear_ys3 = np.array(fold_trajectories3.batch_metrics)
            neural_ys3 = np.array(two_layer3[model.name][fold_id1].batch_metrics)

        Linear_ys3 += linear_ys3
        Neural_ys3 += neural_ys3

    if dataset != "yahoo":
        Linear_ys1 /= 5
        Neural_ys1 /= 5
        Linear_ys2 /= 5
        Neural_ys2 /= 5
        Linear_ys3 /= 5
        Neural_ys3 /= 5

    Linear_ys2 = [sum(group) / 2 for group in zip(*[iter(Linear_ys2)] * 2)]
    Neural_ys2 = [sum(group) / 2 for group in zip(*[iter(Neural_ys2)] * 2)]

    Linear_ys3 = [sum(group) / 40 for group in zip(*[iter(Linear_ys3)] * 40)]
    Neural_ys3 = [sum(group) / 40 for group in zip(*[iter(Neural_ys3)] * 40)]

    Linear_ys1 = smoothen_trajectory(Linear_ys1, group_size=4)
    Linear_ys2 = smoothen_trajectory(Linear_ys2, group_size=4)
    Linear_ys3 = smoothen_trajectory(Linear_ys3, group_size=4)

    Neural_ys1 = smoothen_trajectory(Neural_ys1, group_size=4)
    Neural_ys2 = smoothen_trajectory(Neural_ys2, group_size=4)
    Neural_ys3 = smoothen_trajectory(Neural_ys3, group_size=4)

    a = ax[mid]
    xs = np.array(range(len(Linear_ys1))) * m * 1e-3
    a.plot(xs, Linear_ys1, label=f"Linear, n_client=2000", color='C0')
    a.plot(xs, Linear_ys2, label=f"Linear, n_client=1000", color='C0', linestyle='dashed', marker='x', markevery=15, markersize=6)
    a.plot(xs, Linear_ys3, label=f"Linear, n_client=50", color='C0', linestyle='dashed', marker='o', markevery=15, markersize=6)

    xs = np.array(range(len(Linear_ys1))) * m * 1e-3
    a.plot(xs, Neural_ys1, label=f"Neural, n_client=2000", color='C1')
    a.plot(xs, Neural_ys2, label=f"Neural, n_client=1000", color='C1', linestyle='dashed', marker='x', markevery=15, markersize=6)
    a.plot(xs, Neural_ys3, label=f"Neural, n_client=50", color='C1', linestyle='dashed', marker='o', markevery=15, markersize=6)

    ax[mid].set_title(f"{model.name}")

    if metric == "MRR":
        ax[0].set_ylabel("Mean batch MaxRR")
    # ax[mid].set_ylim([0.35, 0.74])
    ax[mid].legend(loc='lower right', ncol=2, fontsize=7)
    mid += 1

ax[1].set_xlabel("# interactions, thousands")

plt.show()
# f.savefig(f'./plots/mslr10k_foltr_client_both_p0.9.png', bbox_inches='tight')
