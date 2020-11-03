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

dataset = 'yahoo'
metric = "MRR"
n_clients = 2000
p = '1.0'

if dataset == 'mq2007':
    if metric == "MRR":
        foltr_path = "./foltr-results/v0_mq2007_foltr_results_{}clients_p{}.npy".format(n_clients, p)
        foltr_path09 = "./foltr-results/v0_mq2007_foltr_results_{}clients_p{}.npy".format(n_clients, 0.9)
        foltr_path05 = "./foltr-results/v0_mq2007_foltr_results_{}clients_p{}.npy".format(n_clients, 0.5)
        oltr_path = "./PDGD/mq2007/mq2007_batch_update_size{}_grad_add/fold{}/{}_run1_ndcg.txt"

    else:
        foltr_path = "./v1-ndcg/mq2007_DCG_{}clients_p{}.npy".format(n_clients, p)

        oltr_path = "./PDGD/mq2007/mq2007_batch_update_size{}_grad_add/fold{}/{}_run1_cmrr.txt"

elif dataset == 'mslr10k':
    if metric == "MRR":
        foltr_path = "./foltr-results/v0_mslr_foltr_results_{}clients_p{}.npy".format(n_clients, p)
        foltr_path09 = "./foltr-results/v0_mslr_foltr_results_{}clients_p{}.npy".format(n_clients, 0.9)
        foltr_path05 = "./foltr-results/v0_mslr_foltr_results_{}clients_p{}.npy".format(n_clients, 0.5)
        oltr_path = "./PDGD/mslr10k/MSLR10K_batch_update_size{}_grad_add/fold{}/{}_run1_cmrr.txt"
    else:
        foltr_path = "./v1-ndcg/mslr10k_DCG_{}clients_p{}.npy".format(n_clients, p)
        oltr_path = "./PDGD/mslr10k/MSLR10K_batch_update_size{}_grad_add/fold{}/{}_run1_cndcg.txt"
else:
    if metric == "MRR":
        foltr_path = "./foltr-results/v0_yahoo_foltr_results_{}clients_p{}.npy".format(n_clients, p)
        foltr_path09 = "./foltr-results/v0_yahoo_foltr_results_{}clients_p{}.npy".format(n_clients, 0.9)
        foltr_path05 = "./foltr-results/v0_yahoo_foltr_results_{}clients_p{}.npy".format(n_clients, 0.5)
        oltr_path = "./PDGD/yahoo/yahoo_batch_update_size{}_grad_add/fold{}/{}_run1_cmrr.txt"

foltr = np.load(foltr_path, allow_pickle=True)
foltr09 = np.load(foltr_path09, allow_pickle=True)
foltr05 = np.load(foltr_path05, allow_pickle=True)
common_params = dict(online_metric=ExpectedMetric(1.0),
                     n_clients=n_clients,
                     sessions_budget=2000000,
                     seed=seed,
                     sessions_per_feedback=4,
                     antithetic=True,
                     lr=1e-3,
                     noise_std=1e-2)


click_model2sessions2trajectory = foltr.tolist()
click_model2sessions2trajectory09 = foltr09.tolist()
click_model2sessions2trajectory05 = foltr05.tolist()

sns.set(style="darkgrid")
plt.close('all')
rcParams['figure.figsize'] = 22, 4
f, ax = plt.subplots(nrows=1, ncols=3, sharex=True)

linear, two_layer = click_model2sessions2trajectory
linear09, two_layer09 = click_model2sessions2trajectory09
linear05, two_layer05 = click_model2sessions2trajectory05

m = common_params['sessions_per_feedback'] * common_params['n_clients']
mid = 0


for row, model in enumerate([PERFECT_MODEL, NAVIGATIONAL_MODEL, INFORMATIONAL_MODEL]):
    Linear_ys = np.zeros(250)
    Linear_ys09 = np.zeros(250)
    Linear_ys05 = np.zeros(250)
    Neural_ys = np.zeros(250)
    Neural_ys09 = np.zeros(250)
    Neural_ys05 = np.zeros(250)

    for fold_id, fold_trajectories in linear[model.name].items():
        if model.name == "Informational":
            click_model = "informational"
        if model.name == "Navigational":
            click_model = "navigational"
        if model.name == "Perfect":
            click_model = "perfect"

        if metric == "DCG":
            linear_ys = np.array(fold_trajectories.ndcg_clients) / n_clients
            neural_ys = np.array(two_layer[model.name][fold_id].ndcg_clients) / n_clients
        else:
            linear_ys = np.array(fold_trajectories.batch_metrics)
            neural_ys = np.array(two_layer[model.name][fold_id].batch_metrics)

        Linear_ys += linear_ys
        Neural_ys += neural_ys

    for fold_id, fold_trajectories in linear09[model.name].items():
        if model.name == "Informational":
            click_model = "informational"
        if model.name == "Navigational":
            click_model = "navigational"
        if model.name == "Perfect":
            click_model = "perfect"

        if metric == "DCG":
            linear_ys09 = np.array(fold_trajectories.ndcg_clients) / n_clients
            neural_ys09 = np.array(two_layer09[model.name][fold_id].ndcg_clients) / n_clients
        else:
            linear_ys09 = np.array(fold_trajectories.batch_metrics)
            neural_ys09 = np.array(two_layer09[model.name][fold_id].batch_metrics)

        Linear_ys09 += linear_ys09
        Neural_ys09 += neural_ys09

    for fold_id, fold_trajectories in linear05[model.name].items():
        if model.name == "Informational":
            click_model = "informational"
        if model.name == "Navigational":
            click_model = "navigational"
        if model.name == "Perfect":
            click_model = "perfect"

        if metric == "DCG":
            linear_ys05 = np.array(fold_trajectories.ndcg_clients) / n_clients
            neural_ys05 = np.array(two_layer05[model.name][fold_id].ndcg_clients) / n_clients
        else:
            linear_ys05 = np.array(fold_trajectories.batch_metrics)
            neural_ys05 = np.array(two_layer05[model.name][fold_id].batch_metrics)

        Linear_ys05 += linear_ys05
        Neural_ys05 += neural_ys05

    if dataset != "yahoo":
        Linear_ys /= 5
        Neural_ys /= 5
        Linear_ys09 /= 5
        Neural_ys09 /= 5
        Linear_ys05 /= 5
        Neural_ys05 /= 5

    Linear_ys = smoothen_trajectory(Linear_ys, group_size=4)
    Neural_ys = smoothen_trajectory(Neural_ys, group_size=4)
    Linear_ys09 = smoothen_trajectory(Linear_ys09, group_size=4)
    Linear_ys05 = smoothen_trajectory(Linear_ys05, group_size=4)
    Neural_ys09 = smoothen_trajectory(Neural_ys09, group_size=4)
    Neural_ys05 = smoothen_trajectory(Neural_ys05, group_size=4)
    xs = np.array(range(len(Linear_ys))) * m * 1e-3
    a = ax[mid]
    a.plot(xs, Linear_ys, label=f"Linear, p=1.0", color='C0')
    a.plot(xs, Linear_ys09, label=f"Linear, p=0.9", color='C0', linestyle='dashed', marker='x', markevery=10, markersize=4)
    a.plot(xs, Linear_ys05, label=f"Linear, p=0.5", color='C0', linestyle='dashed', marker='o', markevery=10, markersize=4)
    a.plot(xs, Neural_ys, label=f"Neural, p=1.0", color='C1')
    a.plot(xs, Neural_ys09, label=f"Neural, p=0.9", color='C1', linestyle='dashed', marker='x', markevery=10, markersize=4)
    a.plot(xs, Neural_ys05, label=f"Neural, p=0.5", color='C1', linestyle='dashed', marker='o', markevery=10, markersize=4)
    ax[mid].set_title(f"{model.name}")

    if metric == "MRR":
        ax[0].set_ylabel("Mean batch MaxRR")
    ax[mid].legend(loc='lower right', ncol=2, fontsize=12)

    mid += 1

ax[1].set_xlabel("# interactions, thousands")

plt.show()
# f.savefig('./plots/{}_foltr_c2000_ps.png'.format(dataset), bbox_inches='tight')