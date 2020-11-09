import pickle
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

# if you want to plot online nDCG performance, set parameters here
dataset = 'yahoo'
metric = "online nDCG" # plotting the 'online nDCG' performance; otherwise, using 'offline nDCG' as output
n_clients = 2000
p = '1.0'
do_PDGD = True
do_p = False
ranker = 'both'

# if you want to plot offline nDCG performance, set parameters here
# dataset = 'yahoo'
# metric = "offline nDCG" # plotting the 'online nDCG' performance; otherwise, using 'offline nDCG' as output
# n_clients = 2000
# p = '1.0'
# do_PDGD = True
# do_p = False
# ranker = 'both'


if dataset == 'mq2007':
    if metric == "online nDCG":
        foltr_path = "./foltr-results/RQ4_mq2007_nDCG_{}clients_p{}.npy".format(n_clients, p)
        foltr_path09 = "./foltr-results/RQ4_mq2007_nDCG_{}clients_p{}.npy".format(n_clients, 0.9)
        foltr_path05 = "./foltr-results/RQ4_mq2007_nDCG_{}clients_p{}.npy".format(n_clients, 0.5)
        oltr_path = "./PDGD/mq2007/mq2007_batch_update_size{}_grad_add/fold{}/{}_run1_cndcg.txt"
    elif metric == "offline nDCG":
        foltr_path = "./foltr-results/RQ4_mq2007_nDCG_{}clients_p{}.npy".format(n_clients, p)
        foltr_path09 = "./foltr-results/RQ4_mq2007_nDCG_{}clients_p{}.npy".format(n_clients, 0.9)
        foltr_path05 = "./foltr-results/RQ4_mq2007_nDCG_{}clients_p{}.npy".format(n_clients, 0.5)
        oltr_path = "./PDGD/mq2007/mq2007_batch_update_size{}_grad_add/fold{}/{}_run1_ndcg.txt"
elif dataset == 'mslr10k':
    if metric == "online nDCG":
        foltr_path = "./foltr-results/RQ4_mslr10k_nDCG_{}clients_p{}.npy".format(n_clients, p)
        foltr_path09 = "./foltr-results/RQ4_mslr10k_nDCG_{}clients_p{}.npy".format(n_clients, 0.9)
        foltr_path05 = "./foltr-results/RQ4_mslr10k_nDCG_{}clients_p{}.npy".format(n_clients, 0.5)
        oltr_path = "./PDGD/MSLR10K/MSLR10K_batch_update_size{}_grad_add/fold{}/{}_run1_cndcg.txt"
    elif metric == "offline nDCG":
        foltr_path = "./foltr-results/RQ4_mslr10k_nDCG_{}clients_p{}.npy".format(n_clients, p)
        foltr_path09 = "./foltr-results/RQ4_mslr10k_nDCG_{}clients_p{}.npy".format(n_clients, 0.9)
        foltr_path05 = "./foltr-results/RQ4_mslr10k_nDCG_{}clients_p{}.npy".format(n_clients, 0.5)
        oltr_path = "./PDGD/MSLR10K/MSLR10K_batch_update_size{}_grad_add/fold{}/{}_run1_ndcg.txt"
elif dataset == 'yahoo':
    if metric == "online nDCG":
        foltr_path = "./foltr-results/RQ4_yahoo_nDCG_{}clients_p{}.npy".format(n_clients, p)
        foltr_path09 = "./foltr-results/RQ4_yahoo_nDCG_{}clients_p{}.npy".format(n_clients, 0.9)
        foltr_path05 = "./foltr-results/RQ4_yahoo_nDCG_{}clients_p{}.npy".format(n_clients, 0.5)
        oltr_path = "./PDGD/yahoo/yahoo_batch_update_size{}_grad_add/fold{}/{}_run1_cndcg.txt"
    elif metric == "offline nDCG":
        foltr_path = "./foltr-results/RQ4_yahoo_nDCG_{}clients_p{}.npy".format(n_clients, p)
        foltr_path09 = "./foltr-results/RQ4_yahoo_nDCG_{}clients_p{}.npy".format(n_clients, 0.9)
        foltr_path05 = "./foltr-results/RQ4_yahoo_nDCG_{}clients_p{}.npy".format(n_clients, 0.5)
        oltr_path = "./PDGD/yahoo/yahoo_batch_update_size{}_grad_add/fold{}/{}_run1_ndcg.txt"

foltr = np.load(foltr_path, allow_pickle=True)
foltr09 = np.load(foltr_path09, allow_pickle=True)
foltr05 = np.load(foltr_path05, allow_pickle=True)

common_params = dict(online_metric=ExpectedMetric(1.0),
                     n_clients=2000,
                     sessions_budget=2000000,
                     seed=seed,
                     sessions_per_feedback=4,
                     antithetic=True,
                     lr=1e-3,
                     noise_std=1e-2)

click_model2sessions2trajectory = foltr.tolist()
click_model2sessions2trajectory09 = foltr09.tolist()
click_model2sessions2trajectory05 = foltr05.tolist()

# click_model2sessions2trajectory_b = b.tolist()

sns.set(style="darkgrid")
plt.close('all')
# rcParams['figure.figsize'] = 12, 2
rcParams['figure.figsize'] = 22, 5
f, ax = plt.subplots(nrows=1, ncols=3, sharex=True)

linear, two_layer = click_model2sessions2trajectory
linear09, two_layer09 = click_model2sessions2trajectory09
linear05, two_layer05 = click_model2sessions2trajectory05

m = common_params['sessions_per_feedback'] * common_params['n_clients']
mid = 0
for row, model in enumerate([PERFECT_MODEL, NAVIGATIONAL_MODEL, INFORMATIONAL_MODEL]):
    if n_clients == 2000:
        Linear_ys = np.zeros(250)
        Linear_ys09 = np.zeros(250)
        Linear_ys05 = np.zeros(250)
        Neural_ys = np.zeros(250)
        Neural_ys09 = np.zeros(250)
        Neural_ys05 = np.zeros(250)

        all_PDGD_ys = np.zeros(250)
    elif n_clients == 1000:
        Linear_ys = np.zeros(500)
        Linear_ys09 = np.zeros(500)
        Linear_ys05 = np.zeros(500)
        Neural_ys = np.zeros(500)
        Neural_ys09 = np.zeros(500)
        Neural_ys05 = np.zeros(500)
        all_PDGD_ys = np.zeros(500)
    else:
        Linear_ys = np.zeros(10000)
        Linear_ys09 = np.zeros(10000)
        Linear_ys05 = np.zeros(10000)
        Neural_ys = np.zeros(10000)
        Neural_ys09 = np.zeros(10000)
        Neural_ys05 = np.zeros(10000)
        all_PDGD_ys = np.zeros(10000)

    for fold_id, fold_trajectories in linear[model.name].items():
        if model.name == "Informational":
            click_model = "informational"
        if model.name == "Navigational":
            click_model = "navigational"
        if model.name == "Perfect":
            click_model = "perfect"

        with open(oltr_path.format(n_clients*4, fold_id+1, click_model),
                  "rb") as fp:
            PDGD_ys = pickle.load(fp)
            PDGD_ys = np.array(PDGD_ys)
            all_PDGD_ys += PDGD_ys
            # data = [sum(group) / 40 for group in zip(*[iter(data)] * 40)]

        if metric == "online nDCG":
            linear_ys = np.array(fold_trajectories.ndcg_clients) / n_clients
            neural_ys = np.array(two_layer[model.name][fold_id].ndcg_clients) / n_clients
        else:
            linear_ys = np.array(fold_trajectories.ndcg_server)
            neural_ys = np.array(two_layer[model.name][fold_id].ndcg_server)

        Linear_ys += linear_ys
        Neural_ys += neural_ys

    for fold_id, fold_trajectories in linear09[model.name].items():
        if model.name == "Informational":
            click_model = "informational"
        if model.name == "Navigational":
            click_model = "navigational"
        if model.name == "Perfect":
            click_model = "perfect"

        if metric == "online nDCG":
            linear_ys09 = np.array(fold_trajectories.ndcg_clients) / n_clients
            neural_ys09 = np.array(two_layer09[model.name][fold_id].ndcg_clients) / n_clients
        else:
            linear_ys09 = np.array(fold_trajectories.ndcg_server)
            neural_ys09 = np.array(two_layer09[model.name][fold_id].ndcg_server)

        Linear_ys09 += linear_ys09
        Neural_ys09 += neural_ys09

    for fold_id, fold_trajectories in linear05[model.name].items():
        if model.name == "Informational":
            click_model = "informational"
        if model.name == "Navigational":
            click_model = "navigational"
        if model.name == "Perfect":
            click_model = "perfect"

        if metric == "online nDCG":
            linear_ys05 = np.array(fold_trajectories.ndcg_clients) / n_clients
            neural_ys05 = np.array(two_layer05[model.name][fold_id].ndcg_clients) / n_clients
        else:
            linear_ys05 = np.array(fold_trajectories.ndcg_server)
            neural_ys05 = np.array(two_layer05[model.name][fold_id].ndcg_server)

        Linear_ys05 += linear_ys05
        Neural_ys05 += neural_ys05

    if dataset != "yahoo":
        Linear_ys /= 5
        Neural_ys /= 5
        Linear_ys09 /= 5
        Neural_ys09 /= 5
        Linear_ys05 /= 5
        Neural_ys05 /= 5
        all_PDGD_ys /= 5

    if n_clients == 1000:
        Linear_ys = [sum(group) / 2 for group in zip(*[iter(Linear_ys)] * 2)]
        Neural_ys = [sum(group) / 2 for group in zip(*[iter(Neural_ys)] * 2)]
        all_PDGD_ys = [sum(group) / 2 for group in zip(*[iter(all_PDGD_ys)] * 2)]
    elif n_clients == 50:
        Linear_ys = [sum(group) / 40 for group in zip(*[iter(Linear_ys)] * 40)]
        Neural_ys = [sum(group) / 40 for group in zip(*[iter(Neural_ys)] * 40)]
        all_PDGD_ys = [sum(group) / 40 for group in zip(*[iter(all_PDGD_ys)] * 40)]
    else:
        Linear_ys = smoothen_trajectory(Linear_ys, group_size=4)
        Neural_ys = smoothen_trajectory(Neural_ys, group_size=4)
        all_PDGD_ys = smoothen_trajectory(all_PDGD_ys, group_size=4)

    xs = np.array(range(len(Linear_ys))) * m * 1e-3
    a = ax[mid]

    if do_p:
        Linear_ys09 = smoothen_trajectory(Linear_ys09, group_size=4)
        Linear_ys05 = smoothen_trajectory(Linear_ys05, group_size=4)
        Neural_ys09 = smoothen_trajectory(Neural_ys09, group_size=4)
        Neural_ys05 = smoothen_trajectory(Neural_ys05, group_size=4)

        if ranker == 'linear':
            a.plot(xs, Linear_ys, label=f"p=1.0")
            a.plot(xs, Linear_ys09, label=f"p=0.9")
            a.plot(xs, Linear_ys05, label=f"p=0.5")
        elif ranker == 'neural':
            a.plot(xs, Neural_ys, label=f"p=1.0")
            a.plot(xs, Neural_ys09, label=f"p=0.9")
            a.plot(xs, Neural_ys05, label=f"p=0.5")
        else:
            a.plot(xs, Linear_ys, label=f"Linear, p=1.0", color='C0')
            a.plot(xs, Linear_ys09, label=f"Linear, p=0.9", color='C0', linestyle='dashed', marker='x', markevery=10,
                   markersize=4)
            a.plot(xs, Linear_ys05, label=f"Linear, p=0.5", color='C0', linestyle='dashed', marker='o', markevery=10,
                   markersize=4)
            a.plot(xs, Neural_ys, label=f"Neural, p=1.0", color='C1')
            a.plot(xs, Neural_ys09, label=f"Neural, p=0.9", color='C1', linestyle='dashed', marker='x', markevery=10,
                   markersize=4)
            a.plot(xs, Neural_ys05, label=f"Neural, p=0.5", color='C1', linestyle='dashed', marker='o', markevery=10,
                   markersize=4)
    else:
        a.plot(xs, Linear_ys, label=f"1-Layer")
        a.plot(xs, Neural_ys, label=f"2-Layer")

        if do_PDGD:
            xs = np.array(range(len(all_PDGD_ys))) * m * 1e-3
            a.plot(xs, all_PDGD_ys, label=f"PDGD")

    ax[mid].set_title(f"{model.name}")

    if metric != "MRR":
        ax[0].set_ylabel("Mean batch nDCG")
    else:
        ax[0].set_ylabel("Mean batch MaxRR")
    ax[mid].legend(loc='lower right', ncol=2, fontsize=8)

    mid += 1

ax[1].set_xlabel("# interactions, thousands")

plt.show()
# if do_PDGD:
#     f.savefig('./plots/{}_foltr_PDGD_{}_c{}_p{}.png'.format(dataset, metric, n_clients, p), bbox_inches='tight')
# if do_p:
#     f.savefig('./plots/{}_foltr_{}_{}_c{}_ps.png'.format(dataset, metric, ranker, n_clients, p), bbox_inches='tight')
#
