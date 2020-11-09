import pickle
from matplotlib.pylab import plt
import numpy as np
from foltr.client.click_simulate import CcmClickModel
from foltr.client.metrics import PrivatizedMetric, ExpectedMetric
import seaborn as sns
from util import smoothen_trajectory
from pylab import rcParams
seed = 7

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
p = '1.0'
do_PDGD = True

if dataset == 'mq2007':
    if metric == "MRR":
        foltr_path = "./foltr-results/RQ3_mq2007_MaxRR_{}clients_p{}.npy".format(n_clients, p)
        oltr_path = "./PDGD/mq2007/mq2007_batch_update_size{}_grad_add/fold{}/{}_run1_cmrr.txt"
elif dataset == 'mslr10k':
    if metric == "MRR":
        foltr_path = "./foltr-results/RQ3_mslr10k_MaxRR_{}clients_p{}.npy".format(n_clients, p)
        oltr_path = "./PDGD/MSLR10K/MSLR10K_batch_update_size{}_grad_add/fold{}/{}_run1_cmrr.txt"
elif dataset == 'yahoo':
    if metric == "MRR":
        foltr_path = "./foltr-results/RQ3_yahoo_MaxRR_{}clients_p{}.npy".format(n_clients, p)
        oltr_path = "./PDGD/yahoo/yahoo_batch_update_size{}_grad_add/fold{}/{}_run1_cmrr.txt"


foltr = np.load(foltr_path, allow_pickle=True)
common_params = dict(online_metric=ExpectedMetric(1.0),
                     n_clients=2000,
                     sessions_budget=2000000,
                     seed=seed,
                     sessions_per_feedback=4,
                     antithetic=True,
                     lr=1e-3,
                     noise_std=1e-2)

click_model2sessions2trajectory = foltr.tolist()

sns.set(style="darkgrid")
plt.close('all')
rcParams['figure.figsize'] = 21, 5
f, ax = plt.subplots(nrows=1, ncols=3, sharex=True)

linear, two_layer = click_model2sessions2trajectory

m = common_params['sessions_per_feedback'] * common_params['n_clients']
mid = 0
for row, model in enumerate([PERFECT_MODEL, NAVIGATIONAL_MODEL, INFORMATIONAL_MODEL]):
    Linear_ys = np.zeros(250)
    Neural_ys = np.zeros(250)
    all_PDGD_ys = np.zeros(250)
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

        if metric == "DCG":
            linear_ys = np.array(fold_trajectories.ndcg_clients) / n_clients
            neural_ys = np.array(two_layer[model.name][fold_id].ndcg_clients) / n_clients
        else:
            linear_ys = np.array(fold_trajectories.batch_metrics)
            neural_ys = np.array(two_layer[model.name][fold_id].batch_metrics)

        Linear_ys += linear_ys
        Neural_ys += neural_ys

        # if len(linear_ys) > 250:
        #     ys = [sum(group) / 40 for group in zip(*[iter(ys)] * 40)]

    if dataset != "yahoo":
        Linear_ys /= 5
        Neural_ys /= 5
        all_PDGD_ys /= 5

    Linear_ys = smoothen_trajectory(Linear_ys, group_size=4)
    Neural_ys = smoothen_trajectory(Neural_ys, group_size=4)

    # all_PDGD_ys = [sum(group) / 2 for group in zip(*[iter(all_PDGD_ys)] * 2)]

    all_PDGD_ys = smoothen_trajectory(all_PDGD_ys, group_size=4)
    xs = np.array(range(len(Linear_ys))) * m * 1e-3
    a = ax[mid]
    a.plot(xs, Linear_ys, label=f"Linear")
    a.plot(xs, Neural_ys, label=f"Neural")

    if do_PDGD:
        xs = np.array(range(len(all_PDGD_ys))) * m * 1e-3
        a.plot(xs, all_PDGD_ys, label=f"PDGD")
    ax[mid].set_title(f"{model.name}")

    if metric != "MRR":
        ax[0].set_ylabel("Mean batch nDCG")
    else:
        ax[0].set_ylabel("Mean batch MaxRR")
    ax[mid].legend(loc='lower right')

    mid += 1

ax[1].set_xlabel("# interactions, thousands")

plt.show()
# f.savefig('./plots/{}_foltr_PDGD_DCG_c2000_p1.0.png'.format(dataset), bbox_inches='tight')


