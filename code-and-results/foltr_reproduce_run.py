import sys
sys.path.append('../')
from multiprocessing import Pool
import itertools
import numpy as np
import torch
from foltr.data.datasets import MqDataset
from foltr.data.LetorDataset import LetorDataset
from foltr.client.click_simulate import PbmClickModel, CcmClickModel
from foltr.client.metrics import PrivatizedMetric, ExpectedMetric
from foltr.click_optimize import train_uniform
from foltr.client.rankers import LinearRanker, TwoLayerRanker


seed = 7
torch.manual_seed(seed)
np.random.seed(seed)
PERFECT_MODEL = CcmClickModel(click_relevance={0: 0.0, 1: 0.2, 2: 0.4, 3: 0.8, 4: 1.0},
                              stop_relevance={0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}, name="Perfect", depth=10)
NAVIGATIONAL_MODEL = CcmClickModel(click_relevance={0: 0.05, 1: 0.3, 2: 0.5, 3: 0.7, 4: 0.95},
                                   stop_relevance={0: 0.2, 1: 0.3, 2: 0.5, 3: 0.7, 4: 0.9}, name="Navigational", depth=10)
INFORMATIONAL_MODEL = CcmClickModel(click_relevance={0: 0.4, 1: 0.6, 2: 0.7, 3: 0.8, 4: 0.9},
                                    stop_relevance={0: 0.1, 1: 0.2, 2: 0.3, 3: 0.4, 4: 0.5}, name="Informational", depth=10)
# PERFECT_MODEL = CcmClickModel(click_relevance={0: 0.0, 1: 0.5, 2: 1.0},
#                               stop_relevance={0: 0.0, 1: 0.0, 2: 0.0}, name="Perfect", depth=10)
# NAVIGATIONAL_MODEL = CcmClickModel(click_relevance={0: 0.05, 1: 0.5, 2: 0.95},
#                                    stop_relevance={0: 0.2, 1: 0.5, 2: 0.9}, name="Navigational", depth=10)
# INFORMATIONAL_MODEL = CcmClickModel(click_relevance={0: 0.4, 1: 0.7, 2: 0.9},
#                                     stop_relevance={0: 0.1, 1: 0.3, 2: 0.5}, name="Informational", depth=10)

n_features = 136 #136(MS),46(MQ)
n_folds = 5
debias = False
n_clients = 50
p = 1.0
all_click_models = [INFORMATIONAL_MODEL, NAVIGATIONAL_MODEL, PERFECT_MODEL]
metric = "DCG"
dataset_path = "./data/MSLR-WEB10K"
save_path = "./results/mslr10k_DCG_50clients_p1.0.npy"
# dataset_path = "./data/MQ2007"
# save_path = "./results/mq2007_DCG_2000clients_p0.5.npy"

linear_ranker_generator = lambda: LinearRanker(n_features)
two_layer_ranker_generator = lambda: TwoLayerRanker(n_features, 10)
dataset = MqDataset.from_path(dataset_path, n_features, n_folds, "./cache")
common_params = dict(online_metric=PrivatizedMetric(p, debias=debias, metric=metric),
                     n_clients=n_clients,
                     sessions_budget=2000000,
                     seed=seed,
                     sessions_per_feedback=4,
                     antithetic=True,
                     lr=1e-3,
                     noise_std=1e-2,
                     n_features=n_features,
                     dataset_path=dataset_path,
                     linear_ranker_generator=linear_ranker_generator,
                     two_layer_ranker_generator=two_layer_ranker_generator,
                     dataset=dataset)

def do_task(task):
    fold_id, click_model, ranker_generator_id = task
    params = common_params.copy()
    # workaround pickle

    ranker_generator = [params["linear_ranker_generator"], params["two_layer_ranker_generator"]][ranker_generator_id]
    fold = params["dataset"].folds[fold_id]
    params.update(dict(click_model=click_model, ranker_generator=ranker_generator))
    letordataset = LetorDataset("{}/Fold{}/test.txt".format(params['dataset_path'], fold_id+1),
                                params['n_features'], query_level_norm=True, #False(MQ),True(MS)
                                cache_root="./cache")

    task_info = "click_model:{} folder:{}".format(click_model.name, fold_id+1)
    train_result = train_uniform(params=params, traindata=fold.train, testdata=fold.test, letordataset=letordataset, message=task_info)
    return train_result


def run(path, tasks):
    tasks = list(tasks)

    n_cpu = min(5, len(tasks))
    with Pool(n_cpu) as p:
        results = p.map(do_task, tasks)
    #
    click_model2sessions2trajectory = [{}, {}]
    for task, trajectory in zip(tasks, results):
        fold_id, click_model, ranker_id = task
        click_model = click_model.name
        if click_model not in click_model2sessions2trajectory[ranker_id]:
            click_model2sessions2trajectory[ranker_id][click_model] = {}
        click_model2sessions2trajectory[ranker_id][click_model][fold_id] = trajectory

    exp_results = np.array(click_model2sessions2trajectory)
    np.save(path, exp_results)


if __name__ == "__main__":
    tasks = itertools.product(range(len(dataset.folds)),
                              [INFORMATIONAL_MODEL, NAVIGATIONAL_MODEL, PERFECT_MODEL],
                              range(2))
    run(save_path, tasks)
