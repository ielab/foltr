# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import sys
sys.path.append('../')

from matplotlib.pylab import plt
from multiprocessing import Pool
import itertools
import numpy as np
import torch
from foltr.data.datasets import MqDataset
from foltr.client.click_simulate import NAVIGATIONAL_MODEL, INFORMATIONAL_MODEL, PERFECT_MODEL
from foltr.client.metrics import PrivatizedMaxRR, ExpectedMaxRR
from foltr.click_optimize import train_uniform
from foltr.client.rankers import LinearRanker, TwoLayerRanker
import scipy
import json
import seaborn as sns
from util import smoothen_trajectory
from pylab import rcParams


seed = 7
torch.manual_seed(seed)
np.random.seed(seed)

all_click_models = [NAVIGATIONAL_MODEL, INFORMATIONAL_MODEL, PERFECT_MODEL]

data2008 = MqDataset.from_path(root_path="../../data/MQ2008/", n_features=46, n_folds=5, cache_root="../../cache/")

linear_ranker_generator = lambda: LinearRanker(46)
two_layer_ranker_generator = lambda: TwoLayerRanker(46, 10)

common_params = dict(online_metric=PrivatizedMaxRR(1.0),
                     n_clients=2000, #2000
                     sessions_budget=2000000,
                     seed=seed,
                     sessions_per_feedback=4,
                     antithetic=True,
                     lr=1e-3,
                     noise_std=1e-2)


def do_task(task):
    fold_id, click_model, ranker_generator_id = task
    # workaround pickle
    ranker_generator = [linear_ranker_generator, two_layer_ranker_generator][ranker_generator_id]
    fold = data2008.folds[fold_id]
    params = common_params.copy()
    params.update(dict(click_model=click_model, ranker_generator=ranker_generator))

    train_result = train_uniform(params, fold.train)
    return train_result


tasks = itertools.product(range(len(data2008.folds)),
                          [NAVIGATIONAL_MODEL, INFORMATIONAL_MODEL, PERFECT_MODEL],
                          range(2))
tasks = list(tasks)

n_cpu = min(80, len(tasks))
with Pool(n_cpu) as p:
    results = p.map(do_task, tasks)

click_model2sessions2trajectory = [{}, {}]
for task, trajectory in zip(tasks, results):
    fold_id, click_model, ranker_id = task
    click_model = click_model.name
    if click_model not in click_model2sessions2trajectory[ranker_id]:
        click_model2sessions2trajectory[ranker_id][click_model] = {}
    click_model2sessions2trajectory[ranker_id][click_model][fold_id] = trajectory

exp_results = np.array(click_model2sessions2trajectory)
np.save("../results/v0_mq2008_foltr_results_2000clients_p1.0.npy", exp_results)
