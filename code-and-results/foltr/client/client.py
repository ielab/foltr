# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Optional, NamedTuple
import torch
import copy
import numpy as np
from .rankers import perturb_model
from .metrics import PrivatizedMetric, MetricValue
#from .click_simulate import CcmClickModel
from .click_simulate import PbmClickModel
from foltr.data.datasets import DataSplit


ClientMessage = NamedTuple("ClientMessage", [
    ("seed", int),
    ("metric", MetricValue),
    ("antithetic_metric", Optional[MetricValue])])


class RankingClient:
    """
    Instances of this class emulate clients.
    """

    def __init__(self, dataset: DataSplit, init_model: torch.nn.Module, seed: int,
                 click_model: PbmClickModel, #CcmClickModel
                 metric: PrivatizedMetric, antithetic: bool,
                 noise_std: float):
        """
        :param dataset: A DataSplit, representing a (query -> {document relevances, document features}) mapping
                for the queries the client can submit
        :param init_model: A ranking model
        :param seed: Random seed
        :param click_model: A click model that emulate the user's behaviour; together with a click metric it is used
                to reflect the ranking quality
        :param metric: A metric instance, applied on the simulated click return the measured quality
        :param antithetic: Indicates if anthithetic variates are used; if so, each model perturbation is applied together
                with its antithetic
        :param noise_std: Standard deviation of the noise applied to get the perturbed model
        """
        self.dataset = dataset
        self.model = copy.deepcopy(init_model)
        self.random_state = np.random.RandomState(seed)
        self.click_model = click_model
        self.metric = metric
        self.antithetic = antithetic
        self.noise_std = noise_std

        self.unique_queries = list(dataset.keys())

    def update_model(self, model: torch.nn.Module) -> None:
        """
        Updates the client-side model
        :param model: The new model
        """
        self.model = copy.deepcopy(model)

    #evaluation method: NDCG@k
    def eval_ranking_ndcg(self, ranking: np.ndarray, k = 10) -> float:

        dcg = 0.0
        rel_set = []
        rel_set = sorted(ranking.copy().tolist(), reverse=True)

        for i in range(0, min(k, ranking.shape[0])):
            #document relevance label
            r = ranking[i]
            dcg += ((2 ** r - 1) / np.log2(i + 2))

        n = ranking.shape[0] if ranking.shape[0] < k else k
        idcg = 0
        for i in range(n):
            idcg += ((2 ** rel_set[i] - 1) / np.log2(i + 2))

        #deal with invalid value
        if idcg == 0:
            ndcg = 0.0
        else:
            ndcg = dcg / idcg

        return ndcg

    #evaluation method: MRR
    def eval_ranking_mrr(self, ranking: np.ndarray, k = 10) -> float:

        rr = 0.0
        got_rr = False

        for i in range(0, min(k, ranking.shape[0])):
            # document relevance label
            r = ranking[i]
            if r in {1, 2} and got_rr == False:
                rr = 1/(i+1)
                got_rr = True

        return rr

    def ranker_metrics(self, n_interactions: int, ranker: torch.nn.Module) -> MetricValue:
        """
        Runs submits queries to a ranking model and gets its performance
        :param n_interactions: Number of interactions to perform
        :param ranker: The ranker to be evaluated
        :return: The estimate of the metric value
        """
        ranker.eval()
        per_interaction_metric = []
        per_interaction_client_ndcg = []
        per_interaction_client_mrr = []
        for _ in range(n_interactions): # sessions_per_feedback
            qid = self.random_state.randint(0, len(self.unique_queries))
            qid = self.unique_queries[qid]
            per_query = self.dataset[qid]
            features = per_query.features
            features = torch.from_numpy(features).float()
            with torch.no_grad():
                ranking_scores = ranker.forward(features).numpy()[:, 0]
            ranking_order = np.argsort(ranking_scores)[::-1]
            ranking_user_sees = per_query.relevance_labels[ranking_order] # relevance label  np.ndarray
            per_interaction_client_ndcg.append(self.eval_ranking_ndcg(ranking_user_sees))
            per_interaction_client_mrr.append(self.eval_ranking_mrr(ranking_user_sees))
            clicks = self.click_model(ranking_user_sees, self.random_state)
            per_interaction_metric.append(self.metric(clicks)) # privatized dcg & non_privatized dcg
        mean_privatized_metric = np.mean(
            [x.privatized for x in per_interaction_metric])
        mean_true_metric = np.mean(
            [x.non_privatized for x in per_interaction_metric])
        mean_client_ndcg = np.mean(per_interaction_client_ndcg)
        mean_client_mrr = np.mean(per_interaction_client_mrr)

        return MetricValue(privatized=mean_privatized_metric, non_privatized=mean_true_metric), mean_client_ndcg, mean_client_mrr

    def get_click_feedback(self, n_interactions: int) -> ClientMessage:
        """
        Runs an interaction simulation within the client and generate the message from the client
        :param n_interactions: Number of interactions to simulate
        :return: The client's communication message: a tuple of a random seed and the metric value. In case antithetic
                variates are used, we return the both variates
        """
        seed = self.random_state.randint(0, 2**32)
        self.random_state.seed(seed)

        if not self.antithetic:
            perturbed_model = perturb_model(
                self.model, seed, self.noise_std, swap=False)
            metric, ndcg, mrr = self.ranker_metrics(n_interactions, perturbed_model)
            message = ClientMessage(
                seed=seed, metric=metric, antithetic_metric=None)
        else:
            assert n_interactions % 2 == 0
            model_direct = perturb_model(
                self.model, seed, self.noise_std, swap=False)
            metric_direct, ndcg_direct, mrr_direct = self.ranker_metrics(
                n_interactions // 2, model_direct)
            model_anti = perturb_model(
                self.model, seed, self.noise_std, swap=True)
            metric_anti, ndcg_anti, mrr_anti = self.ranker_metrics(n_interactions // 2, model_anti)
            message = ClientMessage(
                seed=seed, metric=metric_direct, antithetic_metric=metric_anti)
            ndcg = (ndcg_direct + ndcg_anti) / 2
            mrr = (mrr_direct + mrr_anti) / 2

        return message, ndcg, mrr

