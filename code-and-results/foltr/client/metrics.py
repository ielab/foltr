# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import NamedTuple, Union, List
import numpy as np
import random
import torch
from foltr.data import DataSplit
from .click_simulate import CcmClickModel


# we need both privatized and not: the first is actually used for the optimization, the second is used
# to get the figures
MetricValue = NamedTuple(
    'MetricValue', [('privatized', float), ('non_privatized', float)])


class PrivatizedMetric:
    """
    Implements a privatized MaxRR metric with a privatization noise.
    """

    def __init__(self, p: float, cutoff: int = 10, debias: bool = False, metric: str = "DCG"):
        """
        :param p: Privatization parameter, within [0, 1]. Essentially, the probability of the output not being corrupted
         - with p = 1.0 returns true MaxRR all the time.
        :param cutoff: Max depth of the ranked lists. The returned metric values take cutoff + 1 values.
        """
        self.debias = debias
        self.metric = metric
        self.p = p
        self.cutoff = cutoff
        # zero for the case when no click occurred
        self.possible_outputs = [0.0] + \
            [1.0 / (1.0 + r) for r in range(cutoff)]

    def __call__(self, clicks: Union[np.ndarray, List[float]]) -> MetricValue:
        """
        Calculates the metric value for the one-encoded clicks metrics.
        :param clicks: A numpy array encoding per-position click events with {0, 1}, e.g.
        [1, 0, 0, 0, 1] encodes clicks on positions 0 and 4 on a ranked list of length 5.
        :return: A MetricValue instance with privatized and true value metrics.
        Examples:
        >>> PrivatizedMaxRR(1.0)([1.0, 0.0, 0.0, 0.0])
        MetricValue(privatized=1.0, non_privatized=1.0)
        >>> PrivatizedMaxRR(1.0)([0.0, 0.0, 0.0, 0.0])
        MetricValue(privatized=0.0, non_privatized=0.0)
        """
        # using (DCG / MRR) as clicks metrics
        n_docs = len(clicks)

        if self.metric == "DCG":
            dcg = 0.0
            for i in range(0, min(self.cutoff, n_docs)):
                if self.debias:
                    dcg += ((2 ** clicks[i] - 1) / (np.log2(i + 2) * (1.0 / (i + 1))))
                else:
                    dcg += (2 ** clicks[i] - 1) / np.log2(i + 2)

            if np.random.random() < self.p:
                return MetricValue(privatized=dcg, non_privatized=dcg)
            else:
                while True:
                    sampled_value = self.get_random_privatized_dcg(self.cutoff)
                    if sampled_value != dcg:
                        return MetricValue(privatized=sampled_value, non_privatized=dcg)

        if self.metric == "MRR":
            reciprocal_rank = 0.0
            for i in range(0, min(self.cutoff, n_docs)):
                if clicks[i] > 0:
                    if self.debias:
                        reciprocal_rank = 1.0 / (1.0 + i)
                        break
                    else:
                        reciprocal_rank = 1.0 / (1.0 + i)
                        break
            if np.random.random() < self.p:
                return MetricValue(privatized=reciprocal_rank, non_privatized=reciprocal_rank)
            else:
                while True:
                    sampled_value = random.sample(self.possible_outputs, 1)[0]
                    if sampled_value != reciprocal_rank:
                        return MetricValue(privatized=sampled_value, non_privatized=reciprocal_rank)

    def get_random_privatized_dcg(self, cutoff: int):
        """
        To get privatized DCG metrics
        """
        initial_ranking = [0*x for x in range(cutoff)]
        n_relevant = random.sample([r+1 for r in range(cutoff)], 1)[0]
        for i in range(n_relevant):
            initial_ranking[i] = 1

        random_ranking = random.sample(initial_ranking, k=len(initial_ranking))
        p_dcg = 0.0
        for i in range(0, cutoff):
            if self.debias:
                p_dcg += ((2 ** random_ranking[i] - 1) / (np.log2(i + 2) * (1.0 / (i + 1))))
            else:
                p_dcg += (2 ** random_ranking[i] - 1) / np.log2(i + 2)

        return p_dcg


class ExpectedMetric:
    """
    This class is used to calculate the expectation of MaxRR over a ranked list given its relevance scores
    and a CCM click model. Essentially, it induces an offline metric.
    """

    def __init__(self, click_model: CcmClickModel, cutoff: int = 10):
        """
        :param click_model: An instance of CCM model; the expectation is calculated w.r.t. this model
        :param cutoff: The cut-off level - documents below it are not used to calculate the MaxRR
        """
        self.click_model = click_model
        self.cutoff = cutoff

    # def eval_ranking(self, ranking: np.ndarray) -> float:
    #     """
    #     Maps a ranked list  into the expectation of MaxRR
    #     :param ranking: a np.ndarray vector of document relevance labels
    #     :return: the expected MaxRR w.r.t. click_model
    #
    #     As an example, consider a model of a user who always clicks on a highly relevant result and immediately stops;
    #     under such a model MaxRR would be the reciprocal rank of the first highly relevant document:
    #     >>> model = CcmClickModel(click_relevance={0: 0.0, 1: 0.0, 2: 1.0},
    #     ...                       stop_relevance={0: 0.0, 1: 0.0, 2: 1.0}, name="Model", depth=10)
    #     >>> metric = ExpectedMaxRR(model)
    #     >>> doc_relevances = np.array([1.0, 0.0, 0.0, 2.0, 0.0, 1.0])
    #     >>> metric.eval_ranking(doc_relevances)
    #     0.25
    #     >>> doc_relevances = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    #     >>> metric.eval_ranking(doc_relevances)
    #     0.0
    #     """
    #     click_relevance = self.click_model.click_relevance
    #
    #     metric = 0.0
    #     p_not_clicked_yet = 1.0
    #     for i in range(min(self.cutoff, ranking.shape[0])):
    #         r = ranking[i]
    #         p_click = click_relevance[r]
    #
    #         p_first_click = p_click * p_not_clicked_yet
    #         p_not_clicked_yet *= 1.0 - p_click
    #         metric += p_first_click / (i + 1.0)
    #     return metric

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

        return ndcg, idcg

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

    def eval_ranker_ndcg(self, ranker: torch.nn.Module, data: DataSplit) -> float:
        """
        Evaluates a ranker over all queries in a dataset provided in `data`. To do that, applies the ranker for a
        queries, sort the documents according to ranking scores, gets the quality of the score.
        :param ranker: A ranker to be evaluated.
        :param data: A dataset to use in the evaluation.
        :return: Averaged over all queries value of the ranking quality, assessed by ExpactedMaxRR.
        """

        average_metric = 0.0
        count = 0
        for slice in data.values():
            features = torch.from_numpy(slice.features).float()
            scores_slice = ranker.forward(features).data.numpy()[:, 0]
            ranking_order = np.argsort(scores_slice)[::-1]
            relevances = slice.relevance_labels[ranking_order]
            ndcg, idcg = self.eval_ranking_ndcg(relevances)
            if idcg == 0:
                continue
            average_metric += ndcg
            count += 1
        return average_metric / count

    def eval_ranker_mrr(self, ranker: torch.nn.Module, data: DataSplit) -> float:
        """
        Evaluates a ranker over all queries in a dataset provided in `data`. To do that, applies the ranker for a
        queries, sort the documents according to ranking scores, gets the quality of the score.
        :param ranker: A ranker to be evaluated.
        :param data: A dataset to use in the evaluation.
        :return: Averaged over all queries value of the ranking quality, assessed by ExpactedMaxRR.
        """

        average_metric = 0.0
        count = 0
        for slice in data.values():
            features = torch.from_numpy(slice.features).float()
            scores_slice = ranker.forward(features).data.numpy()[:, 0]
            ranking_order = np.argsort(scores_slice)[::-1]
            relevances = slice.relevance_labels[ranking_order]
            average_metric += self.eval_ranking_mrr(relevances)
            count += 1
        return average_metric / count
