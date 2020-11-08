# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Dict, Any, NamedTuple, List
import torch
import numpy as np
from tqdm import tqdm

from foltr.client.client import RankingClient
from foltr.client import rankers
from foltr.data.datasets import DataSplit

# used for RQ1/RQ2/RQ3
# TrainResult = NamedTuple("TrainResult", [
#                         ('batch_metrics', List[float]),
#                         ('expected_metrics', List[float]),
#                         ('ranker', torch.nn.Module)])

# used for RQ4
TrainResult = NamedTuple("TrainResult", [
                        ('batch_metrics', List[float]),
                        ('expected_metrics', List[float]),
                        ('ranker', torch.nn.Module),
                        ('ndcg_server', float),
                        ('mrr_server', float),
                        ('ndcg_clients', float),
                        ('mrr_clients',float)])

def train_uniform(params: Dict[str, Any], traindata: DataSplit, testdata: DataSplit ,letordataset, message) -> TrainResult:
    """
    :param traindata: dataset used for training server ranker
    :param testdata: dataset used for testing true performance of server ranker - using true relevance label
    :param letordataset: dataset used for calculating NDCG - same dataset as 'testdata'
    """
    seed = params['seed']
    np.random.seed(seed)
    torch.manual_seed(seed)

    n_clients = params['n_clients']
    sessions_per_feedback = params['sessions_per_feedback']
    click_model = params['click_model']
    online_metric = params['online_metric']
    noise_std = params['noise_std']
    ranker = params['ranker_generator']()
    optimizer = torch.optim.Adam(ranker.parameters(), lr=params['lr'])
    antithetic = params['antithetic']

    clients = [RankingClient(traindata, ranker, seed + client_id, click_model, online_metric, antithetic, noise_std)
               for client_id in range(n_clients)]

    n_iterations = params['sessions_budget'] // n_clients // sessions_per_feedback # 250

    batch_rewards = []
    expected_rewards = []
    ndcg_rewards = []
    mrr_rewards = []
    ndcg_clients = []
    mrr_clients = []

    for i in tqdm(range(n_iterations), desc=message):
        i += 1
        ranker.zero_grad()  # zero_grad(): pytorch function - initialize model's gradient to 0
        feedback = []
        ndcg_client = 0.0
        mrr_client = 0.0
        for client in clients:
            f, n, m = client.get_click_feedback(sessions_per_feedback)
            feedback.append(f)
            ndcg_client += n
            mrr_client += m
        # Those non-privatized values are solely used for plotting the performance curves
        if not antithetic:
            batch_reward = np.mean([f.metric.non_privatized for f in feedback])
        else:
            batch_reward = np.mean(
                [0.5 * (f.metric.non_privatized + f.antithetic_metric.non_privatized) for f in feedback])
        batch_rewards.append(batch_reward)

        rankers.update_gradients(ranker, noise_std, feedback, inverse=True) # train the server ranker
        optimizer.step()
        for client in clients:
            client.update_model(ranker)

        ndcg_clients.append(ndcg_client)
        mrr_clients.append(mrr_client)

        if (i * n_clients) % 1000 == 0 or i == 1:
            ndcg, mrr = get_ndcg(ranker, testdata, letordataset) # get evaluation metrics for the server ranker
            ndcg_rewards.append(ndcg)
            mrr_rewards.append(mrr)

    return TrainResult(batch_metrics=batch_rewards, ranker=ranker, expected_metrics=expected_rewards,
                       ndcg_server=ndcg_rewards, mrr_server=mrr_rewards,
                       ndcg_clients=ndcg_clients, mrr_clients=mrr_clients)

def get_ndcg(ranker, dataset, dataset_ndcg):
    query_result_list = {}

    # for query in dataset.get_all_querys():
    #     docid_list = np.array(dataset.get_candidate_docids_by_query(query))
    #     docid_list = docid_list.reshape((len(docid_list), 1))
    #     feature_matrix = dataset.get_all_features_by_query(query)
    #     with torch.no_grad():
    #         score_list = ranker.forward(feature_matrix).numpy()[:, 0]
    for query, slice in dataset.items():
        features = torch.from_numpy(slice.features).float()
        with torch.no_grad():
            scores_slice = ranker.forward(features).data.numpy()[:, 0]
        try:
            query1 = query.split(':')[1]
            docid_list = np.array(dataset_ndcg.get_candidate_docids_by_query(query1))
        except Exception as e:
            raise e
        docid_list = docid_list.reshape((len(docid_list), 1))
        docid_score_list = np.column_stack((docid_list, scores_slice))
        docid_score_list = np.flip(docid_score_list[docid_score_list[:, 1].argsort()], 0)

        query_result_list[query1] = docid_score_list[:, 0]

    ndcg, mrr = average_ndcg_at_k(dataset_ndcg, query_result_list, 10)
    return ndcg, mrr


#ndcg@k & MRR
def average_ndcg_at_k(dataset, query_result_list, k):
    ndcg = 0.0
    rr = 0
    num_query = 0
    for query in dataset.get_all_querys():
        if len(dataset.get_relevance_docids_by_query(query)) == 0:  # for this query, ranking list is None
            # num_query += 1
            continue
        else:
            pos_docid_set = set(dataset.get_relevance_docids_by_query(query))
        dcg = 0.0
        got_rr = False
        for i in range(0, min(k, len(query_result_list[query]))):
            docid = query_result_list[query][i]
            relevance = dataset.get_relevance_label_by_query_and_docid(query, docid)
            dcg += ((2 ** relevance - 1) / np.log2(i + 2))
            if relevance in {1, 2, 3, 4} and got_rr == False: # relevance in {0,1,2}
                rr += 1/(i+1)
                got_rr = True

        rel_set = []
        for docid in pos_docid_set:
            rel_set.append(dataset.get_relevance_label_by_query_and_docid(query, docid))
        rel_set = sorted(rel_set, reverse=True)
        n = len(pos_docid_set) if len(pos_docid_set) < k else k

        idcg = 0
        for i in range(n):
            idcg += ((2 ** rel_set[i] - 1) / np.log2(i + 2))

        if idcg != 0:
            ndcg += (dcg / idcg)

        num_query += 1
    return ndcg / float(num_query), rr / float(num_query)
