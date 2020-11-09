# Federated Online Learning to Rank with Evolution Strategies: A Reproducibility Study

This repo contains the code used to run experiments for the paper 'Federated Online Learning to Rank with Evolution Strategies: A Reproducibility Study', submmited to ECIR 2021.

Our work intends to reproduce the orginial work: Federated online learning to rank with evolution strategies (FOLTR-ES). The original repository can be found at: https://github.com/facebookresearch/foltr-es

Here are few steps for reproduce our experiments.

## Setup python environment
Create a conda environment for running this code using the code below.

````
conda create --name federated python=3.6
source activate federated
# assuming you want to checkout the repo in the current directory
git clone https://github.com/ielab/foltr.git && cd foltr
pip install -r requirements.txt 
````

## Download datasets
In the paper, four datasets are used, MQ2007/2008, MSLR-WEB10K and Yahoo!Webscope.
- MQ2007/2008 can be downloaded from the Microsoft Research [website](https://www.microsoft.com/en-us/research/project/letor-learning-rank-information-retrieval/). 
- MSLR-WEB10K can be downloaded from the Microsoft Research [website](https://www.microsoft.com/en-us/research/project/mslr/).  
- Yahoo!Webscope can be downloaded from [Yahoo Webscope program](https://webscope.sandbox.yahoo.com/catalog.php?datatype=c) and we used Set 1 of C14B dataset in our paper.

After downloading data files, they have to be unpacked within the `./code-and-results/data` folder.

## Reproducing results
The main functions for our methods are stored at `./code-and-results/foltr` folder. The main fuctions for the original methods by [FOLTR-ES](https://github.com/facebookresearch/foltr-es) are stored at `./code-and-results/foltr-original` folder. 

To reproduce our experiments reuslt, set up corresponding parameters and run file `./code-and-results/foltr_reproduce_run.py`
```
python foltr_reproduce_run.py
```

## OLTR baselines
We use Pairwise Differentiable Gradient Descent (PDGD) as the baselines. This method is proposed by Oosterhuis and de Rijke at CIKM 2018 [https://dl.acm.org/doi/pdf/10.1145/3269206.3271686](https://dl.acm.org/doi/pdf/10.1145/3269206.3271686).

Our implementation of PDGD is in another github repo: [https://github.com/ArvinZhuang/OLTR](https://github.com/ArvinZhuang/OLTR).
The run script that can reproduce the PDGD results presented in our paper is: `experiments/run_PDGD_batch_update.py`

## Results and Figures
Our experiments result files and code to reproduce the plots in the paper are in the folder: `./code-and-results/results/`.  

- Result files for experiments in each research question can be found at `./code-and-results/results/foltr-results`. Result files for OLTR baselines can be found at `./code-and-results/results/PDGD`.
- Figures for RQ1 are in `./code-and-results/results/figures/RQ1` folder. Figures for RQ2 are in `./code-and-results/results/figures/RQ2` folder. Figures for RQ3 and RQ4 are in `./code-and-results/results/figures/RQ3-4` folder.

### Figures for RQ1: performance of FOLTR-ES across datasets (averaged on all dataset splits)
(a) Mean batch MaxRR for MQ2007 (averaged on all dataset splits)
![image](https://github.com/ielab/foltr/blob/master/code-and-results/results/figures/RQ1/mq2007_foltr_c2000_ps.png)
(b) Mean batch MaxRR for MQ2008 (averaged on all dataset splits)
![image](https://github.com/ielab/foltr/blob/master/code-and-results/results/figures/RQ1/mq2008_foltr_c2000_ps.png)
(c) Mean batch MaxRR for MSLR10k(averaged on all dataset splits)
![image](https://github.com/ielab/foltr/blob/master/code-and-results/results/figures/RQ1/mslr10k_foltr_c2000_ps.png)
(d) Mean batch MaxRR for Yahoo
![image](https://github.com/ielab/foltr/blob/master/code-and-results/results/figures/RQ1/yahoo_foltr_c2000_ps.png)

### Figures for RQ2: performance of FOLTR-ES with respect to number of clients
(a) Mean batch MaxRR for MQ2007 (averaged on all dataset splits)
![image](https://github.com/ielab/foltr/blob/master/code-and-results/results/figures/RQ2/mq2007_foltr_client_both_p0.9.png)
(b) Mean batch MaxRR for MQ2008 (averaged on all dataset splits)
![image](https://github.com/ielab/foltr/blob/master/code-and-results/results/figures/RQ2/mq2008_foltr_client_both_p0.9.png)
(c) Mean batch MaxRR for MSLR10k (averaged on all dataset splits)
![image](https://github.com/ielab/foltr/blob/master/code-and-results/results/figures/RQ2/mslr10k_foltr_client_both_p0.9.png)
(d) Mean batch MaxRR for Yahoo
![image](https://github.com/ielab/foltr/blob/master/code-and-results/results/figures/RQ2/yahoo_foltr_client_both_p0.9.png)

### Figures for RQ3: performance of FOLTR-ES and PDGD across datasets
(a) Mean batch MaxRR for MQ2007 (averaged on all dataset splits)
![image](https://github.com/ielab/foltr/blob/master/code-and-results/results/figures/RQ3-4/mq2007_foltr_PDGD_mrr_c2000_p1.0.png)
(b) Mean batch MaxRR for MQ2008 (averaged on all dataset splits)

(c) Mean batch MaxRR for MSLR10k (averaged on all dataset splits)
![image](https://github.com/ielab/foltr/blob/master/code-and-results/results/figures/RQ3-4/mslr10k_foltr_PDGD_mrr_c2000_p1.0.png)
(d) Mean batch MaxRR for Yahoo
![image](https://github.com/ielab/foltr/blob/master/code-and-results/results/figures/RQ3-4/yahoo_foltr_PDGD_mrr_c2000_p1.0.png)

### Figures for RQ4: performance of FOLTR-ES in terms of online nDCG@10
(a) Mean batch nDCG@10 for MQ2007 (averaged on all dataset splits)
![image](https://github.com/ielab/foltr/blob/master/code-and-results/results/figures/RQ3-4/mq2007_foltr_DCG_both_c2000_ps.png)
(b) Mean batch nDCG@10 for MQ2008 (averaged on all dataset splits)

(c) Mean batch nDCG@10 for MSLR10k (averaged on all dataset splits)
![image](https://github.com/ielab/foltr/blob/master/code-and-results/results/figures/RQ3-4/mslr10k_foltr_DCG_both_c2000_ps.png)
(d) Mean batch nDCG@10 for Yahoo
![image](https://github.com/ielab/foltr/blob/master/code-and-results/results/figures/RQ3-4/yahoo_foltr_DCG_both_c2000_ps.png)

### Figures for RQ4: performance of FOLTR-ES and PDGD in terms of offine nDCG@10
(a) Mean batch nDCG@10 for MQ2007 (averaged on all dataset splits)
![image](https://github.com/ielab/foltr/blob/master/code-and-results/results/figures/RQ3-4/mq2007_foltr_PDGD_offline_ndcg_c2000_p1.0.png)
(b) Mean batch nDCG@10 for MQ2008 (averaged on all dataset splits)

(c) Mean batch nDCG@10 for MSLR10k (averaged on all dataset splits)
![image](https://github.com/ielab/foltr/blob/master/code-and-results/results/figures/RQ3-4/mslr10k_foltr_PDGD_offline_ndcg_c2000_p1.0.png)
(d) Mean batch nDCG@10 for Yahoo
![image](https://github.com/ielab/foltr/blob/master/code-and-results/results/figures/RQ3-4/yahoo_foltr_PDGD_offline_ndcg_c2000_p1.0.png)
