# Federated Online Learning to Rank with Evolution Strategies: A Reproducibility Study

This repo contains the code used to run experiments for the paper 'Federated Online Learning to Rank with Evolution Strategies: A Reproducibility Study', submmited to ECIR 2021.

Our work intends to reproduce the orginial work: Reproducing federated online learning to rank with evolution strategies (FOLTR-ES). The original repository can be found at: https://github.com/facebookresearch/foltr-es

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

MQ2007/2008 can be downloaded from the Microsoft Research [website](https://www.microsoft.com/en-us/research/project/letor-learning-rank-information-retrieval/).
MSLR-WEB10K can be downloaded from the Microsoft Research [website](https://www.microsoft.com/en-us/research/project/mslr/).
Yahoo!Webscope can be downloaded from [Yahoo Webscope program](https://webscope.sandbox.yahoo.com/catalog.php?datatype=c) and we used the Set 1 of C14B in our paper.

After downloading data files, they have to be unpacked within the `./code-and-results/data` folder.

## Reproducing results
The main functions for our methods are stored at `./code-and-results/foltr` folder. The main fuctions for the original methods by [FOLTR-ES](https://github.com/facebookresearch/foltr-es) are stored at `./code-and-results/foltr-original` folder. 

To reproduce our experiments reuslt, set up corresponding parameters and run file `./code-and-results/foltr_reproduce_run.py`
```
python foltr_reproduce_run.py
```

Our exeperiements result files and code to reproduce the plots in the paper are in the folder: `./code-and-results/results/`

## OLTR baselines
We use Pairwise Differentiable Gradient Descent (PDGD) as the baselines. This method is proposed by Oosterhuis and de Rijke at CIKM 2018 [https://dl.acm.org/doi/pdf/10.1145/3269206.3271686](https://dl.acm.org/doi/pdf/10.1145/3269206.3271686).


Our implementation of PDGD is in another github repo: [https://github.com/ArvinZhuang/OLTR](https://github.com/ArvinZhuang/OLTR).
The run script that can reproduce the PDGD results presented in our paper is: `experiments/run_PDGD_batch_update.py`
