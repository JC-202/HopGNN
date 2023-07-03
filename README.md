# HopGNN

A **PyTorch** implementation of HopGNN "**From Node Interaction to Hop Interaction: New Effective and Scalable Graph Learning Paradigm**". (CVPR 2023)

(https://arxiv.org/abs/2211.11761)

## Abstract
<p align="justify">
Existing Graph Neural Networks (GNNs) follow the message-passing mechanism that conducts information interaction among nodes iteratively. While considerable progress has been made, such node interaction paradigms still have the following limitation. 
First, the scalability limitation precludes the broad application of GNNs in large-scale industrial settings since the node interaction among rapidly expanding neighbors incurs high computation and memory costs. 
Second, the over-smoothing problem restricts the discrimination ability of nodes, i.e., node representations of different classes will converge to indistinguishable after repeated node interactions. In this work, we propose a novel hop interaction paradigm to address these limitations simultaneously. 
The core idea is to convert the interaction target among nodes to pre-processed multi-hop features inside each node.
We design a simple yet effective HopGNN framework that can easily utilize existing GNNs to achieve hop interaction. Furthermore, we propose a multi-task learning strategy with a self-supervised learning objective to enhance HopGNN. We conduct extensive experiments on 12 benchmark datasets in a wide range of domains, scales, and smoothness of graphs. Experimental results show that our methods achieve superior performance while maintaining high scalability and efficiency.
</p>

## Dependencies
- python 3.7.3
- pytorch 1.10.1
- dgl 0.6.0
- ogb 1.2.3
- torch-geometric 2.0.3

## Code Architecture
    |── datasets                # datasets and load scripts
    |── utils                   # Common useful modules(transform, loss function)
    |── models                  # models of hopgnn
    |── scripts                 # experiments for each dataset
    └── train                   # train scripts
    

## Usage 

Try node classification
```
bash scripts/citation.sh
bash scripts/heterophily.sh
bash scripts/large.sh 
```

## Citation
```
@inproceedings{chen2023node,
  title={From Node Interaction to Hop Interaction: New Effective and Scalable Graph Learning Paradigm},
  author={Chen, Jie and Li, Zilong and Zhu, Yin and Zhang, Junping and Pu, Jian},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={7876--7885},
  year={2023}
}
```