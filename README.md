This repository is based on the [DKGR project](@article{zhang2021learning,
  title={Learning to Walk with Dual Agents for Knowledge Graph Reasoning},
  author={Zhang, Denghui and Yuan, Zixuan and Liu, Hao and Lin, Xiaodong and Xiong, Hui},
  journal={arXiv preprint arXiv:2112.12876},
  year={2021}
}), which is described in detail below. In this fork, I have combined two powerful AI paradigms, reinforcement learning (RL) and transformers, by implementing a transformer as the policy network for RL agents.

There are only a handful of articles that integrate transformers with reinforcement learning (RL), including the work by [Chen.L et al.](https://arxiv.org/abs/2106.01345), which introduced the "Transformer Decision" framework. This framework presents reinforcement learning (RL) in the context of sequential conditional modeling. Transformer Decision has achieved excellent results in offline model-free reinforcement learning tasks on Atari, OpenAI Gym, and Door-to-Key environments.

A paper by [Chen.C. et al](https://arxiv.org/abs/2202.09481) introduces a model-based reinforcement learning agent called "TransDreamer", whose policy network is transformer-based. This model demonstrates better performance than previous approaches on 2D visual tasks and 3D first-person visual tasks. Additionally, [Upadhyay et al.](https://arxiv.org/abs/1912.03918) used transformers as the policy network in Q-Learning and evaluated its performance in video game environments.

# Learning to Walk with Dual Agents for Knowledge Graph Reasoning

Pytorch Implementation for AAAI' 2022 paper: Learning to Walk with Dual Agents for Knowledge Graph Reasoning

## Framework Overview
<!-- <<<<<<< HEAD -->
This paper proposed a dual-agent based reinforcement learning approach to tackle the KG reasoning problem. Existing RL-based learning to walk methods rely solely on one entity-level agent to explore large KGs, which works well on finding short reasoning paths, but usually succumb to longer patterns. Hence, we propose to divide large KGs into semantic clusters first, then utilize a cluster-level agent (named Giant) to assist entity-level agent (named Dwarf) and co-explore KG reasoning.  To fulfill the purpose, we design a Collaborative Policy Network and Mutual Reinforcement Reward system to train two agents synchronously. 
<!-- ======= -->
<!-- This paper proposed a dual-agent based reinforcement learning approach (CURL) to tackle the KG reasoning problem. Existing RL-based learning to walk methods rely solely on one entity-level agent to explore large KGs, which works well on finding short reasoning paths, but usually succumb to longer patterns. Hence, we propose to divide large KGs into semantic clusters first, then utilize a cluster-level agent (named Giant) to assist entity-level agent (named Dwarf) and co-explore KG reasoning.  To fulfill the purpose, we design a Collaborative Policy Network and Mutual Reinforcement Reward system to train two agents synchronously.  -->
<!-- >>>>>>> 4389ec8d8b09a06bf1a174a5a415e994a4c12154 -->

<p align="center">
<img width="800" height="200.5" src="./figs/framework.png" align=center>
</p>


## Requirements
<!-- To install the various python dependencies (including pytorch) -->
```
pip install -r requirements.txt
```
- python 3.8.5
- scipy 1.6.2
- tqdm 4.62.3
- torch 1.7.1
- numpy 1.19.2


## Training & Testing
The hyperparam configs for each experiments are included in the [configs](https://github.com/RutgersDM/DKGR/tree/master/configs) directory. To start a particular experiment, just do
```
sh run.sh configs/${dataset}.sh
```
where the `${dataset}.sh` is the name of the config file. For example, 
```
sh run.sh configs/nell.sh
```

## Output
The code outputs the evaluation of CURL on the datasets provided. The metrics used for evaluation are Hits@{1,3,5,10,20}, MRR, and MAP.  Along with this, the code also outputs the answers CURL reached in a file.

<!-- ## Citation -->
<!-- If you use our code, please cite the paper
```
@InProceedings{curl2022,
  author    = {Denghui Zhang, Zixuan Yuan, Hao Liu, Xiaodong Lin, Hui Xiong},
  title     = {Learning to Walk with Dual Agents for Knowledge Graph Reasoning},
  booktitle = {Proceedings of the Thirty-Sixth AAAI Conference on Artificial Intelligence (AAAI 2022)},
  month     = {September},
  year      = {2022},
  address   = {Copenhagen, Denmark},
  publisher = {ACL}
}
``` -->
## Citation
If you find our paper useful or use our code, please kindly cite the paper.
```
@article{zhang2021learning,
  title={Learning to Walk with Dual Agents for Knowledge Graph Reasoning},
  author={Zhang, Denghui and Yuan, Zixuan and Liu, Hao and Lin, Xiaodong and Xiong, Hui},
  journal={arXiv preprint arXiv:2112.12876},
  year={2021}
}
```

## Acknowledgement
* [MINERVA implementation](https://github.com/shehzaadzd/MINERVA)
