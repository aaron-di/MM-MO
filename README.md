<h1 align="center"> It’s Morphing Time: Unleashing the Potential of Multiple LLMs via Multi-Objective Optimization </h1>

This repository includes the code for the paper “_**It’s Morphing Time: Unleashing the Potential of Multiple LLMs via Multi-Objective Optimization**_”, which has been accepted at **IEEE Transactions on Evolutionary Computation**. The paper is available at: https://ieeexplore.ieee.org/abstract/document/11177252

## 💥 News 💥

- 🎉🎉🎉 **[Sep 17, 2025]** Our paper is accepted at IEEE Transactions on Evolutionary Computation! The camera ready version is coming soon.

## Overview

<img src="./assets/MM-MO.png">

## 🚀 Getting Started

### Install dependencies

Please install the required dependencies of the following projects:

1. OpenCompass
   https://github.com/open-compass/opencompass

2. MergeKit
   https://github.com/arcee-ai/mergekit

3. BoTorch
   https://github.com/pytorch/botorch

### File Structure
```
MM-MO/
├── 📂 config/         # Stores all generated model merge configurations
├── 📂 merge_info/     # Stores evaluation results of all merged models
├── 📂 merged/         # Temporarily stores merged models; automatically cleaned up after evaluation to avoid excessive disk usage
├── 📂 save_logs/      # Stores all log files
├── 📂 utils/          # Stores all related utility tools

├── 📄 mm_mo.py                       # 🚀 MM-MO main program; all core logic is centralized here for easier debugging and modification
├── 📄 evaluate_model_fitness.py      # 🧪 Evaluates sparsity-related metrics of merged models
├── 📄 evaluate_model_opencompass.py  # 🏆 Evaluates merged model performance across different tasks (via OpenCompass)
└── 📄 merge_local.py                 # 🌋 Merges models and saves them locally (via MergeKit)
```


## Citation

If you find our work helpful, please cite the following BibTeX entry:

```
@ARTICLE{11177252,
  author={Li, Bingdong and Di, Zixiang and Yang, Yanting and Qian, Hong and Yang, Peng and Hao, Hao and Tang, Ke and Zhou, Aimin},
  journal={IEEE Transactions on Evolutionary Computation}, 
  title={It’s Morphing Time: Unleashing the Potential of Multiple LLMs via Multi-Objective Optimization}, 
  year={2025},
  volume={},
  number={},
  pages={1-1},
  keywords={Merging;Optimization;Adaptation models;Computational modeling;Data models;Training data;Interference;Overfitting;Measurement;Training;Large language model;model merging;multi-objective optimization},
  doi={10.1109/TEVC.2025.3613937}}
```