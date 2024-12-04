# PFCA: Efficient Path-Filtering with Causal Analysis Approach for Healthcare Risk Prediction
codebase for PFCA

![version](https://img.shields.io/badge/version-v3.5-green)
![python](https://img.shields.io/badge/python-3.9.19-blue)
![pytorch](https://img.shields.io/badge/pytorch-2.0.1-brightgreen)

This repository contains our implementation of **PFCA**.

### Data Format
The patient EHR data, **features** is formatted as 
```
[sample_num * tensor(1, feature_num)]
```
The **label** is formatted as
```
[sample_num * tensor(1, target_num)]
```
The graph data like **rel_index** is formatted as
```
tensor(sample_num, visit_num, feature_num, target_num, path_num, K, rel_num)
```

### Benchmark Datasets

* [MIMIC-III](https://physionet.org/content/mimiciii/1.4/)
* [MIMIC-IV](https://physionet.org/content/mimiciv/3.0/)

### External Medical Knowledge Graph

* [UMLS](https://www.nlm.nih.gov/research/umls/index.html)

### Data Preprocess
Please use datapreprocess_iii_kg.py and datapreprocess_iv_kg.py to process knowledge graph. To generate the adjancent matrix of the knowledge graph, just run the corresponding python file:
1. `python datapreprocess_iii_kg.py` or `python datapreprocess_iv_kg.py`

After processing the knowledge graph, please use datapreprocess_iii.py and datapreprocess_iv.py to process MIMIC-III and MIMIC-IV datasets:
`datapreprocess_iii.py --task task_name`

For example:

`datapreprocess_iii.py --task diagnosis_prediction`

### Baseline Models

| Model                | Code                                                                                              | Reference                                                                        |
|----------------------|---------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------|
| LSTM                 | model/lstm.py                                                                                     | [LSTM]()                                                                         |
| Dipole               | model/Dipole.py                                                                                   | [Dipole](https://arxiv.org/pdf/1706.05764)                                       |
| Retain               | model/retain.py                                                                                   | [Retain](https://arxiv.org/pdf/1608.05745)                                       |
| HAP                  | model/HAP.py                                                                                      | [HAP](https://dl.acm.org/doi/10.1145/3394486.3403067)                            |
| MedPath              | model/medpath.py                                                                                  | [MedPath](https://dl.acm.org/doi/pdf/10.1145/3442381.3449860)                    |
| GraphCare            | model/graphcare.py                                                                                | [GraphCare](https://arxiv.org/pdf/2305.12788)                                    |
| HAR                  | model/stageaware.py                                                                               | [HAR](https://ieeexplore.ieee.org/document/10236511)                             |

#### Running Scripts

The running scripts are available in run.sh. 

##### Best AUC and F1 for CAPF

The running scripts are available in "Section 1: Best AUC and F1 for CAPF" of run.sh.
Be reminded to save the checkpoints in the format of ".ckpt" after the model is trained.

#####  Best AUC and F1 for Baseline Methods

The running scripts are available in "Section 2: Best AUC and F1 for Baseline Methods" of run.sh.

##### Interpretation

The running scripts are available in "Section 3: Interpretation" of run.sh.
The significant paths and the corresponding attention values will be printed out during the evaluate stage.

#### Requirement

* python>=3.9.19
* PyTorch>=2.0.1
