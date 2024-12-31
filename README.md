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
The graph data like **path_index**, which represents paths extracted from the personalized knowledge graphs (PKGs), is formatted as
```
tensor(sample_num, visit_num, feature_num, path_num)
```

### Benchmark Datasets

* [MIMIC-III](https://physionet.org/content/mimiciii/1.4/)
* [MIMIC-IV](https://physionet.org/content/mimiciv/3.0/)

### External Medical Knowledge Graph

* [UMLS](https://www.nlm.nih.gov/research/umls/index.html)

### Data Preprocess
```
Data/
├── mimic-iii/
│   ├── ...
├── mimic-iv/
│   ├── ...
├── UMLS/
│   ├── ...
├── datapreprocess_iii_kg.py
├── datapreprocess_iii.py
├── datapreprocess_iv_kg.py
├── datapreprocess_iv.py
```
As is shown above, please store the downloaded MIMIC-III and MIMIC-IV data in the "mimic-iii" and "mimic-iv" directories within the "Data" folder. Similarly, place the UMLS knowledge graph in the "UMLS" directory inside the "Data" folder. 

#### Processing Knowledge Graph
To process the knowledge graph, use `datapreprocess_iii_kg.py` and `datapreprocess_iv_kg.py`. Run the corresponding Python file to generate the adjacency matrix of the knowledge graph based on MIMIC-III and MIMIC-IV datasets:
1. For MIMIC-III adjacency matrix:
`python datapreprocess_iii_kg.py`
2. For MIMIC-IV adjacency matrix:
`python datapreprocess_iii_kg.py`

### Processing MIMIC Datasets
After processing the knowledge graph, use datapreprocess_iii.py and datapreprocess_iv.py to process MIMIC-III and MIMIC-IV datasets. Specify the task using the `--task` parameter. Available tasks are as follows:
1. `diagnosis_prediction`
2. `medication_recommendation`
3. `readmission_prediction`

For example, to process the MIMIC-III dataset for diagnosis prediction:

`python datapreprocess_iii.py --task diagnosis_prediction`

### Baseline Models

| Model                | Code                                                                                              | Reference                                                                        |
|----------------------|---------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------|
| LSTM                 | model/lstm.py                                                                                     | [LSTM](https://ieeexplore.ieee.org/abstract/document/6795963)                                                                         |
| Dipole               | model/Dipole.py                                                                                   | [Dipole](https://arxiv.org/pdf/1706.05764)                                       |
| Retain               | model/retain.py                                                                                   | [Retain](https://arxiv.org/pdf/1608.05745)                                       |
| HAP                  | model/HAP.py                                                                                      | [HAP](https://dl.acm.org/doi/10.1145/3394486.3403067)                            |
| MedPath              | model/medpath.py                                                                                  | [MedPath](https://dl.acm.org/doi/pdf/10.1145/3442381.3449860)                    |
| GraphCare            | model/graphcare.py                                                                                | [GraphCare](https://arxiv.org/pdf/2305.12788)                                    |
| HAR                  | model/stageaware.py                                                                               | [HAR](https://ieeexplore.ieee.org/document/10236511)                             |

#### Running Scripts

The running scripts are available in run.sh. 

##### Best AUC and F1 for PFCA

The running scripts are available in "Section 1: Best AUC and F1 for PFCA" of run.sh.
Be reminded to save the checkpoints in the format of ".ckpt" after the model is trained.

##### Hyperparameter Study for PFCA

The running scripts are available in "Section 2: Hyperparameter Study for PFCA" of run.sh.

##### Ablation Study for PFCA

The running scripts are available in "Section 3: Ablation Study for PFCA" of run.sh.

#####  Best AUC and F1 for Baseline Methods

The running scripts are available in "Section 4: Best AUC and F1 for Baseline Methods" of run.sh.

##### Interpretation

The running scripts are available in "Section 5: Interpretation" of run.sh.
The significant paths and the corresponding attention values will be printed out during the evaluate stage.

#### Requirement

* python>=3.9.19
* PyTorch>=2.0.1
