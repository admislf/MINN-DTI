# Effective drug-target interaction prediction with mutual interaction neural network

## MINN-DTI
* Source code for the paper "Effective drug-target interaction prediction with mutual interaction neural network".

* MINN-DTI is a model for drug-target interaction (DTI) prediction. MINN-DTI combines an interacting-transformer module (called Interformer) with an improved Communicative Message Passing Neural Network (CMPNN) (called Inter-CMPNN) to better capture the two-way impact between drugs and targets, which are represented by molecular graph and distance map respectively.

![MINN-DTI](image/Fig.1.jpg)
* The code was built based on [DrugVQA](https://github.com/prokia/drugVQA), [CMPNN](https://github.com/SY575/CMPNN) and [transformerCPI](https://github.com/lifanchen-simm/transformerCPI). Thanks a lot for their code sharing!

## Dataset
All data used in this paper are publicly available and consistent with that used by DrugVQA , which can be accessed here : [DrugVQA](https://github.com/prokia/drugVQA).

## Environment
* base dependencies:
```
  - dgl
  - dgllife
  - numpy
  - pandas
  - python>=3.7
  - pytorch>=1.7.1
  - rdkit
```
* We also provide an environment file for Anaconda users. You can init your environment by ```conda env create -f environment.yaml```.
* Need download the chemprop package from [CMPNN](https://github.com/SY575/CMPNN) and put it in model/ directory.

## Usage
* All default arguments are provided in the [model/data.py](./model/data.py) for training and [model/dataTest.py](./model/dataTest.py) for test.
* Run [model/main.py](./model/main.py) to train a model and [model/mainTest.py](./model/mainTest.py) to test.
