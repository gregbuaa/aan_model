#  Discriminative Feature Adaptation via CMD for UDA.

In this paper, we present Aligned Adaptation Networks (AAN) to match both the marginal and conditional distributions across domains for UDA. 


Two datasets *Amazon-Review* and *Amazon-Text* are provided in **dataset/** with *.json*.


Maximum mean discrepancy (MMD) and Conditional MMD (CMMD) are implemented in **model/criterion.py**

AANs and AAN-As are implmented in **model/models.py** with three different types of extractors $T(\cdot;\theta_T)$: MLP, TextCNN and BertGRU.

Model trainings are represented in `aan_mlp.ipynb`, `aan_cnn.ipynb` and `aan_bert.ipynb`.


## Installation
1. Clone this repsitory.
```sh
git clone https://github.com/gregbuaa/aan_model.git
cd aan_model
```

2. Install the dependencies. The code runs with PyTorch-1.2.0 in our experiments. 


```sh
pip install -r requirements.txt 
```

3. Play with the Jupyter notebooks.
```sh
jupyter notebook
```

## Reproducing our results
Run all the notebooks to reproduce the experiments on
[Amazon_Feature](aan_mlp.ipynb) with MLP extractor, [Amazon_Text_CNN](aan_cnn.ipynb) with TextCNN extractor and [Amazon_Text_BERT](aan_bert.ipynb)  presented in the paper.

## Using the Model
AANs can be extended to other different tasks such as Image Classification easily.
 

## Citation
```
@inproceedings{zhang2021discriminative,
  title={Discriminative Feature Adaptation via Conditional Mean Discrepancy for Cross-domain Text Classification},
  author={Zhang, Bo and Zhang, Xiaoming and Liu, Yun and Chen, Lei},
  booktitle={International Conference on Database Systems for Advanced Applications (DASFAA)},
  year={2021}
}
```


