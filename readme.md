# An Annotated Dataset for Explainable Interpersonal Risk Factors of Mental Disturbance in Social Media Posts

This repository contains the dataset and code for the paper "An Annotated Dataset for Explainable Interpersonal Risk Factors of Mental Disturbance in Social Media Posts" (accepted at Findings of ACL 2023).

## Abstract

With a surge in identifying suicidal risk and its severity in social media posts, we argue that a more consequential and explainable research is required for optimal impact on clinical psychology practice and personalized mental healthcare. The success of computational intelligence techniques for inferring mental illness from social media resources, points to natural language processing as a lens for determining Interpersonal Risk Factors (IRF) in human writings. Motivated with limited availability of datasets for social NLP research community, we construct and release a new annotated dataset with human-labelled explanations and classification of IRF affecting mental disturbance on social media: (i) Thwarted Belongingness (TBe), and (ii) Perceived Burdensomeness (PBu). We establish baseline models on our dataset facilitating future research directions to develop real-time personalized AI models by detecting patterns of TBe and PBu in emotional spectrum of user's historical social media profile.

## Dataset

The dataset is available in the `data` folder. It contains 3522 annotated samples in the following files:

- train.csv: the training set
- val.csv: the validation set
- test.csv: the test set

Each file contains the following columns:

- `text`: the text of the post
- `belong`: the label for the Thwarted Belongingness (TBe)
- `belong_exp`: the explanation for the Thwarted Belongingness (TBe)
- `burden`: the label for the Perceived Burdensomeness (PBu)
- `burden_exp`: the explanation for the Perceived Burdensomeness (PBu)

## Code

The code is available in the `src` folder. It contains the source code required to reproduce the results of the paper. The code is written in Python 3.8 and the required packages are listed in `requirements.txt`. To install the required packages, run the following command:

```bash
pip install -r requirements.txt
```

## Benchmark Model Results

### Classification

| **Model**      | Thwarted Belongingness |        |           |           | Perceived Burdensomeness |        |           |           |
| -------------- | :--------------------: | :----: | :-------: | :-------: | :----------------------: | :----: | :-------: | :-------: |
|                |       Precision        | Recall | F1-score  | Accuracy  |        Precision         | Recall | F1-score  | Accuracy  |
| **LSTM**       |         61.40          | 92.77  |   72.00   |   63.67   |          44.65           | 80.90  |   54.69   |   62.35   |
| **GRU**        |         63.57          | 91.26  | **73.06** | **66.70** |          60.87           | 74.77  | **63.75** | **78.90** |
| **BERT**       |         69.70          | 76.97  |   72.30   |   68.97   |          56.47           | 53.00  |   52.20   |   72.56   |
| **RoBERTa**    |         71.23          | 73.54  |   71.35   |   68.97   |          67.27           | 37.52  |   45.51   |   74.93   |
| **DistilBERT** |         70.24          | 74.08  |   71.15   |   68.50   |          51.15           | 31.89  |   36.93   |   71.71   |
| **MentalBERT** |         77.97          | 77.40  | **76.73** | **75.12** |          64.22           | 65.75  | **62.77** | **78.33** |
| **OpenAI+LR**  |         79.00          | 83.59  | **81.23** |   78.62   |          82.66           | 63.08  |   71.55   |   84.58   |
| **OpenAI+RF**  |         79.06          | 80.68  |   79.86   |   77.48   |          83.33           | 49.23  |   61.90   |   81.36   |
| **OpenAI+SVM** |         81.31          | 80.34  |   80.83   | **78.90** |          79.15           | 74.77  | **76.90** | **86.19** |
| **OpenAI+MLP** |         81.40          | 75.56  |   78.37   |   76.92   |          72.08           | 77.85  |   74.85   |   83.92   |
| **OpenAI+XGB** |         81.22          | 79.83  |   80.52   |   78.62   |          80.36           | 68.00  |   73.67   |   85.05   |

### Explanability (ROUGE-1)

| Model | Task | ROUGE-1 Precision | ROUGE-1 Recall | ROUGE-1 F1 |
| :---: | :--: | :---------------: | :------------: | :--------: |
| LIME  | TBe  |       14.24       |     53.05      |   20.88    |
| LIME  | PBu  |       18.47       |     46.83      |   25.18    |
| SHAP  | TBe  |       15.74       |     50.16      |   22.27    |
| SHAP  | PBu  |       20.77       |     49.89      |   27.92    |

## Citation

If you use the dataset or code in your work, please cite the following paper:

```
@inproceedings{garg-etal-2023-an,
    title = "An Annotated Dataset for Explainable Interpersonal Risk Factors of Mental Disturbance in Social Media Posts",
    author = "Garg, Muskan  and
      Shahbandegan, Amirmohammad  and
      Chadha, Amrit  and
      Mago, Vijay",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2023",
    year = "2023",
    publisher = "Association for Computational Linguistics"
}
```
