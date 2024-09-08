# Accurate Stock Movement Prediction via Multi-Scale and Multi-Domain Modeling

This project is a PyTorch implementation of Accurate Stock Movement Prediction via Multi-Scale and Multi-Domain Modeling. 
This paper proposes ZoomStock, an accurate method for stock movement prediction. ZoomStock captures complex patterns in stock price data with multi-scale and multi-domain modeling.

## Prerequisites

Our implementation is based on Python 3.7 and PyTorch.

- Python 3.7
- PyTorch 1.4.0

## Datasets

We use 6 datasets in our work, which are not included in this repository due to
their size but can be downloaded easily by following the url linked to each of the dataset names. You can run
`preprocess.py` to preprocess the stock data, and `data.py` to access the data.

|Name    | Country|  Stocks|     Days|Dates|
|:-------|------:|------:|---------:|-------:|
|[ACL18](https://github.com/fulifeng/Adv-ALSTM)      |  US| 87|   652|      2013-06-03 to 2015-12-31|
|[ACL23](https://github.com/anonymous231129/ZoomStock) |    US| 87|    504|       2018-01-02 to 2023-04-27|
|[KOSPI](https://github.com/anonymous231129/ZoomStock)   |    South Korea|  200|     1528|       2018-01-02 to 2023-11-09|
|[HK23](https://github.com/anonymous231129/ZoomStock)    |  Hong Kong| 26|   1528|      2018-01-02 to 2023-11-09|
|[TWSE23](https://github.com/anonymous231129/ZoomStock)  |  Taiwan| 37|   1528|      2018-01-02 to 2023-11-09|
|[DE23](https://github.com/anonymous231129/ZoomStock)|  Germany| 23|    1528|       2018-01-02 to 2023-11-09|

## Usage

We explain the code of ZoomStock which reproduces the experimental results of our paper. The `run.py` file will run the code for training and evaluating ZoomStock. Moreover, the training and evaluation will occur five times with different seeds. You also need to specify which dataset you would like to train the model with. In other words, you just have to type the following command.

```
python run.py --data acl18
```

This demo script will train and evaluate ZoomStock using the acl18 dataset. You also have the option of modifying the `main.py` file to use different hyperparameters.