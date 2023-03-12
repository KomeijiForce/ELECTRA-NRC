# Non-Replacement Confidence (NRC)
The official implementation for paper "Evaluate Confidence Instead of Perplexity for Commonsense Reasoning"

# Environment
Use
```pip install -r requirements.txt```
to install dependencies. You can also check the environment here:
```
NumPy: 1.23.5
PyTorch: 1.13.0
Transformers: 4.25.1
jsonlines
```
# How to run?
First download the datasets with scripts in ```./data/```, e.g.:
```
bash data/download.csqa.sh
```
This will download the CommonsenseQA dataset to ```./data/CSQA/```. Then use
```
python nrc/unsupervised/inference.py
```
to run reasoning on the CommonsenseQA dev dataset. You can edit ```./config/unsupervised.json``` to run experiments with different models on different datasets. Current, we support running pre-trained language models with objective: masked language modeling, casual language modeling, or replaced token detection. Notice that DeBERTaV3 is not available because we current cannot load it to calculate NRC.
