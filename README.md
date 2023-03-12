# Non-Replacement Confidence (NRC)
The official implementation for paper "Evaluate Confidence Instead of Perplexity for Commonsense Reasoning"

NRC is a novel metric for both supervised and unsupervised commonsense reasoning. It is based on the ELECTRA architecture and has unique properties such as **Equal Synonym Positiveness** and **Negative Sample Learning**, which align it more closely with the nature of commonsense reasoning.

Thanks to these properties, NRC has shown impressive performance in evaluating unsupervised commonsense reasoning. Moreover, it can also serve as an objective for tuning models to achieve better results in supervised settings.

Overall, NRC offers a promising new approach to measuring and improving commonsense reasoning in natural language processing. Its innovative design and strong performance make it a valuable tool for researchers and practitioners alike.

![image](https://github.com/KomeijiForce/ELECTRA-NRC/blob/main/instance.png)

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
to run reasoning on the CommonsenseQA dev dataset. You can edit ```./config/unsupervised.json``` to run experiments with different models on different datasets. Current, we support running pre-trained language models with objective: masked language modeling, casual language modeling, or replaced token detection. Notice that DeBERTaV3 is not available because we currently cannot load its AutoModelForPreTraining Version to calculate NRC.

# Performance

| Method                | Trg. | CSQA | ARC_E | ARC_C | COPA | Swag | SCT  | SQA  | CQA  | Avg. |
|-----------------------|--------|-------|-------|--------|-------|--------|--------|-------|--------|--------|
| PPL_GPT2-XL       | A       | 42.6  | 50.8  | 28.8   | 73.6  | 65.3   | 70.6   | 45.5  | 35.5   | 51.6   |
| PPL_GPT2-M      | A       | 38.5  | 44.4  | 24.9   | 68.4  | 59.7   | 54.0   | 44.3  | 27.0   | 45.0   |
| PPL_BERT             | Q       | 40.6  | 37.2  | 26.7   | 64.2  | 44.5   | 63.5   | 39.6  | 32.9   | 43.7   |
|                             | A       | 28.0  | 37.1  | 22.7   | 61.2  | 63.4   | 58.2   | 40.4  | 30.7   | 42.7   |
|                             | QA    | 32.8  | 36.8  | 23.7   | 64.2  | 64.1   | 61.2   | 38.5  | 29.6   | 43.9   |
| PPL_RoBERTa     | Q       | 49.3  | 40.5  | 35.6   | 70.6  | 48.1   | 61.5   | 39.7  | 38.6   | 48.0   |
|                             | A       | 39.8  | 44.2  | 27.1   | 68.4  | 71.0   | 67.3   | 45.5  | 36.1   | 49.9   |
|                             | QA    | 49.0  | 45.5  | 31.8   | 75.2  | 74.5   | 71.7   | 46.2  | 36.5   | 53.8   |
| NRC                    | Q       | 51.2  | 46.8  | 38.6   | 82.6  | 24.5   | 65.0   | 40.6  | 41.2   | 48.8   |
|                             | A       | 45.0  | 47.9  | 37.1   | 71.2  | 77.4   | 74.7   | 46.1  | 41.9   | 55.2   |
|                             | QA    | 54.1  | 52.1  | 39.8   | 78.4  | 75.4   | 77.1   | 47.7  | 44.3   | 58.6   |

*Results on Unsupervised Commonsense Reason
