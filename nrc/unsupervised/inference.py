import os, sys

path = os.getcwd()
sys.path.append(path)

from nrc import InferenceModel
import json, jsonlines
import numpy as np

from tqdm import tqdm
import logging

def preprocess(question, answer, action):
    if action in ["rtd", "lm"]:
        question = f"Question: {question}"
        answer = f"Answer: {answer}."
    else:
        question = f"Q: {question}"
        answer = f"A: {answer}."
        
    return {"question":question,"answer":answer}

if __name__ == "__main__":
    
    config = json.load(open("config/unsupervised.json"))

    logging.basicConfig(level=logging.INFO)

    logger = logging.getLogger(__name__)
    
    logger.info(config)

    if config["model_path"] in [
        f"google/electra-small-discriminator",
        f"google/electra-base-discriminator",
        f"google/electra-large-discriminator",
    ]:
        config["action"] = "rtd"

    logger.info("Loading the Model for Reasoning...")
    inference = InferenceModel(**config)

    logger.info("Reading the dataset...")
    items = [item for item in jsonlines.open(config["data_path"])]

    accuracy = []

    bar = tqdm(items)

    for item in bar:

        question = item['question']['stem']

        answers =  [answer["text"] for answer in item['question']['choices']]

        labels =  [answer["label"] for answer in item['question']['choices']]

        qas = [preprocess(question, answer, config["action"]) for answer in answers]

        scores = [inference.inference(qa["question"], qa["answer"], target=config["target"]) for qa in qas]

        p, g = labels[np.argmax(scores)], item['answerKey']
        accuracy.append(p==g)

        bar.set_description(f"@Inference (Unsupervised) #Model:{config['model_path']} #Accuracy:{np.mean(accuracy)*100:.3}%")
    
    logger.info(f"@Inference (Unsupervised) #Model:{config['model_path']} #Accuracy:{np.mean(accuracy)*100:.3}%")