import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from peft import PeftModel, PeftConfig
import yaml
import argparse

DEVICE = "cpu"
MODEL_NAME = "meta-llama/Llama-3.2-1B"
OUTPUT_DIR = "output"
DATA_PATH = "data"
RANDOM_SEED = 42

# label maps
id2label = {0: "Normal", 1: "Suspicious"}
label2id = {v: k for k, v in id2label.items()}


def load_config():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    return config

# set the config values
def set_config(config):
    global MODEL_NAME, OUTPUT_DIR, DEVICE, DATA_PATH, RANDOM_SEED
    if config["model_name"]:
        MODEL_NAME = config["model_name"]
    if config["output_dir"]:
        OUTPUT_DIR = config["output_dir"]
    if config["device"]:
        DEVICE = config["device"]
    if config["data_path"]:
        DATA_PATH = config["data_path"]
    if config["random_seed"]:
        RANDOM_SEED = int(config["random_seed"])
    print("Config loaded.")

# clear the cuda cache
def clear_cache():
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    torch.cuda.reset_peak_memory_stats()

# load the dataset
def load_output_dataset(path):
    """
    load the dataset from a path

    :param path: path to jsonl data
    :return: dataset object
    """
    dataset = load_dataset("json", data_files=path)
    print("Dataset loaded.")
    print(f"Size: {len(dataset['train'])}")
    print(f"Dataset structure: {dataset}")
    return dataset


def run(data_path):
    """
    run the model
    """
    global config, logits
    # Load the model and tokenizer from the output directory
    base_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id)
    config = PeftConfig.from_pretrained(OUTPUT_DIR)
    model = PeftModel.from_pretrained(base_model, OUTPUT_DIR, peft_config=config).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR, add_prefix_space=True)

    # add pad token if none exists
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        model.resize_token_embeddings(len(tokenizer))
    print("Model and tokenizer loaded.")

    # load the data
    dataset = load_output_dataset(data_path)
    print("trained model predictions:")
    print("--------------------------")

    isCorrect_trained = 0
    total_trained = 0
    accuracy_trained = 0
    not_zero = 0
    log_interval = 100
    suspicious = []
    for i, entry in enumerate(dataset["train"]):
        text = entry["text"]
        total_trained += 1
        try:
            inputs = tokenizer.encode(text, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                # the raw logits of the model
                logits = model(inputs).logits
            # the prediction
            predictions = torch.argmax(logits)
            if predictions == entry["label"]:
                isCorrect_trained += 1
            if predictions != 0:
                not_zero += 1
                suspicious.append(entry)

            if (i) % log_interval == 0:
                print(f"Processed: {total_trained}, Correct: {isCorrect_trained}, not0: {not_zero}", end="\r")
        except:
            print("Skipped one row")
            total_trained -= 1
    accuracy_trained = isCorrect_trained / total_trained
    print(f"Accuracy: {accuracy_trained}")

    return suspicious


if __name__ == "__main__":

    # Load the config file
    config = load_config()
    set_config(config)

    #load command line args
    parser = argparse.ArgumentParser(
        prog="Basic Model Script",
        description="Basic script for using model trained by classifier.py"
    )
    parser.add_argument('-d', '--data',
                        help="path to jsonl data",
                        type=str,
                        default=DATA_PATH)

    args = parser.parse_args()
    # clear cache
    clear_cache()


    sus = run(args.data)
    print("Suspicious entries:")
    for entry in sus:
        print(f"id: {entry['id']} index: {entry['idx']}")
    if len(sus) ==0:
        print("no suspicious entries found")