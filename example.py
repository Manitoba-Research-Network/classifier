import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
import yaml

DEVICE = "cpu"
MODEL_NAME = "meta-llama/Llama-3.2-1B"
OUTPUT_DIR = "output"
DATA_PATH = "data"
RANDOM_SEED = 42

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
    dataset = load_dataset("json", data_files=path)
    dataset = dataset["train"].train_test_split(test_size=0.2, seed=RANDOM_SEED)
    print("Dataset loaded.")
    print(f"Train size: {len(dataset['train'])}")
    print(f"Test size: {len(dataset['test'])}")
    print(f"Dataset structure: {dataset}")
    return dataset

if __name__ == "__main__":
    # label maps
    id2label = {0: "Normal", 1: "Suspicious"}
    label2id = {v:k for k,v in id2label.items()}

    # Load the config file
    config = load_config()
    set_config(config)

    # clear cache
    clear_cache()

    # Load the model and tokenizer from the output directory
    model = AutoModelForSequenceClassification.from_pretrained(
        OUTPUT_DIR,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id).to(DEVICE)
    
    tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR, add_prefix_space=True)

    # add pad token if none exists
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        model.resize_token_embeddings(len(tokenizer))
    print("Model and tokenizer loaded.")

    #load the data
    dataset = load_output_dataset(DATA_PATH)

    print("trained model predictions:")
    print("--------------------------")
    isCorrect_trained = 0
    total_trained = 0
    accuracy_trained = 0
    not_zero = 0
    log_interval = 100
    for i, entry in enumerate(dataset["test"]):
        text = entry["text"]
        total_trained += 1
        try:
            inputs = tokenizer.encode(text, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                logits = model(inputs).logits
            predictions = torch.argmax(logits)
            if predictions == entry["label"]:
                isCorrect_trained += 1
            if predictions != 0:
                not_zero += 1
    
            if (i) % log_interval == 0:
                print(f"Processed: {total_trained}, Correct: {isCorrect_trained}, not0: {not_zero}", end="\r")
        except:
            print("Skipped one row")
            total_trained -= 1

    accuracy_trained = isCorrect_trained / total_trained
    print(f"Accuracy: {accuracy_trained}")