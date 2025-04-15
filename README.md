# elastic_data_classifier

Trainer script for training Llama model with given config setting and data, then output the model to specific directory.

## Trainer Script Usage

### Environment

#### Python Packages

Install required packages with:

```pip install -r requirements.txt```

#### .env file

The script requires a file called `.env` with the following format:

```env
hugging_face_PAG = <paste your hugging face token here>
```

#### config.ymal

`model_name`:  model name to be trained
`pretrained_model_exists`: true if the pretrained exist, false if not exist

- if pretrained model not exist, the program will download the pretrained model from hugging face using the PAG given by user
  - need to make sure the PAG have the access to that particular model
- if you want to install the pretrained model seperately, please put the files in the order below
  - <meta-llama> -> <model name> -> <the model files, including the tokenizer>
- if you want to download the pretrained model again, please also make sure you delete every files and directories in <meta-llama>

`device`: device to run the model on (cuda or cpu)
`data_path`: Path to the data file
`output_dir`: Path to the output directory
`random_seed`: Random seed for reproducibility
`max_length`: Maximum token length for the model

- this parameter is postive correlated to the model training time

`learning_rate`: Learning rate for the trainer
`batch_size`: Batch size for training
`epochs`: Number of epochs to train the model

### Running

```bash
python3 classifier.py
```

## Model Usage

- the output folder will contains model file and the model tokenizer

- the prediction usage is:

```python
# label maps
id2label = {0: "Normal", 1: "Suspicious"}
label2id = {v:k for k,v in id2label.items()}

# load the trained model and tokenizer from the output path
model = AutoModelForSequenceClassification.from_pretrained(
    trained_model_path,
    num_labels=len(id2label),
    id2label=id2label,
    label2id=label2id).to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained(trained_tokenizer_path, add_prefix_space=True)

# add pad token if none exists
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    pretrained_model.resize_token_embeddings(len(tokenizer))

# input the text to the model
inputs = tokenizer.encode(text, return_tensors="pt").to(DEVICE)
# get the logits from the trained model
with torch.no_grad():
  logits = model(inputs).logits
# get the output label
predictions = torch.argmax(logits)
```
