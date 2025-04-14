# elastic_data_classifier

Trainer script for training Llama model with given config setting and data, then output the model to specific directory.

## Usage

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
`device`: device to run the model on (cuda or cpu)
`data_path`: Path to the data file
`output_dir`: Path to the output directory
`random_seed`: Random seed for reproducibility
`max_length`: Maximum token length for the model
`learning_rate`: Learning rate for the trainer
`batch_size`: Batch size for training
`epochs`: Number of epochs to train the model

### Running

```bash
python3 classifier
```
