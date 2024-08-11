# LegalLens NER Task 
Our model, **LM_CRF_NER** , is a custom Language Model integrated with Conditional Random Fields designed for Named Entity Recognition (NER).
## Model Overview 
 
- **Architecture:**  Language Model + Conditional Random Fields (LM + CRF)
 
- **Language Model:**  Legal Longformer (selected for best performance on the evaluation set)
 
- **Task:**  Named Entity Recognition (NER)
 
- **Datasets:**  [LegalLensNER (HuggingFace)](https://huggingface.co/datasets/darrow-ai/LegalLensNER) and the private test set
 
- **Performance on the Evaluation Set:**

| Language Model | Macro Precision | Macro Recall | Macro F1 | 
| --- | --- | --- | --- | 
| bert-base-cased | 0.8675 | 0.8904 | 0.8780 | 
| nlpaueb/legal-bert-base-uncased | 0.8946 | 0.8907 | 0.8920 | 
| dslim/bert-base-NER | 0.8876 | 0.8925 | 0.8895 | 
| roberta-base | 0.8943 | 0.9002 | 0.8968 | 
| lexlms/legal-roberta-base | 0.9254 | 0.8939 | 0.9089 | 
| allenai/longformer-base-4096 | 0.8938 | 0.8861 | 0.8891 | 
| **lexlms/legal-longformer-base** | **0.9264** | **0.9217** | **0.9238** |

## Getting Started 

To get started, follow the instructions below to set up your environment and run the model:

### 1. Configure Settings 

All necessary settings are managed in the "All settings" section of the notebook. You can modify these settings according to your requirements.

### 2. Specify Input and Output Directories 

Ensure the input data folder contains the following three files:
 
- **Train Set:**  `trainset_NER_LegalLens.csv`
 
- **Dev Set:**  `devset_NER_LegalLens.csv`
 
- **Test Set:**  `testset_NER_LegalLens.xlsx`


```python
# Files directory
data_dir = '/kaggle/input/l-ner-data/' # Update this path with your data directory containing the train, dev, and test sets.
output_dir = '/kaggle/working/'        # Update this path to where you want to save the model checkpoints.
```

### 3. Run the Training Procedure 

You can choose whether to run the training procedure and whether to load the trained model locally or from our HuggingFace's API. The API key is already provided in the code: ```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```


```python
do_train = True                        # Set to True to run the training procedure.
use_local_trained_model = True         # Set to True to use the locally trained model.
                                       # Set to False to load the trained model from Hugging Face. If do_train == False, this will be set to False.
```

### 4. Modify Model and Training Settings 

Adjust the following settings to customize the model and training parameters:


```python
# Model settings
language_model_name = 'lexlms/legal-longformer-base' # Replace with other language models from Hugging Face if desired.
do_lower_case = False
max_seq_length = 256

# Training settings
batch_size = 16
learning_rate0 = 5e-5
lr0_crf_fc = 8e-5
weight_decay_finetune = 1e-5
weight_decay_crf_fc = 5e-6
total_train_epochs = 30
gradient_accumulation_steps = 1
warmup_proportion = 0.1
```

## Conclusion 

Once you’ve configured the settings and data, you’re ready to run the code and generate the results! 
See the ```Methodology Description.pdf``` file for more details.
