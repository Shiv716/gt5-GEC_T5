# -----------
# Author: Shivang Chaudhary
# Course: MSc Artificial Intelligence
# Year: 2023-24
# Following is the GEC 'gt5' that is experimented on limited versions of cLang-8 and CoNLL-2014 dataset
# -----------

import os
import random

import pandas as pd
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, \
    TextClassificationPipeline, AutoModelForSequenceClassification, DataCollatorWithPadding, AutoModelForSeq2SeqLM, \
    pipeline, AdamW, T5Config
import evaluate

# LOADING THE DATASET FOR THE MODEL
# Load CLang-8 dataset : The dataset labelling -> Column1= all incorrect sentences, Column2= all correct sentences
df = pd.read_csv('Data/clang8.csv')

# Create Hugging Face Dataset
# dataset = Dataset.from_pandas(df)

# LOAD TOKENIZER
tokenizer = AutoTokenizer.from_pretrained('t5-base')

# ----Introducing dropout:-
config = T5Config.from_pretrained('t5-base', dropout_rate=0.1)
# ----

# Load pretrained model :-
model = T5ForConditionalGeneration.from_pretrained('t5-base',
                                                   config=config
                                                   # num_labels=len(unique_labels),
                                                   # label2id=label2id,
                                                   # id2label=id2label,
                                                   )


# ------- Applying data-augmentation
# Function introduces random errors.
def introduce_errors(sentence):
    words = sentence.split()
    if len(words) < 2:
        # Skips very short sentences
        return sentence

    error_type = random.choice(['swap', 'insert', 'delete', 'replace'])

    if error_type == 'swap' and len(words) > 1:
        # Introduces a swap error between random adjacent words
        index = random.randint(0, len(words) - 2)
        words[index], words[index + 1] = words[index + 1], words[index]
    elif error_type == 'insert':
        # Inserting a random word at a random position
        random_word = random.choice(words)
        index = random.randint(0, len(words))
        words.insert(index, random_word)
    elif error_type == 'delete' and len(words) > 1:
        # Deleting a random word
        index = random.randint(0, len(words) - 1)
        words.pop(index)
    elif error_type == 'replace':
        # Replacing a random word with another random word
        index = random.randint(0, len(words) - 1)
        random_word = random.choice(words)
        words[index] = random_word

    return ' '.join(words)


# Creating a new column with augmented data
df['augmented'] = df['Column1'].apply(introduce_errors)

# Combine the original and augmented data
augmented_df = pd.DataFrame({
    'Column1': pd.concat([df['Column1'], df['augmented']]),
    'Column2': pd.concat([df['Column2'], df['Column2']])
})

# Updated dataset for further experimentation:-
dataset = Dataset.from_pandas(augmented_df)

# ------------------------------------


# Splitting the dataset
train_test_split = dataset.train_test_split(test_size=0.3)
train_dataset = train_test_split['train']
eval_dataset = train_test_split['test']


# ------


# PRE-PROCESSING FUNCTION:-
def preprocess_function(examples):
    inputs = examples['Column1']
    targets = examples['Column2']

    model_inputs = tokenizer(inputs, padding='max_length', truncation='longest_first', return_tensors='pt'
                             , max_length=512)

    # with tokenizer.as_target_tokenizer():
    labels = tokenizer(targets, padding='max_length', truncation='longest_first', return_tensors='pt'
                       , max_length=512)

    model_inputs['labels'] = labels['input_ids']
    # Used for testing purposes ---
    # print(model_inputs)
    # print(labels)
    # ---
    return model_inputs


# -------Retaining the dataset for backup (if required in later processing phase.)
original_eval_dataset = eval_dataset
# --------

# -- MAKING THE TRAIN AND VALIDATION SET
train_dataset = train_dataset.map(preprocess_function, batched=True)
eval_dataset = eval_dataset.map(preprocess_function, batched=True)
# print(eval_dataset) -> For testing phase.
# ----


# Setting up evaluation - THIS IS BACKUP-METRIC, NOT THE FOCUS OF THE PROJECT.
metric = evaluate.load('accuracy')

# Defining training arguments
training_args = TrainingArguments(
    output_dir='./results_clang_10epc', # Store the results for epochs 3,6 and 10 respectively.
    evaluation_strategy='steps',
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=10,  # Can experiment with epochs, diverse results expected with 3,6 and 10.
    weight_decay=0.01,
    save_steps=1000,  # Saving checkpoint every 1000 steps
    save_total_limit=3,  # Limit the total amount of checkpoints. Deletes the older checkpoints.
    logging_dir='./logs_clang8_tuned',  # Directory for storing logs, can be renamed for different epochs to store data.
    logging_steps=1000,  # Log every 1000 steps
    load_best_model_at_end=True,  # Load the best model when finished training
)
# --> metric_for_best_model="accuracy" -> NOT THE PROJECT'S FOCUS.


# Define custom data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
# -----

# ------- Adding AdamW optimizer
optimizer = AdamW(model.parameters(), lr=training_args.learning_rate)
# ------------------------------

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    optimizers=(optimizer, None),
    # compute_metrics=compute_metrics ( IF REQUIRED )
)

# Training the model
trainer.train()

# ------- BUILDING THE GEC PIPE-LINE
gec_pipeline = pipeline(task='text2text-generation', model=model, tokenizer=tokenizer)
# Save predictions and references
predictions = gec_pipeline([example['Column1'] for example in eval_dataset], max_length=512, num_return_sequences=1)
pred_texts = [prediction['generated_text'] for prediction in predictions]
references = [example['Column2'] for example in eval_dataset]
# ------


# ---------------------------------------------------------------------------------

# Err_files_evals for storing the corrected sentences and scores respectively.
# Saving to verify:-
dir_path = 'Err_files_evals'
# Verify if the directory is present or not:-
if not os.path.exists(dir_path):
    os.makedirs(dir_path)
    print("Directory", dir_path, "created.")
else:
    print("Directory", dir_path, "already exists.")

# ---
df = pd.DataFrame(predictions)
df.to_csv("Err_files_evals/predictions.csv", index=False)  # Save to CSV
# ---


# Save predictions and references-----
with open('Err_files_evals/predictions.txt', 'w') as pred_file, \
        open('Err_files_evals/references.txt', 'w') as ref_file:
    for pred, ref in zip(pred_texts, eval_dataset['Column2']):
        pred_file.write(pred.strip() + '\n')
        ref_file.write(ref.strip() + '\n')
# -----

# ---- Evaluation phase: Obtaining the needed files:-
# Getting the original inputs:-
# Prepare input files for ERRANT
with open('Err_files_evals/clang8.incorrect', 'w') as orig_file:
    for incorrect in eval_dataset['Column1']:
        orig_file.write(incorrect.strip() + '\n')

# ------

# Ensure the lengths of dataset['Column1'] and pred_texts are the same:-
assert len(eval_dataset['Column1']) == len(pred_texts), "Mismatch in number of original texts and predictions"
# ------

# -- FINAL TESTING THROUGH ERRANT ON NEIGHBOURING SCRIPT, 'Gecmodel_errEval.py'.
