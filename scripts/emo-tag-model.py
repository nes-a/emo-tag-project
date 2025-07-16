import pandas as pd
import ast
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F
from datasets import Dataset
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, f1_score, precision_score, recall_score, classification_report
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback, DataCollatorWithPadding
from datasets import Dataset # This import is redundant, already imported above
import wandb
import os

# Keep if you intend to use wandb; remove/set to "true" if not.
os.environ["WANDB_DISABLED"] = "false" 

print("All imports processed.")

# --- Data Paths ---
# IMPORTANT: Update these paths to where you've stored your data.
# These paths are relative to where you run this script.
# Ensure your 'data' folder is structured as:
# data/
# ├── go-emotions/
# │   ├── train.tsv
# │   ├── dev.tsv
# │   └── test.tsv
# └── synthetic/
#     └── synthetic-emotion-data.csv (or your synthetic data file name)

GO_EMOTIONS_BASE_DIR = "./data/go-emotions/"
SYNTHETIC_DATA_PATH = "./data/synthetic/synthetic-emotion-data.csv" # Ensure this matches your generate-messages.py output

go_emotions_train_filepath = os.path.join(GO_EMOTIONS_BASE_DIR, "train.tsv")
go_emotions_dev_filepath = os.path.join(GO_EMOTIONS_BASE_DIR, "dev.tsv")
go_emotions_test_filepath = os.path.join(GO_EMOTIONS_BASE_DIR, "test.tsv")
synthetic_data_filepath = SYNTHETIC_DATA_PATH # Correctly references the variable above

# --- Model Save Path ---
# IMPORTANT: Update this path to where you want to save your trained model.
# This path is relative to where you run this script.
MODEL_SAVE_PATH = "./models/emo-tag/"

# --- Constants and Mappings ---
GOEMOTIONS_LABELS = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion',
    'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment',
    'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism',
    'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]

GOEMOTIONS_ID_TO_LABEL = {
    0: 'admiration', 1: 'amusement', 2: 'anger', 3: 'annoyance', 4: 'approval', 5: 'caring', 6: 'confusion',
    7: 'curiosity', 8: 'desire', 9: 'disappointment', 10: 'disapproval', 11: 'disgust', 12: 'embarrassment',
    13: 'excitement', 14: 'fear', 15: 'gratitude', 16: 'grief', 17: 'joy', 18: 'love', 19: 'nervousness',
    20: 'optimism', 21: 'pride', 22: 'realization', 23: 'relief', 24: 'remorse', 25: 'sadness', 26: 'surprise'
}

# --- Data Parsing and Loading Functions ---
def parse_synthetic_emotion_string(emotion_str):
    """
    Parses a string representation of a list of emotions into an actual list.
    Handles potential 'neutral' or empty lists, and single string emotions.
    """
    if pd.isna(emotion_str):
        return ['neutral']

    try:
        parsed_list = ast.literal_eval(emotion_str)
        if isinstance(parsed_list, list):
            return parsed_list if parsed_list else ['neutral']
        else:
            return [str(parsed_list)] if str(parsed_list).strip() else ['neutral']
    except (ValueError, SyntaxError):
        if emotion_str.strip():
            return [emotion_str.strip()]
        else:
            return ['neutral']

def load_and_process_go_emotions_split(filepath):
    """
    Loads and processes a GoEmotions TSV split, assigning 'neutral' where no other emotions are found.
    """
    try:
        df_split = pd.read_csv(filepath, sep='\t', encoding='utf-8', header=None, names=['text', 'emotion_ids_str', 'comment_id'])

        df_split['emotion_ids_list'] = df_split['emotion_ids_str'].apply(
            lambda x: [int(label_id) for label_id in str(x).split(',') if label_id.strip().isdigit()]
        )

        df_split['emotion_names_raw'] = df_split['emotion_ids_list'].apply(
            lambda ids: [GOEMOTIONS_ID_TO_LABEL[idx] for idx in ids if idx in GOEMOTIONS_ID_TO_LABEL]
        )

        df_split['emotion'] = df_split['emotion_names_raw'].apply(
            lambda x: ['neutral'] if not x else x
        )

        df_processed = df_split[['text', 'emotion']].copy()
        return df_processed
    except FileNotFoundError:
        print(f"Error: {filepath.split('/')[-1]} not found. Please ensure all 3 GoEmotions TSV files are in the specified data path.")
        return pd.DataFrame(columns=['text', 'emotion'])
    except Exception as e:
        print(f"An error occurred loading or processing {filepath.split('/')[-1]}: {e}")
        return pd.DataFrame(columns=['text', 'emotion'])

print("--------------------------------------------------------------------------------")

# --- Load and Process Synthetic Data ---
print("\n--- Loading synthetic data ---")
try:
    df_synthetic = pd.read_csv(synthetic_data_filepath)
    if 'emotion' in df_synthetic.columns:
        df_synthetic['emotion'] = df_synthetic['emotion'].apply(parse_synthetic_emotion_string)
        print(f"Loaded and processed synthetic data. Shape: {df_synthetic.shape}")
    else:
        print(f"Error: 'emotion' column not found in synthetic data file: {synthetic_data_filepath}")
        print(f"Available columns: {df_synthetic.columns.tolist()}")
        df_synthetic = pd.DataFrame(columns=['text', 'emotion']) # Create empty DF to prevent errors
except FileNotFoundError:
    print(f"Error: Synthetic data file not found at {synthetic_data_filepath}. Please check the path.")
    df_synthetic = pd.DataFrame(columns=['text', 'emotion']) # Create empty DF to prevent errors
except Exception as e:
    print(f"An error occurred loading or processing synthetic data: {e}")
    df_synthetic = pd.DataFrame(columns=['text', 'emotion']) # Create empty DF to prevent errors
print("--------------------------------------------------------------------------------")

# --- Combine Datasets ---
print("\n--- Combining datasets ---")
if not df_go_emotions_train.empty and not df_synthetic.empty and \
   'text' in df_go_emotions_train.columns and 'emotion' in df_go_emotions_train.columns and \
   'text' in df_synthetic.columns and 'emotion' in df_synthetic.columns:
    train_df = pd.concat([df_go_emotions_train, df_synthetic], ignore_index=True)
else:
    print("Column names mismatch, dataframes are empty, or loading failed. Creating empty train_df.")
    train_df = pd.DataFrame(columns=['text', 'emotion'])

val_df = df_go_emotions_val
test_df = df_go_emotions_test

print(f"Combined train_df shape: {train_df.shape}")
print(f"Val_df shape: {val_df.shape}")
print(f"Test_df shape: {test_df.shape}")
print("--------------------------------------------------------------------------------")

# --- Verify Combined Train Dataset Emotion Frequencies ---
print("\n--- Verifying Combined Train Dataset Emotion Frequencies ---")
mlb = MultiLabelBinarizer(classes=GOEMOTIONS_LABELS)

if not train_df.empty:
    labels_binary = mlb.fit_transform(train_df['emotion'])
    df_labels_binary = pd.DataFrame(labels_binary, columns=mlb.classes_)
    combined_train_counts = df_labels_binary.sum().sort_values(ascending=False)

    print("\nEmotion Frequencies in the COMBINED TRAIN dataset:")
    print(combined_train_counts)

    plt.figure(figsize=(18, 9))
    sns.barplot(x=combined_train_counts.index, y=combined_train_counts.values, palette='viridis')
    plt.title('Emotion Frequencies: Combined Train Data (GoEmotions + Synthetic)', fontsize=18)
    plt.xlabel('Emotion', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
else:
    print("Combined train_df is empty. Cannot calculate and visualize frequencies.")
print("--------------------------------------------------------------------------------")

# --- Convert to Hugging Face Dataset ---
print("\n--- Converting to Hugging Face Dataset ---")
if not val_df.empty:
    val_labels_binary = mlb.transform(val_df['emotion'])
    val_hf_dataset = Dataset.from_pandas(pd.DataFrame({'text': val_df['text'], 'labels': val_labels_binary.tolist()}))
else:
    val_hf_dataset = Dataset.from_pandas(pd.DataFrame(columns=['text', 'labels']))

if not test_df.empty:
    test_labels_binary = mlb.transform(test_df['emotion'])
    test_hf_dataset = Dataset.from_pandas(pd.DataFrame({'text': test_df['text'], 'labels': test_labels_binary.tolist()}))
else:
    test_hf_dataset = Dataset.from_pandas(pd.DataFrame(columns=['text', 'labels']))

if not train_df.empty:
    train_hf_dataset = Dataset.from_pandas(pd.DataFrame({'text': train_df['text'], 'labels': labels_binary.tolist()}))
else:
    train_hf_dataset = Dataset.from_pandas(pd.DataFrame(columns=['text', 'labels']))

print("Hugging Face Datasets created.")
print("--------------------------------------------------------------------------------")

# --- Undersampling of 'neutral' Class ---
print("\n--- Starting Undersampling of 'neutral' Class ---")

neutral_examples = []
other_emotion_examples = []

if not train_df.empty:
    for index, row in train_df.iterrows():
        if len(row['emotion']) == 1 and 'neutral' in row['emotion']:
            neutral_examples.append(row)
        else:
            other_emotion_examples.append(row)

    df_neutral_class = pd.DataFrame(neutral_examples)
    df_other_emotions_class = pd.DataFrame(other_emotion_examples)

    target_neutral_count = 2750
    if len(df_neutral_class) > target_neutral_count:
        df_neutral_undersampled = df_neutral_class.sample(n=target_neutral_count, random_state=42)
        print(f"Undersampled 'neutral' from {len(df_neutral_class)} to {len(df_neutral_undersampled)} examples.")
    else:
        df_neutral_undersampled = df_neutral_class
        print(f"'neutral' count ({len(df_neutral_class)}) is already at or below the target ({target_neutral_count}). No undersampling performed.")

    train_df_undersampled = pd.concat([df_other_emotions_class, df_neutral_undersampled], ignore_index=True)
    print(f"New combined (undersampled) train_df shape: {train_df_undersampled.shape}")
else:
    train_df_undersampled = pd.DataFrame(columns=['text', 'emotion'])
    print("Original train_df is empty. Undersampling skipped. train_df_undersampled is empty.")

print("--- 'neutral' Undersampling Complete ---")
print("--------------------------------------------------------------------------------")

# --- Verify Undersampled Train Dataset Emotion Frequencies ---
print("\n--- Verifying Undersampled Train Dataset Emotion Frequencies ---")
mlb_undersampled = MultiLabelBinarizer(classes=GOEMOTIONS_LABELS)

if not train_df_undersampled.empty:
    labels_binary_undersampled = mlb_undersampled.fit_transform(train_df_undersampled['emotion'])
    df_labels_binary_undersampled = pd.DataFrame(labels_binary_undersampled, columns=mlb_undersampled.classes_)
    combined_train_counts_undersampled = df_labels_binary_undersampled.sum().sort_values(ascending=False)

    print("\nEmotion Frequencies in the UNDERSAMPLED TRAIN dataset:")
    print(combined_train_counts_undersampled)

    plt.figure(figsize=(18, 9))
    sns.barplot(x=combined_train_counts_undersampled.index, y=combined_train_counts_undersampled.values, palette='viridis')
    plt.title('Emotion Frequencies: Undersampled Train Data', fontsize=18)
    plt.xlabel('Emotion', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
else:
    print("Undersampled train_df is empty. Cannot calculate and visualize frequencies.")
print("--------------------------------------------------------------------------------")

# --- Convert Undersampled Train Dataset to Hugging Face Dataset ---
print("\n--- Converting Undersampled Train Dataset to Hugging Face Dataset ---")
if not train_df_undersampled.empty:
    train_hf_dataset_undersampled = Dataset.from_pandas(
        pd.DataFrame({'text': train_df_undersampled['text'], 'labels': labels_binary_undersampled.tolist()})
    )
    print(f"train_hf_dataset_undersampled size: {len(train_hf_dataset_undersampled)}")
else:
    train_hf_dataset_undersampled = Dataset.from_pandas(pd.DataFrame(columns=['text', 'labels']))
    print("train_hf_dataset_undersampled is empty.")
print("--------------------------------------------------------------------------------")

# --- Model Initialization ---
print("\n--- Initializing Model for Multi-label Classification ---")

model_checkpoint = "bert-base-uncased"
num_classes = len(GOEMOTIONS_LABELS)

model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint,
    num_labels=num_classes,
    problem_type="multi_label_classification" # Crucial for sigmoid activation and BCEWithLogitsLoss
)
print(f"Model '{model_checkpoint}' loaded for multi-label classification with {num_classes} labels.")
print("--------------------------------------------------------------------------------")

# --- Custom Loss Function (Focal Loss) ---
def focal_loss_with_logits(inputs, targets, alpha=0.25, gamma=2.0, pos_weight=None):
    """
    Computes the Focal Loss for multi-label classification directly from logits.

    Args:
        inputs (torch.Tensor): Raw, unscaled scores (logits) from the model. Shape: (batch_size, num_labels)
        targets (torch.Tensor): True labels (0 or 1). Shape: (batch_size, num_labels)
        alpha (float or torch.Tensor): Weighting factor for positive/negative examples.
        gamma (float): Focusing parameter. Higher values (e.g., 2.0) increase focus
                       on hard, misclassified examples.
        pos_weight (torch.Tensor): A weight of positive examples, used for class imbalance.
                                   Shape: (num_labels,) or scalar.
    Returns:
        torch.Tensor: Scalar Focal Loss.
    """
    bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none', pos_weight=pos_weight)
    prob = torch.sigmoid(inputs)
    p_t = prob * targets + (1 - prob) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    focal_term = (1 - p_t) ** gamma
    loss = alpha_t * focal_term * bce_loss
    return loss.mean()
print("Focal Loss function 'focal_loss_with_logits' defined.")
print("--------------------------------------------------------------------------------")

# --- Custom Trainer for Loss Function Integration ---
class CustomTrainer(Trainer):
    def __init__(self, *args, loss_weights=None, focal_alpha=0.25, focal_gamma=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_weights = loss_weights
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        weights_on_device = self.loss_weights.to(logits.device)
        loss = focal_loss_with_logits(
            inputs=logits,
            targets=labels.float(),
            alpha=self.focal_alpha,
            gamma=self.focal_gamma,
            pos_weight=weights_on_device
        )
        return (loss, outputs) if return_outputs else loss
print("CustomTrainer defined to incorporate Focal Loss with class weighting.")
print("--------------------------------------------------------------------------------")

# --- Tokenizer and Dataset Tokenization ---
print("\n--- Initializing Tokenizer and Tokenizing Datasets ---")
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=256)

tokenized_train_dataset = train_hf_dataset_undersampled.map(tokenize_function, batched=True)
tokenized_val_dataset = val_hf_dataset.map(tokenize_function, batched=True)
tokenized_test_dataset = test_hf_dataset.map(tokenize_function, batched=True)

tokenized_train_dataset = tokenized_train_dataset.remove_columns(['text'])
tokenized_val_dataset = tokenized_val_dataset.remove_columns(['text'])
tokenized_test_dataset = tokenized_test_dataset.remove_columns(['text'])

tokenized_train_dataset.set_format("torch")
tokenized_val_dataset.set_format("torch")
tokenized_test_dataset.set_format("torch")

print("Datasets tokenized and formatted for PyTorch.")
print("--------------------------------------------------------------------------------")

# --- Calculate Class Weights ---
print("\n--- Calculating Class Weights ---")
labels_array = np.array(train_hf_dataset_undersampled['labels'])
label_counts = np.sum(labels_array, axis=0)

epsilon = 1e-6
total_samples_in_train = len(train_hf_dataset_undersampled)
num_classes = len(GOEMOTIONS_LABELS)

weights = np.zeros(num_classes)
for i, count in enumerate(label_counts):
    weights[i] = total_samples_in_train / (num_classes * (count + epsilon))

class_weights_tensor = torch.tensor(weights, dtype=torch.float)

print("Class Frequencies (from undersampled train_hf_dataset_undersampled):")
for i, label in enumerate(GOEMOTIONS_LABELS):
    print(f"- {label}: {label_counts[i]} (Calculated Weight: {class_weights_tensor[i]:.2f})")
print("--------------------------------------------------------------------------------")

# --- Define Evaluation Metrics ---
print("\n--- Defining Evaluation Metrics ---")
def compute_metrics(p):
    predictions = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    labels = p.label_ids

    probabilities = 1 / (1 + np.exp(-predictions))
    binary_predictions = (probabilities > 0.5).astype(int)

    macro_f1 = f1_score(labels, binary_predictions, average='macro', zero_division=0)
    macro_precision = precision_score(labels, binary_predictions, average='macro', zero_division=0)
    macro_recall = recall_score(labels, binary_predictions, average='macro', zero_division=0)
    micro_f1 = f1_score(labels, binary_predictions, average='micro', zero_division=0)
    micro_precision = precision_score(labels, binary_predictions, average='micro', zero_division=0)
    micro_recall = recall_score(labels, binary_predictions, average='micro', zero_division=0)

    return {
        "f1_macro": macro_f1,
        "precision_macro": macro_precision,
        "recall_macro": macro_recall,
        "f1_micro": micro_f1,
        "precision_micro": micro_precision,
        "recall_micro": micro_recall,
    }
print("Compute metrics function 'compute_metrics' defined (focusing on macro averages).")
print("--------------------------------------------------------------------------------")

# --- Training Arguments ---
print("\n--- Setting up Training Arguments ---")
training_args = TrainingArguments(
    output_dir="./results/goemotions_model_v2",
    num_train_epochs=6,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    warmup_steps=500,
    weight_decay=0.02,
    logging_dir="./logs",
    logging_steps=100,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    dataloader_num_workers=2,
    report_to="wandb",
    fp16=torch.cuda.is_available(),
)
print("Training arguments defined.")
print("--------------------------------------------------------------------------------")

# --- Trainer Initialization and Training ---
early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience=3,
    early_stopping_threshold=0.001
)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    compute_metrics=compute_metrics,
    data_collator=data_collator,
    loss_weights=class_weights_tensor,
    focal_alpha=0.25,
    focal_gamma=2.0,
    callbacks=[early_stopping_callback],
)

print("\n--- Starting Model Training ---")
trainer.train()
print("\n--- Training Complete ---")
print("--------------------------------------------------------------------------------")

# --- Validation Predictions for Threshold Analysis ---
print("\n--- Obtaining Validation Predictions for Threshold Analysis ---")
val_predictions_output = trainer.predict(tokenized_val_dataset)
val_logits = val_predictions_output.predictions
val_true = val_predictions_output.label_ids
val_probs = 1 / (1 + np.exp(-val_logits))

thresholds = np.arange(0.05, 0.96, 0.05)
macro_f1_scores = []
micro_f1_scores = []
macro_precision_scores = []
micro_precision_scores = []
macro_recall_scores = []
micro_recall_scores = []

print("Threshold | Macro F1 | Micro F1 | Macro P  | Micro P  | Macro R  | Micro R")
print("-" * 70)

for t in thresholds:
    val_preds = (val_probs > t).astype(int)
    micro_f1 = f1_score(val_true, val_preds, average='micro', zero_division=0)
    macro_f1 = f1_score(val_true, val_preds, average='macro', zero_division=0)
    micro_p = precision_score(val_true, val_preds, average='micro', zero_division=0)
    macro_p = precision_score(val_true, val_preds, average='macro', zero_division=0)
    micro_r = recall_score(val_true, val_preds, average='micro', zero_division=0)
    macro_r = recall_score(val_true, val_preds, average='macro', zero_division=0)

    macro_f1_scores.append(macro_f1)
    micro_f1_scores.append(micro_f1)
    macro_precision_scores.append(macro_p)
    micro_precision_scores.append(micro_p)
    macro_recall_scores.append(macro_r)
    micro_recall_scores.append(micro_r)

    print(f"{t:.2f}      | {macro_f1:.4f}   | {micro_f1:.4f}   | {macro_p:.4f}   | {micro_p:.4f}   | {macro_r:.4f}   | {micro_r:.4f}")

plt.figure(figsize=(12, 6))
plt.plot(thresholds, macro_f1_scores, label='Macro F1', marker='o')
plt.plot(thresholds, micro_f1_scores, label='Micro F1', marker='o')
plt.xlabel('Threshold')
plt.ylabel('F1 Score')
plt.title('F1 Score vs. Classification Threshold')
plt.legend()
plt.grid(True)
plt.show()

optimal_macro_f1_idx = np.argmax(macro_f1_scores)
optimal_threshold_macro_f1 = thresholds[optimal_macro_f1_idx]
print(f"\nOptimal Threshold for Macro F1: {optimal_threshold_macro_f1:.2f}")
print(f"Max Macro F1 at this threshold: {macro_f1_scores[optimal_macro_f1_idx]:.4f}")
print(f"Corresponding Micro F1: {micro_f1_scores[optimal_macro_f1_idx]:.4f}")
print("--------------------------------------------------------------------------------")

# --- Final Evaluation on Test Set ---
print("\n--- Evaluating Model on Test Set ---")
final_test_results = trainer.evaluate(tokenized_test_dataset)
print(f"Final Test Set Evaluation Results: {final_test_results}")
print("--------------------------------------------------------------------------------")

# --- Model Saving ---
# To save the model for later use, uncomment and adjust the path.
# model_save_path = "models/emo-tag-model/" # Adjust to your local project structure
# os.makedirs(model_save_path, exist_ok=True)
# trainer.save_model(model_save_path)
# print(f"Best model saved to {model_save_path} directory.")

# --- Detailed Classification Report on Test Set ---
print("\n--- Generating detailed classification report on Test Set ---")
predictions_output = trainer.predict(tokenized_test_dataset)
logits = predictions_output.predictions
true_labels = predictions_output.label_ids

# Use the optimal_threshold_macro_f1 found from validation for final report
optimal_threshold = optimal_threshold_macro_f1

probabilities = 1 / (1 + np.exp(-logits))
predicted_labels = (probabilities >= optimal_threshold).astype(int)

report = classification_report(true_labels, predicted_labels,
                               target_names=GOEMOTIONS_LABELS,
                               zero_division='warn')
print(report)
print("--------------------------------------------------------------------------------")

# --- Testing Model with Custom Inputs ---
print("\n--- Testing Model with Custom Inputs ---")

trained_model = model
trained_tokenizer = tokenizer

# Set the model's id2label mapping for human-readable output
trained_model.config.id2label = {i: label for i, label in enumerate(GOEMOTIONS_LABELS)}
trained_model.config.label2id = {label: i for i, label in enumerate(GOEMOTIONS_LABELS)}

# Define the prediction function here, as it was missing from the original snippet
def predict_emotions(text, model, tokenizer, labels, threshold=0.5):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=256)
    inputs = {k: v.to(model.device) for k, v in inputs.items()} # Move inputs to model device

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probabilities = torch.sigmoid(logits).cpu().numpy()[0] # Move to CPU and convert to numpy

    # Create a dictionary of label: probability
    prob_dict = {labels[i]: prob for i, prob in enumerate(probabilities)}

    # Filter based on threshold
    predicted_labels = [label for i, label in enumerate(labels) if probabilities[i] >= threshold]

    return predicted_labels, prob_dict

if trained_model and trained_tokenizer:
    model_labels = list(trained_model.config.id2label.values())

    text1 = "I am so incredibly happy with this result, it truly fills me with joy and a sense of pride!"
    predicted_labels1, probs1 = predict_emotions(text1, trained_model, trained_tokenizer, model_labels, threshold=optimal_threshold_macro_f1)
    print("\n--- Test Scenario 1 ---")
    print(f"Text: '{text1}'")
    print(f"Predicted Emotions: {predicted_labels1}")
    sorted_probs1 = sorted(probs1.items(), key=lambda item: item[1], reverse=True)
    print(f"Top Probabilities: {sorted_probs1[:5]}")

    text2 = "The sky is blue today. It's a typical Tuesday afternoon."
    predicted_labels2, probs2 = predict_emotions(text2, trained_model, trained_tokenizer, model_labels, threshold=optimal_threshold_macro_f1)
    print("\n--- Test Scenario 2 ---")
    print(f"Text: '{text2}'")
    print(f"Predicted Emotions: {predicted_labels2}")
    sorted_probs2 = sorted(probs2.items(), key=lambda item: item[1], reverse=True)
    print(f"Top Probabilities: {sorted_probs2[:5]}")

    text3 = "I'm quite annoyed by the constant noise, but also a bit sad it has come to this."
    predicted_labels3, probs3 = predict_emotions(text3, trained_model, trained_tokenizer, model_labels, threshold=optimal_threshold_macro_f1)
    print("\n--- Test Scenario 3 ---")
    print(f"Text: '{text3}'")
    print(f"Predicted Emotions: {predicted_labels3}")
    sorted_probs3 = sorted(probs3.items(), key=lambda item: item[1], reverse=True)
    print(f"Top Probabilities: {sorted_probs3[:5]}")
else:
    print("\nSkipping prediction examples as model/tokenizer are not yet ready.")

print("\n--- Model Training, Evaluation, and Custom Prediction Flow Complete ---")
print("--------------------------------------------------------------------------------")
