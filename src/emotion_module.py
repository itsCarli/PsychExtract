# emotion_module.py - module for emotion detection using a pre-trained RoBERTa model

# import necessary libraries
import numpy as np
import torch
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification

def get_labels() -> list[str]:
  """
  Get the list of emotion labels used by the model.
  Returns:
    list[str]: A list of emotion labels.
  """
  return [
    "anger", "anticipation", "disgust", "fear",
    "joy", "love", "optimism", "pessimism",
    "sadness", "surprise", "trust"
  ]

def load_emotion_model() -> tuple[RobertaForSequenceClassification, RobertaTokenizerFast]:
  """
  Load the RoBERTa multi-label emotion classification model and tokenizer.
  Returns:
    tuple: A tuple containing the loaded emotion model and tokenizer.
  """
  tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
  # specify multi-label classification for correct loss function and output processing
  # ensure model output matches number of emotion labels
  model = RobertaForSequenceClassification.from_pretrained(
    "cardiffnlp/twitter-roberta-base-emotion-multilabel-latest", 
    problem_type="multi_label_classification", 
    num_labels=len(get_labels())
  )
  # move model to appropriate device (GPU if available) for faster inference
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model.to(device)
  # set model to evaluation mode for inference only
  model.eval()
  return model, tokenizer

def predict_emotions(
    text: str, 
    model: RobertaForSequenceClassification, 
    tokenizer: RobertaTokenizerFast
  ) -> dict:
  """
  Predict emotions from the input text using the provided model and tokenizer.
  Args:
    text (str): The input text for emotion prediction.
    model (RobertaForSequenceClassification): The pre-loaded RoBERTa model for emotion classification.
    tokenizer (RobertaTokenizerFast): The corresponding tokenizer for the model.
  Returns:
    dict: A dictionary of emotions and their corresponding probabilities.
  """ 
  # determine the device of the model for proper tensor placement
  device = next(model.parameters()).device

  # tokenize and move to device
  inputs = tokenizer(
    text, 
    return_tensors="pt", # ensure input is in batch format and tokenized correctly for the model
    truncation=True, # truncate long inputs to fit model's max length
    padding=True # pad shorter inputs to ensure consistent batch size
  ).to(device)

  # inference with no gradient calculation for efficiency due to prediction-only mode
  # apply sigmoid to logits to get probabilities for each emotion label in multi-label classification
  with torch.no_grad():
    logits = model(**inputs).logits
    probs = torch.sigmoid(logits)[0]

  # create a dictionary mapping each emotion label to its predicted probability
  labels = get_labels()
  detected = {
    label: float(prob) 
    for label, prob in zip(labels, probs)
  }
  return detected