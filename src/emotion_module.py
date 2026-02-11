import numpy as np
import torch
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification

def get_labels() -> list:
  """
  Get the list of emotion labels used by the model.

  :return: A list of emotion labels.
  :rtype: list
  """
  return [
    "anger", "anticipation", "disgust", "fear",
    "joy", "love", "optimism", "pessimism",
    "sadness", "surprise", "trust"
  ]

def load_emotion_model() -> tuple:
  """
  Load the RoBERTa multi-label emotion classification model and tokenizer.
  
  :return: A tuple containing the loaded model and tokenizer.
  :rtype: tuple
  """
  tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
  model = RobertaForSequenceClassification.from_pretrained(
    "cardiffnlp/twitter-roberta-base-emotion-multilabel-latest", 
    problem_type="multi_label_classification", 
    num_labels=len(get_labels())
  )
  return model, tokenizer

def predict_emotions(text: str, 
                     model: RobertaForSequenceClassification, 
                     tokenizer: RobertaTokenizerFast, 
                     threshold=0.5) -> dict:
  """
  Predict emotions from the input text using the provided model and tokenizer.

  :param text: The input text for emotion prediction.
  :type text: str
  :param model: The pre-loaded RoBERTa model for emotion classification.
  :type model: RobertaForSequenceClassification
  :param tokenizer: The corresponding tokenizer for the model.
  :type tokenizer: RobertaTokenizerFast
  :param threshold: The probability threshold for considering an emotion as detected.
  :type threshold: float
  :return: A dictionary of detected emotions and their corresponding probabilities.
  :rtype: dict
  """
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model.to(device)
  model.eval()

  # Tokenize and move to device
  inputs = tokenizer(text, 
                     return_tensors="pt", 
                     truncation=True, 
                     padding=True).to(device)

  # Inference
  with torch.no_grad():
    logits = model(**inputs).logits
    probs = torch.sigmoid(logits)[0]  # shape: [num_labels]

  labels = get_labels()
  # Only keep emotions above threshold
  detected = {label: float(prob) 
              for label, prob in zip(labels, probs) 
              if prob >= threshold}

  # If no emotions meet the threshold, return the one with the highest probability
  if not detected:
    max_idx = torch.argmax(probs).item()
    detected = {labels[max_idx]: float(probs[max_idx])}
  return detected

def load_and_predict_emotions(text: str) -> dict:
  """
  Load the emotion model and predict emotions from the input text.

  :param text: The input text for emotion prediction.
  :type image_path: str
  :return: A dictionary of detected emotions and their probabilities.
  :rtype: dict
  """
  # Load emotion model and tokenizer
  model, tokenizer = load_emotion_model()
  # Predict emotions from the extracted text
  emotions = predict_emotions(text, model, tokenizer)
  
  return emotions