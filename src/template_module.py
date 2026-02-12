import numpy as np
import random

def format_list_into_string(words: list[str]) -> str:
  """
  Formats a list of keywords into a human-readable string for theme description.
  
  :param keywords: A list of keywords representing themes in the text.
  :param type: list[str]
  :return: A formatted string describing the themes.
  :rtype: str
  """
  if not words:
    return "described experience"
  if len(words) == 1:
    return words[0]
  if len(words) == 2: 
    return " and ".join(words)
  return ", ".join(words[:-1]) + ", and " + words[-1]

# lightweight linguistic detectors
def get_uncertainty_phrases() -> list[str]:
  """
  Provides a list of phrases that indicate 
  uncertainty or difficulty in articulating feelings.
  
  :return: A list of phrases commonly used to express 
  uncertainty or difficulty in explaining emotions.
  :rtype: list[str]
  """
  return [
  "not sure", "hard to name", "can't explain", "cannot explain",
  "something is there", "difficult to explain"
  ]
def get_coping_verbs() -> list[str]:
  """
  Provides a list of verbs that indicate 
  coping strategies or actions taken to manage emotions.
  
  :return: A list of verbs commonly associated with 
  coping strategies or emotional management.
  :rtype: list[str]
  """
  return [
  "writing", "reflecting", "reflection", "breathing",
  "grounding", "sitting with", "slowing"
  ]
def get_somatic_terms() -> list[str]:
  """
  Provides a list of somatic terms that may indicate 
  physical sensations associated with emotions.
  
  :return: A list of somatic terms that could be used to 
  identify physical sensations related to emotional experiences.
  :rtype: list[str]
  """
  return [
  "body", "shoulders", "breathing", "tight",
  "tense", "restless", "tension"
  ]
def get_self_reflective_phrases() -> list[str]:
  """
  Provides a list of self-reflective phrases that may indicate 
  introspection or self-awareness in the text.
  
  :return: A list of self-reflective phrases that could be used to identify 
  moments of introspection or self-awareness in the text.
  :rtype: list[str]
  """  
  return [
  "i noticed", "i realized", "i caught myself",
  "pattern in my reactions", "i keep noticing"
  ]
def get_templates() -> dict[str, str]:
  """
  Provides a dictionary of templates for generating insights based on detected themes.
  
  :return: A dictionary where keys are insight categories and values are lists of template sentences that can be filled in with specific themes or keywords to generate personalized insights.
  :rtype: dict[str, str]
  """ 
  return {
    "Emotional Load": [
      "This entry suggests a relatively high emotional load, particularly in relation to {theme}.",
      "The overall tone of this entry indicates emotional heaviness connected to {theme}."
    ],
    "Emotional Clarity against Ambiguity": [
      "The feelings described here appear difficult to clearly define, especially around {theme}.",
      "This entry reflects some uncertainty or ambiguity in how emotions related to {theme} are understood."
    ],
    "Regulation and Coping Mode": [
      "This entry highlights an active attempt to regulate emotions through reflection, particularly in response to {theme}.",
      "The writer appears to be engaging in a coping process while thinking about {theme}."
    ],
    "Arousal or Restlessness Level": [
      "The language used suggests heightened internal activation or restlessness related to {theme}.",
      "This entry reflects a state of tension or agitation associated with {theme}."
    ],
    "Self-Relation and Appraisal": [
      "This entry shows reflective self-evaluation in relation to {theme}.",
      "The writer appears to be assessing their own reactions or patterns while considering {theme}."
    ]
  }


def contains_any(text: str, phrase_list: list[str]) -> bool:
  """
  Checks if any of the phrases in the list are present in the text.
  
  :param text: The text to analyze for the presence of phrases.
  :type text: str
  :param phrase_list: A list of phrases to check for in the text.
  :type phrase_list: list[str]
  :return: True if any of the phrases from the list are found in the text, False otherwise.
  :rtype: bool
  """
  text = text.lower()
  return any(p in text for p in phrase_list)

def count_any(text: str, phrase_list: list[str]) -> int:
  """
  Counts how many phrases from the list are present in the text.
  
  :param text: The text to analyze for the presence of phrases.
  :type text: str
  :param phrase_list: A list of phrases to check for in the text.
  :type phrase_list: list[str]
  :return: The count of how many phrases from the list are found in the text.
  :rtype: int
  """
  text = text.lower()
  return sum(p in text for p in phrase_list)

def detect_insights(text: str, emotions: dict[str, float]) -> list[str]:
  """
  Detects psychological insights based on the predicted emotions and the content of the text.
  
  :param emotions: A dictionary of predicted emotions with their corresponding intensity scores.
  :type emotions: dict[str, float]
  :param text: The original text extracted from the image, used for linguistic analysis to detect insights.
  :type text: str
  :return: A list of detected insights based on the emotional profile and linguistic cues in the text.
  :rtype: list[str]
  """
  insights = []
  # Emotional Load
  mean_neg = np.mean(emotions["sadness"] + emotions["fear"] + emotions['pessimism'])
  if mean_neg > 0.6:
    insights.append("Emotional Load")
  # Emotional Clarity vs Ambiguity
  uncertainty_phrases = get_uncertainty_phrases()
  if count_any(text, uncertainty_phrases) >= 1:
    insights.append("Emotional Clarity against Ambiguity")
  # Regulation & Coping
  coping_verbs = get_coping_verbs()
  if contains_any(text, coping_verbs):
    insights.append("Regulation and Coping Mode")
  # Arousal / Restlessness
  somatic_terms = get_somatic_terms()
  if (emotions["fear"] + emotions["anger"] > 0.6 or
      contains_any(text, somatic_terms)):
    insights.append("Arousal or Restlessness Level")
  # Self-Relation & Appraisal
  self_reflective_phrases = get_self_reflective_phrases()
  if contains_any(text, self_reflective_phrases):
    insights.append("Self-Relation and Appraisal")
  return insights

def format_insight_sentences(emotions: dict, insights: list[dict]) -> str:
  """
  Formats the detected insights and predicted emotions into coherent sentences based on predefined templates.
  
  :param emotions: A dictionary of predicted emotions with their corresponding intensity scores, used to determine which insights are relevant.
  :type emotions: dict
  :param insights: A list of dictionaries, each containing an insight category and a generated sentence that provides a personalized interpretation of the emotional content related to the detected themes.
  :type insights: list[dict]
  :return: A formatted string that combines the detected insight categories and the generated sentences based on the templates, providing a personalized interpretation of the emotional content related to the detected themes.
  :rtype: str
  """
  # Format the detected emotions into a coherent sentence
  emotion_list = [emotion for emotion, score in emotions.items() if score >= 0.3]
  emotions_text = format_list_into_string(emotion_list)
  emotion_themes = f"Emotions of {emotions_text} are detected. "
  # Format the detected insight categories into a coherent sentence
  if not insights[0]["category"]:
    insight_themes = "No significant themes were detected. "
    texts = ""
  else:
    categories = [item["category"] for item in insights if "category" in item]
    categories_text = format_list_into_string(categories)
    insight_themes = f"Themes of {categories_text} are detected. "
    # Collect all descriptive texts
    texts = " ".join([item["text"] for item in insights if "text" in item])
  # Combine everything
  return emotion_themes + insight_themes + texts

def generate_insight_sentences(text: str, emotions: dict[str, float], keywords: list) -> str:
  """
  Generates personalized insight sentences based on the detected insights, predicted emotions, and identified themes in the text.

  :param text: The original text extracted from the image, used for linguistic analysis to generate insights.
  :type text: str
  :param emotions: A dictionary of predicted emotions with their corresponding intensity scores, used to determine which insights are relevant. :type emotions: dict[str, float] :param keywords: A list of keywords representing themes in the text, used to personalize the insight sentences. :type keywords: list :return: A list of dictionaries, each containing an insight category and a generated sentence that provides a personalized interpretation of the emotional content related to the detected themes. :rtype: list[dict[str, str]]
  :type emotions: dict[str, float]
  :param  keywords: A list of keywords representing themes in the text, used to personalize the insight sentences.
  :type keywords: list[str]
  :return: A formatted string that combines the detected insight categories and the generated sentences based on the templates, providing a personalized interpretation of the emotional content related to the detected themes.
  :rtype: str
  """
  random.seed(42)

  keywords_text = "their " + format_list_into_string(keywords)
  categories = detect_insights(text, emotions)
  templates = get_templates()
  outputs = []
  
  if not categories:
    outputs.append({
      "category": "",
      "text": ""
    })

  for cat in categories:
    template = random.choice(templates[cat])
    outputs.append({
      "category": cat,
      "text": template.format(theme=keywords_text)
    })

  return format_insight_sentences(emotions, outputs)