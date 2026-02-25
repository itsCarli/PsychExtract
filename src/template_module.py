# template_module.py - module for generating psychological insights 
# based on detected themes and predicted emotions, using predefined templates for personalized insight generation

# import necessary libraries
import numpy as np
import random

def format_list_into_string(words: list[str]) -> str:
  """
  Formats a list of keywords into a human-readable string for theme description.
  Args:
    keywords (list[str]): A list of keywords representing themes in the text.
  Returns:
    str: A formatted string describing the themes.
  """
  if not words:
    # if no keywords, return a generic phrase
    return "described experience"
  if len(words) == 1:
    # if only one keyword, return it directly
    return words[0]
  if len(words) == 2: 
    # if two keywords, join with "and"
    return " and ".join(words)
  # if more than two keywords, join with commas and "and" before the last one
  return ", ".join(words[:-1]) + ", and " + words[-1]

def get_uncertainty_phrases() -> list[str]:
  """
  Provides a list of phrases that indicate 
  uncertainty or difficulty in articulating feelings.
  Returns:
    list[str]: A list of phrases commonly used to express
  """
  return [
  "not sure", "hard to name", "can't explain", "cannot explain",
  "something is there", "difficult to explain"
  ]

def get_coping_verbs() -> list[str]:
  """
  Provides a list of verbs that indicate 
  coping strategies or actions taken to manage emotions.
  Returns:
    list[str]: A list of verbs commonly associated with coping strategies or emotional management.
  """
  return [
  "writing", "reflecting", "reflection", "breathing",
  "grounding", "sitting with", "slowing"
  ]

def get_somatic_terms() -> list[str]:
  """
  Provides a list of somatic terms that may indicate 
  physical sensations associated with emotions.
  Returns:
    list[str]: A list of somatic terms that could be used to identify physical sensations related to emotions in the text.
  """
  return [
  "body", "shoulders", "breathing", "tight",
  "tense", "restless", "tension"
  ]

def get_self_reflective_phrases() -> list[str]:
  """
  Provides a list of self-reflective phrases that may indicate 
  introspection or self-awareness in the text.
  Returns:
    list[str]: A list of self-reflective phrases that could be used to identify introspection or self-awareness in the text.
  """  
  return [
  "i noticed", "i realized", "i caught myself",
  "pattern in my reactions", "i keep noticing"
  ]

def get_templates() -> dict[str, str]:
  """
  Provides a dictionary of templates for generating insights based on detected themes.
  Returns:
    dict: A dictionary where keys are insight categories and values are lists of template 
    sentences that can be filled in with specific themes or keywords to generate personalized insights.
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
  Args:
    text (str): The text to analyze for the presence of phrases.
    phrase_list (list[str]): A list of phrases to check for in the text.
  Returns:
    bool: True if any of the phrases from the list are found in the text, False otherwise.
  """
  text = text.lower()
  return any(p in text for p in phrase_list)

def count_any(text: str, phrase_list: list[str]) -> int:
  """
  Counts how many phrases from the list are present in the text.
  Args:
    text (str): The text to analyze for the presence of phrases.
    phrase_list (list[str]): A list of phrases to check for in the text.
  Returns:
    int: The count of how many phrases from the list are found in the text.
  """
  text = text.lower()
  return sum(p in text for p in phrase_list)

def detect_insights(text: str, emotions: dict[str, float]) -> list[str]:
  """
  Detects psychological insights based on the predicted emotions and the content of the text.
  Args:
    emotions (dict[str, float]): A dictionary of predicted emotions with their corresponding intensity scores, used to determine which insights are relevant.
    text (str): The original text extracted from the image, used for linguistic analysis to detect insights.
  Returns:
    list[str]: A list of detected insights based on the emotional profile and linguistic cues in the text.
  """
  insights = []
  # detect emotional load 
  # based on intensity of negative emotions
  mean_neg = np.mean(emotions["sadness"] + emotions["fear"] + emotions["pessimism"])
  if mean_neg > 0.6:
    insights.append("Emotional Load")
  # detect emotional clarity vs ambiguity 
  # based on presence of uncertainty phrases and intensity of emotions
  uncertainty_phrases = get_uncertainty_phrases()
  if count_any(text, uncertainty_phrases) >= 1:
    insights.append("Emotional Clarity against Ambiguity")
  # detect regulation and coping mode 
  # based on presence of coping verbs and intensity of emotions
  coping_verbs = get_coping_verbs()
  if contains_any(text, coping_verbs):
    insights.append("Regulation and Coping Mode")
  # detect arousal or restlessness level
  # based on intensity of high-arousal emotions and presence of somatic terms
  somatic_terms = get_somatic_terms()
  if (emotions["fear"] + emotions["anger"] > 0.6 or
      contains_any(text, somatic_terms)):
    insights.append("Arousal or Restlessness Level")
  # detect self-relation and appraisal
  # based on presence of self-reflective phrases
  self_reflective_phrases = get_self_reflective_phrases()
  if contains_any(text, self_reflective_phrases):
    insights.append("Self-Relation and Appraisal")
  return insights

def format_insight_sentences(emotions: dict, insights: list[dict], emotion_threshold: float = 0.3) -> str:
  """
  Formats the generated insight sentences into a coherent summary that combines the detected insight 
  categories and the predicted emotions.
  Args:
    emotions (dict): A dictionary of predicted emotions with their corresponding intensity scores, used to determine which insights are relevant.
    insights (list[dict]): A list of dictionaries, each containing an insight category and a generated sentence that provides a personalized interpretation of the emotional content related to the detected themes.
    emotion_threshold (float): A threshold value to filter which emotions are considered significant for generating insights. Emotions with intensity scores above this threshold will be included in the summary.
  Returns:
    str: A formatted string that combines the detected insight categories and the generated sentences based on the templates, providing a personalized interpretation of the emotional content related to the detected themes.
  """
  # format the detected emotions into a coherent sentence
  emotion_list = [emotion for emotion, score in emotions.items() if score >= emotion_threshold]
  emotions_text = format_list_into_string(emotion_list)
  emotion_themes = f"Emotions of {emotions_text} are detected. "
  texts = ""
  insight_themes = ""
  if not insights[0]["category"]:
    # if no insights detected, return a generic phrase
    insight_themes = "No significant themes were detected. "
  else:
    # format the detected insight categories into a coherent sentence
    categories = [item["category"] for item in insights if "category" in item]
    categories_text = format_list_into_string(categories)
    insight_themes = f"Themes of {categories_text} are detected. "
    # collect all descriptive texts
    texts = " ".join([item["text"] for item in insights if "text" in item])
  # combine the emotion themes, insight themes, and descriptive texts into a final summary
  return emotion_themes + insight_themes + texts

def generate_insight_sentences(text: str, emotions: dict[str, float], keywords: list, emotion_threshold: float = 0.3) -> str:
  """
  Generates personalized insight sentences based on the detected themes in the text and the predicted emotions, 
  using predefined templates for each insight category.
  Args:
    text (str): The original text extracted from the image, used for linguistic analysis to generate insights.
    emotions (dict[str, float]): A dictionary of predicted emotions with their corresponding intensity scores, used to determine which insights are relevant.
    keywords (list): A list of keywords representing themes in the text, used to personalize the insight sentences.
    emotion_threshold (float): A threshold value to filter which emotions are considered significant for generating insights
  Returns:
    str: A formatted string that combines the detected insight categories and the generated sentences based on the templates, providing a personalized interpretation of the emotional content related to the detected themes.
  """
  # set random seed for reproducibility of template selection
  random.seed(42)
  # format the keywords into a human-readable string for theme description
  keywords_text = "their " + format_list_into_string(keywords)
  # detect insights based on the predicted emotions and the content of the text
  categories = detect_insights(text, emotions)
  # get the templates for generating insights based on detected themes
  templates = get_templates()
  outputs = []
  if not categories:
    # if no insights detected, add a generic entry to maintain consistent output structure
    outputs.append({
      "category": "",
      "text": ""
    })
  for cat in categories:
    # select a random template for each detected category and fill in the theme description
    template = random.choice(templates[cat])
    outputs.append({
      "category": cat,
      "text": template.format(theme=keywords_text)
    })
  return format_insight_sentences(emotions, outputs, emotion_threshold)