# template_test.py - unit tests for the template module of PsychExtract, testing individual functions and components related to formatting lists, detecting insights, and generating insight sentences.

# import necessary libraries and modules for testing
import unittest
from template_module import (
  format_list_into_string, 
  detect_insights, 
  format_insight_sentences,
  generate_insight_sentences
)

class TestTemplateModule(unittest.TestCase):
  """Unit tests for the template module of PsychExtract, testing individual functions and components related to formatting lists, detecting insights, and generating insight sentences."""
  def format_list_into_string(self): 
    """
    Test the format_list_into_string function with a list of keywords to verify that it correctly formats the list into a readable string.\n
    This test checks that the function can take a list of keywords and format it into a string with proper punctuation and conjunctions, ensuring that the output is grammatically correct and easy to read.
    """
    keywords = ["family", "stress", "work"]
    formatted = format_list_into_string(keywords) 
    self.assertEqual(formatted, "family, stress, and work") 

  def test_detect_insights(self): 
    """
    Test the detect_insights function with sample text and emotion data to verify that it returns a list of insights based on the input.\n
    This test checks that the function can analyze the provided text and emotion data to generate insights, ensuring that the output is a list of insights that are relevant to the input data.
    """
    text = "I feel so sad and scared. I don't know how to cope with this." 
    emotions = { "sadness": 0.8, "fear": 0.7, "pessimism": 0.5, "joy": 0.1, "anger": 0.2 } 
    insights = detect_insights(text, emotions) 
    self.assertIsInstance(insights, list)

  def test_format_insight_sentences(self):
    """
    Test the format_insight_sentences function with sample emotion data and insights to verify that it correctly formats the insights into a readable string.\n
    This test checks that the function can take a dictionary of emotions and a list of insights, and format them into a coherent string that effectively communicates the insights based on the emotional analysis.
    """
    emotions = { "sadness": 0.8, "fear": 0.7, "pessimism": 0.5, "joy": 0.1, "anger": 0.2 }
    insights = [
      {"category": "Emotional Load", "text": "The text shows a high emotional load."},
      {"category": "Regulation and Coping Mode", "text": "There are indications of coping strategies."}
    ]
    formatted = format_insight_sentences(emotions, insights)
    self.assertIsInstance(formatted, str)
    self.assertIn("Emotional Load", formatted)
    self.assertIn("Regulation and Coping Mode", formatted)
    
  def test_generate_insight_sentences(self): 
    """
    Test the generate_insight_sentences function with sample text, emotion data, and keywords to verify that it generates insight sentences based on the input.\n
    This test checks that the function can take the provided text, emotion data, and keywords to generate meaningful insight sentences that reflect the emotional state and key themes present in the input data.
    """
    text = "I feel so sad and scared. I don't know how to cope with this." 
    emotions = { "sadness": 0.8, "fear": 0.7, "pessimism": 0.5, "joy": 0.1, "anger": 0.2 } 
    keywords = ["family", "stress", "work"] 
    outputs = generate_insight_sentences(text, emotions, keywords) 
    self.assertIsInstance(outputs, str)

