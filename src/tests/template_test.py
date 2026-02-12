import unittest
from template_module import (
    format_list_into_string, 
    detect_insights, 
    format_insight_sentences,
    generate_insight_sentences
  )

class TestTemplateModule(unittest.TestCase):
  def format_list_into_string(self): 
    keywords = ["family", "stress", "work"]
    formatted = format_list_into_string(keywords) 
    self.assertEqual(formatted, "family, stress, and work") 

  def test_detect_insights(self): 
    text = "I feel so sad and scared. I don't know how to cope with this." 
    emotions = { "sadness": 0.8, "fear": 0.7, "pessimism": 0.5, "joy": 0.1, "anger": 0.2 } 
    insights = detect_insights(text, emotions) 
    self.assertIsInstance(insights, list)

  def test_format_insight_sentences(self):
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
    text = "I feel so sad and scared. I don't know how to cope with this." 
    emotions = { "sadness": 0.8, "fear": 0.7, "pessimism": 0.5, "joy": 0.1, "anger": 0.2 } 
    keywords = ["family", "stress", "work"] 
    outputs = generate_insight_sentences(text, emotions, keywords) 
    self.assertIsInstance(outputs, str)

