import unittest
from main import run_psychextract

class TestIntegration(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    # Run the full pipeline once for all tests
    cls.result = run_psychextract("C:\\Users\\carli\\OneDrive\\UoL\\FP\\Deliverables\\PsychExtract\\data\\OCR\\raw_handwritten\\text0_a.png")

  def test_ocr(self):
    ocr_detected_text = self.result[0]
    print("OCR Detected text:", ocr_detected_text)
    self.assertIsInstance(ocr_detected_text, str)
  
  def test_emotion_prediction(self):
    emotions = self.result[1]
    print("Detected emotions:", emotions)
    self.assertIsInstance(emotions, dict)
    self.assertIn("joy", emotions)
    self.assertIsInstance(emotions["joy"], float)

  def test_keyword_extraction(self):
    keywords = self.result[2]
    print("Detected keywords:", keywords)
    self.assertIsInstance(keywords, list)
    self.assertGreater(len(keywords), 0)

  def test_insight_generation(self):
    insight_sentences = self.result[3]
    print("Insight sentences:", insight_sentences)
    self.assertIsInstance(insight_sentences, str)
    self.assertGreater(len(insight_sentences), 0)