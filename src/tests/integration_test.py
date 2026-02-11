import unittest
from main import run_psychextract

class TestIntegration(unittest.TestCase):
  def setUp(self):
    self.result = run_psychextract("C:\\Users\\carli\\OneDrive\\UoL\\FP\\Deliverables\\PsychExtract\\data\\OCR\\raw_handwritten\\text0_a.png")

  def test_ocr(self):
    ocr_detected_text = self.result[0]
    print(ocr_detected_text)
    self.assertIsInstance(ocr_detected_text, str)
  
  def test_emotion_prediction(self):
    emotions = self.result[1]
    print(emotions)
    self.assertIsInstance(emotions, dict)
    self.assertIn("joy", emotions)
    self.assertIsInstance(emotions["joy"], float)