import unittest
from emotion_module import load_emotion_model, predict_emotions

class TestEmotionModule(unittest.TestCase):
  def setUp(self):
    self.model, self.tokenizer = load_emotion_model()

  def test_predict_emotions(self):
    text = "I am so happy and excited!"
    result = predict_emotions(text, self.model, self.tokenizer)
    print(result)
    self.assertIsInstance(result, dict)
    self.assertIn("joy", result)
    self.assertIsInstance(result["joy"], float)