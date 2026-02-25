# emotions_test.py - unit tests for the emotion_module functions, including loading the emotion model and predicting emotions from text input

# import neccesary libraries and modules for testing
import unittest
from emotion_module import load_emotion_model, predict_emotions

class TestEmotionModule(unittest.TestCase):
  """Unit tests for the emotion_module functions."""
  def setUp(self):
    """
    Set up the test case by loading the emotion model and tokenizer.\n
    This method is called before each test method to ensure that the model and tokenizer are available for testing.
    """
    self.model, self.tokenizer = load_emotion_model()

  def test_predict_emotions(self):
    """
    Test the predict_emotions function with a sample input text to verify that it returns a dictionary of emotions and probabilities.\n
    This test checks that the output is a dictionary, contains the expected emotion labels, and that the probabilities are of the correct type.
    """
    text = "I am so happy and excited!"
    result = predict_emotions(text, self.model, self.tokenizer)
    print(result)
    self.assertIsInstance(result, dict)
    self.assertIn("joy", result)
    self.assertIsInstance(result["joy"], float)