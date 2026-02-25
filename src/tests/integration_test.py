# integration_test.py - integration tests for the full PsychExtract pipeline, testing the end-to-end functionality of OCR, emotion

# import necessary libraries and modules for testing
import unittest
from main import run_psychextract

class TestIntegration(unittest.TestCase):
  """Integration tests for the full PsychExtract pipeline, testing the end-to-end functionality of OCR, emotion prediction, keyword extraction, insight generation, and text-to-speech synthesis."""
  @classmethod
  def setUpClass(cls):
    """
    Set up the test case by running the full PsychExtract pipeline once for all tests.\n
    This method is called once before all test methods to execute the pipeline and store the results for subsequent tests.
    """
    cls.result = run_psychextract("example_io\\text1_a.png", "example_io\\output.wav")

  def test_ocr(self):
    """
    Test the OCR output from the PsychExtract pipeline to ensure that text is extracted from the input image.\n
    This test checks that the OCR output is a string and prints the detected text for verification.
    """
    ocr_detected_text = self.result[0]
    print("OCR Detected text:", ocr_detected_text)
    self.assertIsInstance(ocr_detected_text, str)
  
  def test_emotion_prediction(self):
    """
    Test the emotion prediction output from the PsychExtract pipeline to verify that emotions are detected from the extracted text.\n
    This test checks that the emotion prediction output is a dictionary, contains expected emotion labels, and that the probabilities are of the correct type.
    """
    emotions = self.result[1]
    print("Detected emotions:", emotions)
    self.assertIsInstance(emotions, dict)
    self.assertIn("joy", emotions)
    self.assertIsInstance(emotions["joy"], float)

  def test_keyword_extraction(self):
    """
    Test the keyword extraction output from the PsychExtract pipeline to ensure that keywords are extracted from the text.\n
    This test checks that the keyword extraction output is a list and contains at least one keyword, while also printing the detected keywords for verification.
    """
    keywords = self.result[2]
    print("Detected keywords:", keywords)
    self.assertIsInstance(keywords, list)
    self.assertGreater(len(keywords), 0)

  def test_insight_generation(self):
    """
    Test the insight generation output from the PsychExtract pipeline to verify that insight sentences are generated based on the extracted text, detected emotions, and keywords.\n
    This test checks that the insight generation output is a non-empty string and prints the generated insight sentences for verification.
    """
    insight_sentences = self.result[3]
    print("Insight sentences:", insight_sentences)
    self.assertIsInstance(insight_sentences, str)
    self.assertGreater(len(insight_sentences), 0)
  
  def test_tts(self):
    """
    Test the text-to-speech output from the PsychExtract pipeline to ensure that audio is generated from the insight sentences.\n
    This test checks that the TTS output is not None and prints the TTS result for verification.
    """
    tts_result = self.result[4]
    print(f"TTS result: {tts_result}")
    self.assertIsNotNone(tts_result)