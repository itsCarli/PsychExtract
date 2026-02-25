# tts_test.py - Unit tests for the text-to-speech module of PsychExtract, testing individual functions and components related to text-to-speech synthesis and output generation.

# import necessary libraries and modules for testing
import unittest
from tts_module import speak

class TestTTSModule(unittest.TestCase):
  """Unit tests for the text-to-speech module of PsychExtract, testing individual functions and components related to text-to-speech synthesis and output generation."""
  def test_tts(self):
    """
    Test the speak function to ensure that it generates a string output from the input text.\n
    This test checks that the function can take a string input and produce a string output, which is expected to be the synthesized speech in text form.\n
    The test verifies that the output is of the correct type and prints the result for verification."""
    res = speak("Test")
    print(res)
    self.assertIsInstance(res, str)