import unittest
from tts_module import speak

class TestTTSModule(unittest.TestCase):
  def test_tts(self):
    res = speak("Test")
    print(res)
    self.assertIsInstance(res, str)