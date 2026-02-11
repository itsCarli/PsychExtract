import unittest
from main import run_psychextract

class TestIntegration(unittest.TestCase):
  def test_ocr(self):
    print('running OCR')
    result = run_psychextract("C:\\Users\\carli\\OneDrive\\UoL\\FP\\Deliverables\\PsychExtract\\data\\OCR\\raw_handwritten\\text0_a.png")
    print(result)
    self.assertIsInstance(result, str)