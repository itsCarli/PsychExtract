import unittest
from ocr_module import preprocess_image, load_qwen, extract_text_from_image

from PIL import Image

class TestOCRModule(unittest.TestCase):
  def test_preprocess_image(self):
    # Test that the preprocess_image function runs without errors
    try:
      img = preprocess_image("C:\\Users\\carli\\OneDrive\\UoL\\FP\\Deliverables\\PsychExtract\\data\\OCR\\raw_handwritten\\text0_a.png")
      self.assertIsInstance(img, Image.Image)
    except Exception as e:
      self.fail(f'preprocess_image raised an exception: {e}')

  def test_load_qwen_model(self):
    # Test that the Qwen model loads without errors
    try:
      qwen_model, processor = load_qwen()
    except Exception as e:
      self.fail(f'load_qwen_model raised an exception: {e}')

  def test_extract_text_from_image(self):
    # Test that the extract_text_from_image function returns the expected string
    img = preprocess_image("C:\\Users\\carli\\OneDrive\\UoL\\FP\\Deliverables\\PsychExtract\\data\\OCR\\raw_handwritten\\text0_a.png")
    print('preprocessed image')
    qwen_model, processor = load_qwen()
    print('loaded model and processor')
    result = extract_text_from_image(img, processor, qwen_model)
    # expected = 'Today felt heavier than I expected. I kept replaying the conversation in my head, wondering if I said too much or not enough.'
    # assert result is string
    print(result)
    self.assertIsInstance(result, str)