# ocr_test.py - unit tests for the OCR module of PsychExtract, testing individual functions and components related to image preprocessing, model loading, and text extraction from images.

# import necessary libraries and modules for testing
import unittest
from PIL import Image
from ocr_module import preprocess_image, load_qwen, extract_text_from_image

class TestOCRModule(unittest.TestCase):
  """Unit tests for the OCR module of PsychExtract, testing individual functions and components related to image preprocessing, model loading, and text extraction from images."""
  def test_preprocess_image(self):
    """
    Test the preprocess_image function to ensure that it correctly processes an input image and returns a PIL Image object.\n
    This test checks that the function can successfully read and preprocess an image file, returning an object of the expected type without raising any exceptions.
    """
    try:
      img = preprocess_image("C:\\Users\\carli\\OneDrive\\UoL\\FP\\Deliverables\\PsychExtract\\data\\OCR\\raw_handwritten\\text0_a.png")
      self.assertIsInstance(img, Image.Image)
    except Exception as e:
      self.fail(f"preprocess_image raised an exception: {e}")

  def test_load_qwen_model(self):
    """
    Test the load_qwen function to ensure that it successfully loads the Qwen-VL model and processor without raising any exceptions.\n
    This test checks that the function can access the necessary resources to load the model and processor, returning the expected objects without errors.
    """
    try:
      _, _ = load_qwen()
    except Exception as e:
      self.fail(f"load_qwen_model raised an exception: {e}")

  def test_extract_text_from_image(self):
    """
    Test the extract_text_from_image function to verify that it can successfully extract text from a preprocessed image using the Qwen-VL model.\n
    This test checks that the function can process the image and return a string of extracted text without raising any exceptions, ensuring that the OCR functionality is working as intended.
    """
    img = preprocess_image("C:\\Users\\carli\\OneDrive\\UoL\\FP\\Deliverables\\PsychExtract\\data\\OCR\\raw_handwritten\\text0_a.png")
    print("preprocessed image")
    qwen_model, processor = load_qwen()
    print("loaded model and processor")
    result = extract_text_from_image(img, processor, qwen_model)
    print(result)
    self.assertIsInstance(result, str)