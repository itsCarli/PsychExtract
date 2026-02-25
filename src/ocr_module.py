# ocr_module.py - module for OCR processing using Qwen-VL model, including image preprocessing and text extraction functions

# import necessary libraries
import numpy as np
from pathlib import Path
from PIL import Image, ImageOps
import cv2
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

def preprocess_image(img_path: str, upscale=2.0) -> Image.Image:
  """
  Preprocess a single image by applying various image processing techniques.
  Args:
    img_path (str): The file path to the input image.
    upscale (float): The factor by which to upscale the image for better OCR performance. Default is 2.0.
  Returns:
    Image.Image: The preprocessed image as a PIL Image object.
  """
  # ensure path is in correct format for PIL
  img_path = Path(img_path)
  # load image with PIL to preserve EXIF data for orientation correction
  im = Image.open(img_path) 
  # handle orientation based on EXIF data, important for mobile photos
  im = ImageOps.exif_transpose(im) 
  # ensure 3-channel RGB for consistent processing
  im = im.convert("RGB") 
  # convert PIL to OpenCV format (BGR) for processing
  img = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR) 
  # grayscale conversion
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
  # contrast enhancement (CLAHE)
  # adjust clipLimit and tileGridSize for better results on handwritten text
  clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(16, 16)) 
  gray = clahe.apply(gray) 
  # adaptive thresholding to binarize the image, helps with OCR accuracy
  bw = cv2.adaptiveThreshold(
    gray,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    31,
    51
  )
  # optional upscaling to improve OCR performance, especially for small or low-res text
  if upscale > 1:
    bw = cv2.resize(
      bw, 
      None, 
      fx=upscale, 
      fy=upscale,
      interpolation=cv2.INTER_CUBIC
    )
  # convert back to PIL format for compatibility with Qwen-VL processor
  return Image.fromarray(bw)
    
def load_qwen() -> tuple[Qwen2_5_VLForConditionalGeneration, AutoProcessor]:
  """
  Load the Qwen-VL model and processor for OCR tasks.
  Returns:
    tuple: A tuple containing the loaded Qwen-VL model and processor.
  """
  qwen_model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
  processor = AutoProcessor.from_pretrained(qwen_model_id)
  # load the model with half precision and automatic device mapping for efficiency
  qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    qwen_model_id,
    torch_dtype=torch.float16,
    device_map="auto"
  )
  return qwen_model, processor

def extract_text_from_image(image: Image.Image,
                            qwen_model: Qwen2_5_VLForConditionalGeneration,
                            qwen_tokenizer: AutoProcessor) -> str:
  """
  Perform OCR on a image path using Qwen-VL and return the string.
  Args:
    image (Image.Image): A PIL Image object to be processed.
    qwen_model (Qwen2_5_VLForConditionalGeneration): An instance of the Qwen-VL model.
    qwen_tokenizer (AutoProcessor): An instance of the Qwen-VL processor.
  Returns:
    str: The extracted text from the image.
  """
  try:
    # define a clear prompt to instruct the model 
    prompt = (
      "Transcribe the handwritten text exactly as it appears. "
      "Output ONLY the transcription."
      "No explanations or role labels."
      "Do not correct spelling, grammar, or punctuation."
    )
    # state the content types clearly for Qwen to understand the input structure
    messages = [{
      "role": "user",
      "content": [
        {"type": "image"},
        {"type": "text", "text": prompt}
      ]
    }]
    # text_input is the formatted prompt for the model
    text_input = qwen_tokenizer.apply_chat_template(
      messages,
      add_generation_prompt=True
    )
    # inputs contains both image and text data in the expected format
    inputs = (qwen_tokenizer(text=text_input, images=image, return_tensors="pt")
                            .to(qwen_model.device))
    # generate the output from the model, specifying a max token limit to control response length
    output = qwen_model.generate(**inputs, max_new_tokens=512)
    # decode the output tokens to get the raw text response from the model
    raw_text = qwen_tokenizer.batch_decode(output, skip_special_tokens=True)[0]
    # remove any role labels or extraneous text, keeping only the transcribed text
    cleaned_text = raw_text.split("assistant")[-1].strip()
    return cleaned_text
  except Exception as e:
    # catch and print any errors during OCR processing, returning None to indicate failure
    print(f"Error processing: {e}\n")
  return None

def preprocess_image_and_extract_text(image_path: str, model: Qwen2_5_VLForConditionalGeneration, processor: AutoProcessor) -> str:
  """
  Preprocess the input image and extract text using the Qwen-VL model.
  Args: 
    image_path (str): The file path to the input image.
    model (Qwen2_5_VLForConditionalGeneration): An instance of the Qwen-VL model.
    processor (AutoProcessor): An instance of the Qwen-VL processor.
  Returns:
    str: The extracted text from the image.
  """
  preprocessed_image = preprocess_image(image_path)
  return extract_text_from_image(preprocessed_image, model, processor)