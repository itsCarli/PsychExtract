import numpy as np

from pathlib import Path
from PIL import Image, ImageOps
import cv2
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

def preprocess_image(img_path: str, upscale=2.0) -> None:
  """
  Preprocess a single image by applying various image processing techniques.

  :param img_path: Path to the input image file.
  :type img_path: str
  :param upscale: Factor by which to upscale the image.
  :type upscale: float
  """
  img_path = Path(img_path)

  # load with PIL (format-agnostic)
  im = Image.open(img_path)
  # fix EXIF orientation
  im = ImageOps.exif_transpose(im)
  # convert to RGB
  im = im.convert("RGB")
  # convert to OpenCV format
  img = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
  # apply grayscale ----
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  # contrast enhancement (CLAHE)
  clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(16, 16))
  gray = clahe.apply(gray)
  # adaptive threshold
  bw = cv2.adaptiveThreshold(
    gray,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    31,
    51)
  # upscale
  if upscale > 1:
    bw = cv2.resize(bw, None, fx=upscale, fy=upscale,
                    interpolation=cv2.INTER_CUBIC)
    
  # Convert back to PIL Image
  return Image.fromarray(bw)
    
def load_qwen():
  qwen_model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
  processor = AutoProcessor.from_pretrained(qwen_model_id)
  qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(qwen_model_id)
  return qwen_model, processor

def extract_text_from_image(image: Image.Image,
                            qwen_tokenizer: AutoProcessor,
                            qwen_model: Qwen2_5_VLForConditionalGeneration) -> str:
  """
  Perform OCR on a image path using Qwen-VL and return the string.

  :param image: A PIL Image object to be processed.
  :type image: Image.Image
  :param qwen_processor: An instance of the Qwen-VL processor.
  :type qwen_processor: AutoProcessor
  :param qwen_model: An instance of the Qwen-VL model.
  :type qwen_model: AutoModelForVision2Seq
  :return: The extracted text from the image.
  :rtype: str
  """

  try:
    prompt = (
      "Transcribe the handwritten text exactly as it appears. "
      "Output ONLY the transcription."
      "No explanations or role labels."
      "Do not correct spelling, grammar, or punctuation."
    )
    messages = [{
      "role": "user",
      "content": [
        {"type": "image"},
        {"type": "text", "text": prompt}
      ]
    }]
    text_input = qwen_tokenizer.apply_chat_template(
      messages,
      add_generation_prompt=True
    )
    inputs = (qwen_tokenizer(text=text_input, images=image, return_tensors="pt")
                            .to(qwen_model.device))
    output = qwen_model.generate(**inputs, max_new_tokens=512)
    raw_text = qwen_tokenizer.batch_decode(output, skip_special_tokens=True)[0]
    # Remove role headers if present
    cleaned_text = raw_text.split("assistant")[-1].strip()
    return cleaned_text
  except Exception as e:
    print(f"Error processing: {e}\n")
  return None

def load_preprocess_and_extract(image_path: str):
  preprocessed_image = preprocess_image(image_path)
  return extract_text_from_image(preprocessed_image, processor, qwen_model)

if __name__ != "__main__":
  # Load Qwen model and processor once at module level
  qwen_model, processor = load_qwen()