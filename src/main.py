# main.py - main entry point for the PsychExtract prototype, orchestrating the full pipeline from OCR to TTS

# python.exe -m pip install --upgrade pip
# pip install numpy transformers torch accelerate torchvision pillow opencv-python yake spacy pyttsx3 streamlit gTTS
# python -m spacy download en_core_web_sm

# modules
from ocr_module import preprocess_image_and_extract_text
from emotion_module import predict_emotions
from keyword_module import extract_and_select_keywords
from template_module import generate_insight_sentences
from tts_module import pyttsx3_speak

def run_psychextract(image_path: str, output_path: str = None):
  """
  Run the PsychExtract pipeline on a given image path, performing OCR, emotion prediction, keyword extraction, insight generation, and optional text-to-speech synthesis.
  Args:
    image_path (str): The file path to the input image containing handwritten text.
    output_path (str, optional): The file path to save the generated audio summary. If None, audio will not be saved to disk.
  Returns:
    tuple: A tuple containing the extracted text, predicted emotions, selected keywords, generated insight sentences, and TTS result.
  """
  text = preprocess_image_and_extract_text(image_path)
  emotions = predict_emotions(text)
  keywords = extract_and_select_keywords(text)
  insight_sentences = generate_insight_sentences(text, emotions, keywords)
  if output_path:
    # only run TTS if output path provided, and save audio to specified path
    tts_res = pyttsx3_speak(insight_sentences, output_path)
  else:
    tts_res = None
  return text, emotions, keywords, insight_sentences, tts_res

if __name__ == "__main__":
  # example usage of the run_psychextract function with a sample image path
  result = run_psychextract("src\\example_io\\text1_a.png", "src\\example_io\\output_audio.mp3")