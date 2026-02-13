# python.exe -m pip install --upgrade pip
# pip install numpy transformers torch torchvision pillow opencv-python yake spacy pyttsx3
# python -m spacy download en_core_web_sm

from ocr_module import load_preprocess_and_extract
from emotion_module import load_and_predict_emotions
from keyword_module import extract_and_select_keywords
from template_module import generate_insight_sentences
from tts_module import speak

def run_psychextract(image_path: str, output_path: str):
  text = load_preprocess_and_extract(image_path)
  # text = "I noticed how tense my body felt this morning. My shoulders were tight, and I struggled to slow my breathing"

  emotions = load_and_predict_emotions(text)

  keywords = extract_and_select_keywords(text)

  insight_sentences = generate_insight_sentences(text, emotions, keywords)

  tts_res = speak(insight_sentences, output_path)

  return text, emotions, keywords, insight_sentences, tts_res

# if __name__ == "__main__":
  # result = run_psychextract("C:\\Users\\carli\\OneDrive\\UoL\\FP\\Deliverables\\PsychExtract\\data\\OCR\\raw_handwritten\\text0_a.png")
  # print(result)