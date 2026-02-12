# python.exe -m pip install --upgrade pip
# pip install numpy transformers torch torchvision pillow opencv-python
# ! pip install yake spacy
# ! python -m spacy download en_core_web_sm

from ocr_module import load_preprocess_and_extract
from emotion_module import load_and_predict_emotions
from keyword_module import extract_and_select_keywords
from template_module import generate_insight_sentences
def run_psychextract(image_path: str):
  # text = load_preprocess_and_extract(image_path)
  text = "i noticed how tense my body felt this morning. my shoulders were tight, and i struggled to slow my breathing"

  emotions = load_and_predict_emotions(text)

  keywords = extract_and_select_keywords(text)

  insight_sentences = generate_insight_sentences(text, emotions, keywords)

  return text, emotions, keywords, insight_sentences

# if __name__ == "__main__":
  # result = run_psychextract("C:\\Users\\carli\\OneDrive\\UoL\\FP\\Deliverables\\PsychExtract\\data\\OCR\\raw_handwritten\\text0_a.png")
  # print(result)