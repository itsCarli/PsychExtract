from ocr_module import load_preprocess_and_extract
from emotion_module import load_and_predict_emotions

def run_psychextract(image_path: str):
  text = load_preprocess_and_extract(image_path)

  emotions = load_and_predict_emotions(text)

  return text, emotions

if __name__ == "__main__":
  result = run_psychextract("C:\\Users\\carli\\OneDrive\\UoL\\FP\\Deliverables\\PsychExtract\\data\\OCR\\raw_handwritten\\text0_a.png")
  print(result)