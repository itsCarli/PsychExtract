from ocr_module import load_preprocess_and_extract

def run_psychextract(image_path: str):
  text = load_preprocess_and_extract(image_path)
  return text

if __name__ == "__main__":
  result = run_psychextract("C:\\Users\\carli\\OneDrive\\UoL\\FP\\Deliverables\\PsychExtract\\data\\OCR\\raw_handwritten\\text0_a.png")
  print(result)