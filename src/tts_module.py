import pyttsx3

def get_tts_engine() -> pyttsx3:
  return pyttsx3.init()

def speak(text, out_path):
  tts_engine = get_tts_engine()
  try:
    tts_engine.save_to_file(text, out_path)
    tts_engine.runAndWait()
  except Exception as e:
    print(f"Error: {e}")
    return None

  return f"Successfully generated TTS file at {out_path}"