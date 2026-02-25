# tts_module.py - module for text-to-speech conversion using pyttsx3 and gTTS, providing functions to generate audio from text input

# import neccesary libraries
import pyttsx3
from gtts import gTTS
import io

def get_pyttsx3_engine() -> pyttsx3:
  """
  Initializes and returns a pyttsx3 TTS engine instance.
  Returns:
    pyttsx3: An instance of the pyttsx3 TTS engine.
  """
  return pyttsx3.init()

def pyttsx3_speak(text, out_path) -> str:
  """
  Converts the given text to speech and saves it to the specified output path using pyttsx3.
  Args:
    text (str): The text to be converted to speech.
    out_path (str): The file path where the generated audio will be saved.
  Returns:
    str: A message indicating the success or failure of the TTS generation.
  """
  tts_engine = get_pyttsx3_engine()
  try:
    tts_engine.save_to_file(text, out_path)
    tts_engine.runAndWait()
  except Exception as e:
    print(f"Error: {e}")
    return None
  return f"Successfully generated TTS file at {out_path}"

def gtts_speak(text: str) -> str:
  """
  Converts the given text to speech and returns it as a byte stream using gTTS.
  Args:
    text (str): The text to be converted to speech.
  Returns:
    io.BytesIO: A byte stream containing the generated audio data.
  """
  tts = gTTS(text, lang="en")
  # write the audio data to a byte stream instead of a file
  audio_bytes = io.BytesIO()
  # gTTS does not support writing directly to a byte stream
  # save to a temporary file and read it back
  tts.write_to_fp(audio_bytes)
  # reset the byte stream position to the beginning after writings
  audio_bytes.seek(0)
  return audio_bytes