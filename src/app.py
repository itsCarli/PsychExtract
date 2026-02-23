# from main import run_psychextract
from ocr_module import load_preprocess_and_extract
from emotion_module import load_and_predict_emotions
from keyword_module import extract_and_select_keywords
from template_module import generate_insight_sentences, format_list_into_string

import streamlit as st
from PIL import Image
import time
import io
from gtts import gTTS

import tempfile
# import os

def save_uploaded_file(uploaded_file):
  with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
    tmp.write(uploaded_file.getbuffer())
    return tmp.name

def main():
  session_state = {
    "ocr_done": False,
    "ocr_text": "",
    "insights_done": False,
    "audio_done": False,
    "audio_bytes": None,
    "feedback": None,
    "uploaded_file": None,
    "uploader_key": "file_uploader"
  }

  for key, default in session_state.items():
    if key not in st.session_state:
      st.session_state[key] = default

  st.set_page_config(page_title="PsychExtract Demo", layout="wide")
  st.title("PsychExtract: Prototype Demo")

  # 1. File upload
  st.write("Upload a handwritten therapy note to see the prototype workflow.")
  uploaded = st.file_uploader(
    "Upload handwritten note (image)", 
    type=["png", "jpg", "jpeg"],
    key=st.session_state.uploader_key)
  if uploaded:
    st.session_state.uploaded_file = uploaded

  if st.session_state.uploaded_file:
    st.subheader("Uploaded Image")
    img = Image.open(st.session_state.uploaded_file)
    st.image(img, width=500)

    # 2. OCR
    if not st.session_state.ocr_done:
      if st.button("Run OCR"):
        st.session_state.ocr_done = True
        st.rerun()  # rerun so button disappears immediately

    if st.session_state.ocr_done:
      if not st.session_state.ocr_text:
        # Run OCR logic if text not yet set
        with st.spinner("Running OCR..."):
          image_path = save_uploaded_file(st.session_state.uploaded_file)
          extracted_text = load_preprocess_and_extract(image_path)
          st.session_state.ocr_text = extracted_text if extracted_text else st.session_state.ocr_text
      st.subheader("Extracted Text")
      st.session_state.ocr_text = st.text_area("Correct the OCR output:", st.session_state.ocr_text, height=150)

      # 3. Generate insights
      if not st.session_state.insights_done:
        if st.button("Generate Psychological Insights"):
          st.session_state.insights_done = True
          st.rerun()  # rerun so button disappears immediately

    # 4. Display results
    if st.session_state.insights_done:
      with st.spinner("Analyzing note..."):
        emotion_threshold = 0.3
        emotions_list = load_and_predict_emotions(st.session_state.ocr_text)
        filtered_emotions = {
          emotion: f"{(round(score, 2) * 100)}%" 
          for emotion, score in emotions_list.items() 
          if score >= emotion_threshold
          }
        st.session_state.emotions = filtered_emotions
        st.session_state.keywords = extract_and_select_keywords(st.session_state.ocr_text)
        st.session_state.summary = generate_insight_sentences(st.session_state.ocr_text, emotions_list, st.session_state.keywords, emotion_threshold)

      # Emotion results
      st.subheader("Emotions Detected")
      st.write(st.session_state.emotions)

      # 5. Linguistic markers results
      st.subheader("Linguistic Markers")
      st.write(f"Mention of: {format_list_into_string(st.session_state.keywords)}")

      # 6. Summary
      st.subheader("Summary")
      st.write(st.session_state.summary)

      # 7. TTS output
      st.subheader("Audio Summary")
      if not st.session_state.audio_done:
        with st.spinner("Generating audio..."):
          tts = gTTS(st.session_state.summary, lang='en')
          audio_bytes = io.BytesIO()
          tts.write_to_fp(audio_bytes)
          audio_bytes.seek(0)
          st.session_state.audio_bytes = audio_bytes
        st.session_state.audio_done = True
      st.audio(st.session_state.audio_bytes, format="audio/mp3")

      # 7. User feedback
      st.subheader("User Feedback")
      st.session_state.feedback = st.radio(
        "Was this summary accurate?",
        ["Yes", "No"],
        index=None,
        key="feedback_radio"
        )
      
  # 0. Reset button
  if st.button("Reset Demo"):
    # for key in session_state.keys():
    #   st.session_state[key] = False if "done" in key else None
    # st.session_state.ocr_text = ""
    # st.session_state.uploader_key = f"file_uploader_{time.time()}"  # force uploader reset
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()  # fully restart the app

if __name__ == "__main__":
  main()
