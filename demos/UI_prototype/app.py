import streamlit as st
from PIL import Image
import time
import io
from gtts import gTTS

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
        time.sleep(2)  # simulate OCR delay
        st.session_state.ocr_text = (
          "I noticed how tense my body felt this morning. "
          "My shoulders were tight, and I struggled to slow my breathing"
        )
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
      time.sleep(5)  # simulate processing

    # Emotion results
    st.subheader("Emotions Detected")
    st.write({"anticipation": 0.22, "joy": 0.43})

    # 5. Linguistic markers results
    st.subheader("Linguistic Markers")
    st.write("Mention of:\n- morning\n- body\n- shoulders")

    # 6. Summary
    st.subheader("Summary")
    summary_text = "Emotions of joy are detected. Themes of Regulation and Coping Mode, Arousal or Restlessness Level, and Self-Relation and Appraisal are detected. This entry highlights an active attempt to regulate emotions through reflection, particularly in response to their morning, body, and shoulders. The language used suggests heightened internal activation or restlessness related to their morning, body, and shoulders. The writer appears to be assessing their own reactions or patterns while considering their morning, body, and shoulders."
    st.write(summary_text)

    # 7. TTS output
    st.subheader("Audio Summary")
    if not st.session_state.audio_done:
      with st.spinner("Generating audio..."):
        tts = gTTS(summary_text)
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
  for key in session_state.keys():
    st.session_state[key] = False if "done" in key else None
  st.session_state.ocr_text = ""
  st.session_state.uploader_key = f"file_uploader_{time.time()}"  # force uploader reset
  st.rerun()  # fully restart the app

# streamlit run app.py
# for demo day streamlit run your_filename.py --server.address 0.0.0.0