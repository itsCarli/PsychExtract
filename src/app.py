# app.py - main Streamlit app for the PsychExtract prototype demo

# modules
from ocr_module import preprocess_image_and_extract_text
from emotion_module import predict_emotions 
from keyword_module import extract_and_select_keywords
from template_module import generate_insight_sentences, format_list_into_string
from tts_module import gtts_speak

# libraries
import streamlit as st
from PIL import Image
import tempfile

# cached resource loading for models
@st.cache_resource
def get_emotion_resources() -> tuple:
  """
  Load the emotion classification model and tokenizer, with caching to optimize performance.
  Returns:
    tuple: A tuple containing the loaded emotion model and tokenizer.
  """
  from emotion_module import load_emotion_model
  return load_emotion_model()

@st.cache_resource
def get_ocr_resources() -> tuple:
  """
  Load the OCR model and processor, with caching to optimize performance.
  Returns: 
    tuple: A tuple containing the loaded OCR model and processor.
  """
  from ocr_module import load_qwen
  return load_qwen()

def save_uploaded_file(uploaded_file) -> str:
  """
  Save the uploaded file to a temporary location and return the file path.
  Args:
    uploaded_file: The file uploaded by the user through Streamlit's file uploader.    
  Returns:
    str: The file path of the saved uploaded file.
  """
  with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
    tmp.write(uploaded_file.getbuffer())
    return tmp.name

def main() -> None:
  """
  The main function that defines the Streamlit app workflow for the PsychExtract prototype demo. 
  
  It handles file upload, OCR processing, emotion prediction, keyword extraction, insight generation, and user feedback collection.
  Returns:
    None: This function does not return any value, as it is responsible for rendering the Streamlit app interface and managing user interactions.
  """

  # initialize session state variables
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

  # set default values for session state variables if not already set
  for key, default in session_state.items():
    if key not in st.session_state:
      st.session_state[key] = default

  # Streamlit app layout and workflow
  st.set_page_config(page_title="PsychExtract Demo", layout="wide")
  st.title("PsychExtract: Prototype Demo")

  # load models with caching
  with st.spinner("Loading AI models..."):
    emotion_model, emotion_tokenizer = get_emotion_resources()
    qwen_model, qwen_processor = get_ocr_resources()

  # image upload
  st.write("Upload a handwritten therapy note in the form of an image to see the prototype workflow.")
  uploaded = st.file_uploader(
    "Upload handwritten note (image)", 
    type=["png", "jpg", "jpeg"],
    key=st.session_state.uploader_key) 
  if uploaded:
    # if a new file is uploaded, update session state
    st.session_state.uploaded_file = uploaded

  # display uploaded image and run through workflow steps if image is uploaded
  if st.session_state.uploaded_file:
    st.subheader("Uploaded Image")
    img = Image.open(st.session_state.uploaded_file)
    st.image(img, width=500)

    # --- OCR ---
    if not st.session_state.ocr_done:
      # only show button if OCR not yet done, and set state to trigger rerun on click
      if st.button("Run OCR"):
        st.session_state.ocr_done = True
        st.rerun()  # rerun so button disappears immediately

    if st.session_state.ocr_done:
      if not st.session_state.ocr_text:
        # run OCR logic if text not yet set
        with st.spinner("Running OCR..."):
          image_path = save_uploaded_file(st.session_state.uploaded_file)
          extracted_text = preprocess_image_and_extract_text(
            image_path,
            qwen_model,
            qwen_processor
          )
          st.session_state.ocr_text = extracted_text if extracted_text else st.session_state.ocr_text
      # display OCR output and allow user to correct it
      st.subheader("Extracted Text")
      st.session_state.ocr_text = st.text_area("Correct the OCR output:", st.session_state.ocr_text, height=150)

      # --- INSIGHT GENERATION ---
      # only show button if insights not yet done, and set state to trigger rerun on click
      if not st.session_state.insights_done:
        if st.button("Generate Psychological Insights"):
          st.session_state.insights_done = True
          st.rerun()  # rerun so button disappears immediately

    if st.session_state.insights_done:
      # only run insights logic if triggered by button click
      with st.spinner("Analyzing note..."):
        emotions_list = predict_emotions(
          st.session_state.ocr_text,
          emotion_model,
          emotion_tokenizer
        )
        emotion_threshold = 0.3
        # filter emotions by threshold and format as percentages for display
        filtered_emotions = {
          emotion: f"{(round(score, 2) * 100)}%" 
          for emotion, score in emotions_list.items() 
          if score >= emotion_threshold
        }
        # update session state with insight results (emotions, keywords, summary)
        st.session_state.emotions = filtered_emotions
        st.session_state.keywords = extract_and_select_keywords(st.session_state.ocr_text)
        st.session_state.summary = generate_insight_sentences(st.session_state.ocr_text, emotions_list, st.session_state.keywords, emotion_threshold)

      # display emotion results
      st.subheader("Emotions Detected")
      st.write(st.session_state.emotions)

      # display keywords
      st.subheader("Linguistic Markers")
      st.write(f"Mention of: {format_list_into_string(st.session_state.keywords)}")

      # display summary
      st.subheader("Summary")
      st.write(st.session_state.summary)

      # --- TTS ---
      st.subheader("Audio Summary")
      if not st.session_state.audio_done:
        # only run TTS logic if not already done
        with st.spinner("Generating audio..."):
          st.session_state.audio_bytes = gtts_speak(st.session_state.summary)
        st.session_state.audio_done = True
      # display audio player if audio bytes available
      st.audio(st.session_state.audio_bytes, format="audio/mp3")

      # user feedback
      st.subheader("User Feedback")
      # only show feedback options if insights have been generated, and store feedback in session state
      st.session_state.feedback = st.radio(
        "Was this summary accurate?",
        ["Yes", "No"],
        index=None,
        key="feedback_radio"
      )
      
  # reset button to clear session state and start fresh
  if st.button("Reset Demo"):
    # clear all session state variables and rerun to reset the app
    for key in list(st.session_state.keys()):
      del st.session_state[key]
    st.rerun()  # fully restart the app

if __name__ == "__main__":
  # run the main function to start the Streamlit app
  main()
