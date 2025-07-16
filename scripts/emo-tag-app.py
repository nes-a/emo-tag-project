```python
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_DIR = "./models/emo_tag" # Ensure this path points to your saved model directory
# If your model is stored elsewhere, update this path.

EMOTION_LABELS = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval',
    'caring', 'confusion', 'curiosity', 'desire', 'disappointment',
    'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear',
    'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism',
    'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise',
    'neutral'
]

@st.cache_resource
def load_hf_model_and_tokenizer(model_directory):
    """
    Loads the Hugging Face tokenizer and model from the specified directory.
    Uses st.cache_resource to cache the model, preventing re-loading on every rerun.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_directory)
    model = AutoModelForSequenceClassification.from_pretrained(model_directory, local_files_only=True)
    model.eval() # Set model to evaluation mode
    return tokenizer, model

tokenizer, model = load_hf_model_and_tokenizer(MODEL_DIR)
IS_PYTORCH_MODEL = isinstance(model, torch.nn.Module)

def preprocess_text(text, tokenizer, max_length=128):
    """
    Tokenizes and preprocesses the input text for the model.
    """
    return_tensors = "pt" if IS_PYTORCH_MODEL else "tf"
    return tokenizer(text, return_tensors=return_tensors, truncation=True, padding="max_length", max_length=max_length)

# --- Streamlit Page Configuration and Custom CSS ---
st.set_page_config(page_title="Emo Tag AI Assistant")

st.markdown(
    """
    <style>
    /* App background */
    .stApp {
        background-color: #f6f4e7;
    }

    /* Header/Toolbar background */
    header[data-testid="stHeader"] {
        background-color: #f6f4e7 !important;
    }
    [data-testid="stAppViewContainer"] {
        padding-top: 1rem !important;
    }

    /* Move main text down */
    .stApp p {
        margin-top: 1.6rem !important;
    }

    /* Apply styling to st.text_area */
    textarea {
        margin-top: 0 !important;
        background-color: #f6f4e7 !important;
        border: 2px solid #d6e10f !important;
        border-radius: 6px;
        color: black !important;
        min-height: 150px !important;
    }

    textarea:focus,
    textarea:hover {
        outline: 2px solid #d6e10f !important;
        box-shadow: 0 0 5px #d6e10f !important;
    }

    /* Button styling (main content buttons) - Updated for stFormSubmitButton */
    div[data-testid="stFormSubmitButton"] button {
        background-color: #f6f4e7 !important;
        color: black !important;
        border: 2px solid #d6e10f !important;
        padding: 8px 24px !important;
        border-radius: 6px !important;
        font-weight: bold !important;
        font-size: 16px !important;
        text-align: center !important;
        line-height: normal !important;
        transition: all 0.3s ease !important;
        margin-top: 0.5rem !important;
        outline: none !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    /* Fix button text vertical alignment by resetting inner p tag margins */
    div[data-testid="stFormSubmitButton"] button p {
        margin: 0 !important;
        padding: 0 !important;
    }
    div[data-testid="stFormSubmitButton"] button:active {
        border-color: #e74c3c !important; /* red border */
        box-shadow: 0 0 5px #e74c3c !important; /* optional red glow */
    }

    /* Sidebar background */
    [data-testid="stSidebar"] {
        background-color: #f6f4e7 !important;
    }

    /* Sidebar button styling */
    [data-testid="stSidebar"] button {
        background-color: #d6e10f !important;
        color: black !important;
        border: none !important;
    }

    [data-testid="stSidebar"] button:hover {
        background-color: #b8cd0b !important;
    }

    /* Sidebar text */
    [data-testid="stSidebar"],
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] a {
        color: black !important;
    }

    /* Single rule for sidebar logo positioning */
    [data-testid="stSidebar"] img {
        margin-top: 0 !important;
        padding-top: 1.5rem !important;
        margin-bottom: 0.5rem !important;
        display: block !important;
        margin-left: auto !important;
        margin-right: auto !important;
    }
    /* Fix sidebar toggle button positioning */
    button[aria-label="Toggle sidebar"] {
        position: fixed !important;
        top: 0 !important;
        left: 0 !important;
        margin-top: 15px !important;
        margin-left: 15px !important;
        z-index: 99999 !important;
        background-color: rgba(214, 225, 15, 0.9) !important;
        border-radius: 6px !important;
        padding: 6px 12px !important;
        color: black !important;
        border: 2px solid #d6e10f !important;
        box_shadow: 0 0 6px #d6e10f !important;
    }

    [data-testid="stSidebar"] .block-container {
        padding-top: 1rem !important;
    }

    [data-testid="stSidebar"] ul {
        margin-top: 0.2rem !important;
        padding-left: 1.8rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Sidebar Content ---
st.sidebar.image("images/emo-tag-logo.svg", width=250)
st.sidebar.markdown(
    "<ul style='margin-top: 0rem; padding-left: 2rem;'>"
    + "".join([f"<li>{e.replace('_', ' ').title()}</li>" for e in EMOTION_LABELS]) +
    "</ul>",
    unsafe_allow_html=True
)

# --- Main Application Content ---
st.write("**Emo Tag** — identifying emotions you can't name")

with st.form(key='emotion_analysis_form'):
    user_input = st.text_area(
        label="",
        height=150,
        placeholder="Enter your text here:",
        label_visibility="collapsed"
    )
    submit_button = st.form_submit_button("Analyze Emotion")

if submit_button:
    if user_input:
        st.write(f"Analyzing: {user_input}")
        with st.spinner("Analyzing emotions..."):
            processed_input = preprocess_text(user_input, tokenizer)
            if IS_PYTORCH_MODEL:
                processed_input = {k: v.to(model.device) for k, v in processed_input.items()}
                with torch.no_grad():
                    outputs = model(**processed_input)
                # Use sigmoid for multi-label classification probabilities, not softmax
                probabilities = torch.sigmoid(outputs.logits).cpu().numpy()[0]
            else:
                st.error("TensorFlow models not supported currently.")
                st.stop()

            # Create DataFrame of emotion scores
            emotion_scores = [{'Emotion': EMOTION_LABELS[i].replace('_', ' ').title(),
                               'Percentage': prob * 100} for i, prob in enumerate(probabilities)]
            emotion_df = pd.DataFrame(emotion_scores).sort_values(by='Percentage', ascending=False)

            # Replace 'Neutral' with 'Incognito' for display
            emotion_df['Emotion'] = emotion_df['Emotion'].replace({'Neutral': 'Incognito'})

            # Emotion category mapping (all lowercase for internal logic)
            emotion_categories = {
                "grounded": ['amusement', 'curiosity', 'sadness'],
                "layered": ['surprise', 'excitement', 'desire', 'confusion', 'grief', 'realization'],
                "protective": ['fear', 'anger', 'disgust', 'annoyance', 'disapproval', 'embarrassment', 'remorse', 'disappointment', 'nervousness'],
                "nurturing": ['gratitude', 'love', 'caring', 'optimism', 'joy', 'pride', 'relief', 'admiration', 'approval'],
                "incognito": ['incognito']
            }

            # Color mapping for categories
            category_colors = {
                "grounded": "#3498db",
                "layered": "#9b59b6",
                "protective": "#e74c3c",
                "nurturing": "#88c057",
                "incognito": "#7f8c8d"
            }

            # Get top emotion name for category detection
            top_emotion = emotion_df.iloc[0]
            top_emo_name = top_emotion['Emotion'].lower()

            # Detect category of top emotion, fallback to 'incognito' if not found
            emotion_category = next((cat for cat, ems in emotion_categories.items() if top_emo_name in ems), "incognito")
            category_color = category_colors.get(emotion_category, "#cccccc")

            related_emotions_list = emotion_categories.get(emotion_category, [])
            related_emotions = ", ".join([e.title() for e in related_emotions_list])
            emotion_category_display = emotion_category.title()

            # Prepare top 3 emotions display
            top_3_emotions = emotion_df.head(3)
            top_3_display = ", ".join([
                f"{row['Emotion']} ({row['Percentage']:.2f}%)"
                for _, row in top_3_emotions.iterrows()
            ])

            # --- Display Top Emotions Summary Box ---
            st.markdown(
                f"""
                <div style="
                    border: 2px solid #d6e10f;
                    background-color: #fffdd0;
                    border-radius: 10px;
                    padding: 1rem;
                    width: 100%;
                    box-sizing: border-box;
                    margin-bottom: 1.5rem;
                ">
                    <h4 style="margin: 0; color: #333;">Emotions:</h4>
                    <p3 style="margin: 0.2rem 0; color: #111;">{top_3_display}</p3>
                </div>
                """,
                unsafe_allow_html=True
            )

            if 'Incognito' in top_3_emotions['Emotion'].values:
                st.info(
                    "Incognito: no dominant emotion could be found in the text. It doesn't mean it's not there."
                )

            # --- Display Percentage Breakdown Bar Chart ---
            st.markdown("#### Percentage Breakdown:")

            # Filter emotions based on a threshold for the bar chart display
            threshold = 1 # Only show emotions with at least 1% probability
            filtered_df = emotion_df[emotion_df['Percentage'] >= threshold]
            if filtered_df.empty: # If no emotions meet threshold, show top 3 anyway
                filtered_df = emotion_df.head(3)

            fig, ax = plt.subplots(figsize=(10, max(6, len(filtered_df) * 0.7)))
            sns.barplot(x='Percentage', y='Emotion', data=filtered_df, palette='viridis', ax=ax)
            ax.set_xlabel('Probability (%)')
            ax.set_ylabel('Emotion')
            ax.set_xlim(0, 100)
            plt.title("Emotion Percentage Breakdown")
            st.pyplot(fig)
```
