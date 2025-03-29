import streamlit as st
from transformers import BartForConditionalGeneration, BartTokenizer
import torch

# Load Model and Tokenizer
@st.cache_resource
def load_model():
    model = BartForConditionalGeneration.from_pretrained("legal_bart_summarizer")
    tokenizer = BartTokenizer.from_pretrained("legal_bart_summarizer")
    return model, tokenizer

model, tokenizer = load_model()

# App Interface
st.title("📜 Legal Document Summarizer")
st.markdown("Paste your legal document below and get an AI-generated summary")

# Input Section
input_text = st.text_area("Input Legal Document", height=300)
max_length = st.slider("Max Summary Length", 100, 512, 256)

# Generate Summary
if st.button("Generate Summary"):
    if not input_text.strip():
        st.error("Please enter some legal text first!")
    else:
        with st.spinner("Analyzing document..."):
            try:
                # Tokenize input
                inputs = tokenizer(
                    input_text,
                    max_length=1024,
                    truncation=True,
                    return_tensors="pt"
                )
                
                # Generate summary
                summary_ids = model.generate(
                    inputs.input_ids,
                    max_length=max_length,
                    num_beams=4,
                    early_stopping=True
                )
                
                # Decode and display
                summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                
                st.subheader("Generated Summary")
                st.markdown(f"```\n{summary}\n```")  # Code block for better readability
                
            except Exception as e:
                st.error(f"Error generating summary: {str(e)}")