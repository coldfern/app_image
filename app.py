import streamlit as st
from PIL import Image
import pytesseract
from pdf2image import convert_from_bytes
from transformers import pipeline
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
import nltk
import pandas as pd

# Ensure required NLTK data is downloaded
nltk.download('punkt')

# Set page config
st.set_page_config(
    page_title="Document Insight Generator",
    page_icon="📄",
    layout="wide"
)

# Initialize summarization pipeline (load only once)
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = load_summarizer()

def image_to_text(image):
    return pytesseract.image_to_string(image)

def pdf_to_text(uploaded_file):
    images = convert_from_bytes(uploaded_file.read())
    text = ""
    for i, image in enumerate(images):
        text += f"--- Page {i+1} ---\n"
        text += image_to_text(image) + "\n\n"
    return text

def generate_summary(text, max_length=150):
    if len(text.split()) < 50:
        return "Text is too short for meaningful summarization."
    
    summary = summarizer(text, max_length=max_length, min_length=30, do_sample=False)
    return summary[0]['summary_text']

def lsa_summary(text, sentences_count=3):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, sentences_count)
    return " ".join([str(sentence) for sentence in summary])

def extract_insights(text):
    sentences = nltk.sent_tokenize(text)
    long_sentences = [s for s in sentences if len(s.split()) > 15]
    questions = [s for s in sentences if s.strip().endswith('?')]
    
    word_freq = pd.Series(nltk.word_tokenize(text)).value_counts()
    top_words = word_freq.head(10).to_dict()
    
    insights = {
        "num_sentences": len(sentences),
        "num_words": len(nltk.word_tokenize(text)),
        "avg_sentence_length": sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0,
        "long_sentences": long_sentences[:3],
        "questions": questions,
        "top_words": top_words
    }
    return insights

def main():
    st.title("📄 Document Insight Generator")
    st.markdown("Upload images or PDFs to extract text and generate summaries with insights")
    
    uploaded_file = st.file_uploader(
        "Upload an image or PDF file",
        type=["png", "jpg", "jpeg", "pdf"],
        accept_multiple_files=False
    )
    
    if uploaded_file is not None:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Uploaded File Preview")
            if uploaded_file.type.startswith('image'):
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)
            elif uploaded_file.type == 'application/pdf':
                st.info("PDF file uploaded")
                images = convert_from_bytes(uploaded_file.read(), first_page=1, last_page=1)
                st.image(images[0], caption="First Page Preview", use_column_width=True)
                uploaded_file.seek(0)
        
        with col2:
            st.subheader("Processing Options")
            processing_mode = st.radio(
                "Select processing mode:",
                ("Fast Summary (BART)", "Detailed Analysis (LSA)", "Full Insights")
            )
            
            if st.button("Process Document"):
                with st.spinner("Extracting text and generating insights..."):
                    try:
                        if uploaded_file.type.startswith('image'):
                            image = Image.open(uploaded_file)
                            extracted_text = image_to_text(image)
                        elif uploaded_file.type == 'application/pdf':
                            extracted_text = pdf_to_text(uploaded_file)
                        
                        with st.expander("View Extracted Text"):
                            st.text_area("", extracted_text, height=300)
                        
                        st.subheader("Summary")
                        if processing_mode == "Fast Summary (BART)":
                            summary = generate_summary(extracted_text)
                            st.write(summary)
                        elif processing_mode == "Detailed Analysis (LSA)":
                            summary = lsa_summary(extracted_text, sentences_count=5)
                            st.write(summary)
                        
                        if processing_mode == "Full Insights":
                            insights = extract_insights(extracted_text)
                            
                            st.subheader("Document Statistics")
                            cols = st.columns(3)
                            cols[0].metric("Sentences", insights["num_sentences"])
                            cols[1].metric("Words", insights["num_words"])
                            cols[2].metric("Avg. Sentence Length", round(insights["avg_sentence_length"], 1))
                            
                            st.subheader("Key Insights")
                            
                            with st.expander("Top Words"):
                                st.bar_chart(pd.Series(insights["top_words"]))
                            
                            if insights["long_sentences"]:
                                with st.expander("Complex Sentences"):
                                    for i, sentence in enumerate(insights["long_sentences"], 1):
                                        st.markdown(f"{i}. {sentence}")
                            
                            if insights["questions"]:
                                with st.expander("Questions Found"):
                                    for i, question in enumerate(insights["questions"], 1):
                                        st.markdown(f"{i}. {question}")
                    
                    except Exception as e:
                        st.error(f"Error processing document: {str(e)}")

if __name__ == "__main__":
    main()
