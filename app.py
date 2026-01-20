import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

st.set_page_config(page_title="GenAI Text Generator")

MODEL_NAME = "gpt2"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    return tokenizer, model

st.title("Generative AI Text Generator")

prompt = st.text_area("Enter your prompt")

if st.button("Generate"):
    tokenizer, model = load_model()
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(
        inputs,
        max_length=150,
        do_sample=True,
        temperature=0.7
    )
    st.write(tokenizer.decode(output[0], skip_special_tokens=True))
