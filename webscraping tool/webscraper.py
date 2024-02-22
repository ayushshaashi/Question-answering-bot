import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import requests
from bs4 import BeautifulSoup

model_name = "EleutherAI/gpt-neo-125M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def scrape_text_from_website(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        extracted_text = '\n'.join([p.get_text() for p in paragraphs])
        return extracted_text
    else:
        print("Error: Unable to retrieve content from the provided URL.")
        return None

def chatbot(input_text):

    scraped_text = scrape_text_from_website(input_text)
    if scraped_text:

        input_ids = tokenizer.encode(scraped_text, return_tensors="pt")

        response_ids = model.generate(input_ids, max_length=50, num_return_sequences=1, temperature=0.7)
        
        response_text = tokenizer.decode(response_ids[0], skip_special_tokens=True)
        
        return response_text
    else:
        return "Error: Unable to generate response."

iface = gr.Interface(fn=chatbot, 
                     inputs="text", 
                     outputs="text",
                     title="Chatbot",
                     description="Enter a website URL to generate a response based on its content.")

iface.launch(share=True)
