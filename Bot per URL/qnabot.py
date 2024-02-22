import gradio as gr
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import requests
from bs4 import BeautifulSoup
import torch

model_name = "distilbert-base-cased-distilled-squad"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

def scrape_text_from_website(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            paragraphs = soup.find_all('p')
            extracted_text = '\n'.join([p.get_text() for p in paragraphs])
            return extracted_text
        else:
            return "Error: Unable to retrieve content from the provided URL."
    except Exception as e:
        return f"Error: {str(e)}"


def chatbot(url, question):
    try:

        scraped_text = scrape_text_from_website(url)
        if "Error:" in scraped_text:
            return scraped_text

        inputs = tokenizer(question, scraped_text, return_tensors="pt", max_length=512)
        
        outputs = model(**inputs)
        answer_start_scores = outputs.start_logits
        answer_end_scores = outputs.end_logits
        
        answer_start = torch.argmax(answer_start_scores)
        answer_end = torch.argmax(answer_end_scores) + 1
        answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end]))
        
        return answer
    except Exception as e:
        return f"Error: {str(e)}"

iface = gr.Interface(fn=chatbot, 
                     inputs=["text", "text"], 
                     outputs="text",
                     title="Question Answering Chatbot per URL",
                     description="Enter a website URL and ask a question based on its content.")

# Launch interface
iface.launch(share=True)