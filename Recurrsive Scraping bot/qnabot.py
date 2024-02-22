import gradio as gr
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import requests
from bs4 import BeautifulSoup
import re

model_name = "distilbert-base-cased-distilled-squad"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

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

def scrape_text_and_urls_from_website(url):
    scraped_text = scrape_text_from_website(url)
    if scraped_text:

        embedded_urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', scraped_text)
        return scraped_text, embedded_urls
    else:
        return None, None

def recursive_scrape_text_from_urls(urls, depth=None):
    all_texts = []
    for url in urls:
        text, embedded_urls = scrape_text_and_urls_from_website(url)
        if text:
            all_texts.append(text)
            if embedded_urls and (depth is None or depth > 0):
                if depth is None:

                    depth = len(embedded_urls)

                all_texts += recursive_scrape_text_from_urls(embedded_urls, depth - 1)
    return all_texts

def chatbot(url, question):
    scraped_texts = recursive_scrape_text_from_urls([url])
    aggregated_text = "\n".join(scraped_texts)
    if aggregated_text:

        inputs = tokenizer(question, aggregated_text, return_tensors="pt")
        
        outputs = model(**inputs)
        answer_start_scores = outputs.start_logits
        answer_end_scores = outputs.end_logits
        
        answer_start = torch.argmax(answer_start_scores)
        answer_end = torch.argmax(answer_end_scores) + 1
        answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end]))
        
        return answer
    else:
        return "Error: Unable to generate response."

iface = gr.Interface(fn=chatbot, 
                     inputs=["text", "text"], 
                     outputs="text",
                     title="Question Answering Chatbot with Recursive Scraping",
                     description="Enter a website URL and ask a question based on its content. The chatbot will recursively scrape data from embedded URLs up to a certain depth.")

iface.launch(share=True)
