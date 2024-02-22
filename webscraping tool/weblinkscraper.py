from bs4 import BeautifulSoup
import requests

url = 'https://docs.npmjs.com'
page = requests.get(url)
soup = BeautifulSoup(page.text, 'html.parser')
all_links = soup.find_all('a')
for link in all_links:
    href = link.get('href')
    if href:
        print(href)