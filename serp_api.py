import requests
from bs4 import BeautifulSoup
from langchain_community.document_loaders import WebBaseLoader

examples = [
    {"URL": "https://en.wikipedia.org/wiki/Climate_change", "Question": "What are the impacts of climate change?"},
    {"URL": "https://www.cdc.gov/physical-activity-basics/benefits/?CDC_AAref_Val=https://www.cdc.gov/physicalactivity/basics/pa-health/index.html", "Question": "What are the health benefits of physical activity?"},
]


import requests
from bs4 import BeautifulSoup

def get_content_from_url(url="https://en.wikipedia.org/wiki/ASAP_Rocky"):
    response = requests.get(url)
    html_content = response.text

    # Parse the HTML content
    soup = BeautifulSoup(html_content, 'html.parser')

    # Find all relevant elements
    headings_h2 = soup.select('.mw-heading2')
    headings_h3 = soup.select('.mw-heading3')
    paragraphs = soup.find_all("p")

    # Combine text from headings and paragraphs
    content = []
    
    for heading in headings_h2 + headings_h3:
        content.append(heading.get_text(strip=True))
    
    for para in paragraphs:
        content.append(para.get_text(strip=True))

    return "\n".join(content).strip()



if __name__ == "__main__":
    # Example usage
    print(get_content_from_url())

