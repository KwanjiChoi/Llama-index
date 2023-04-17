import csv
from llama_index import GPTSimpleVectorIndex, download_loader

BeautifulSoupWebReader = download_loader("BeautifulSoupWebReader")
loader = BeautifulSoupWebReader()

article_urls = []
with open('tmp/urls.csv') as f:
    reader = csv.reader(f)
    for row in reader:
        article_urls.append(row[0])
documents = loader.load_data(article_urls)
index = GPTSimpleVectorIndex.from_documents(documents)

# save to disk
index.save_to_disk('index.json')
