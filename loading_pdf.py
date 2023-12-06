from langchain.document_loaders import PyPDFLoader
loader = PyPDFLoader("docs/HBL.pdf")
pages = loader.load()

len(pages)
page = pages[0]
print(page.page_content[0:500])
print(page.metadata)