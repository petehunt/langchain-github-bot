# This is the first simple example from the blog post that processes data
# from Wikipedia and does not use orchestration

from langchain.llms import OpenAI
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.docstore.document import Document
import requests
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.text_splitter import CharacterTextSplitter


def get_wiki_data(title, first_paragraph_only):
    url = f"https://en.wikipedia.org/w/api.php?format=json&action=query&prop=extracts&explaintext=1&titles={title}"
    if first_paragraph_only:
        url += "&exintro=1"
    data = requests.get(url).json()
    return Document(
        page_content=list(data["query"]["pages"].values())[0]["extract"],
        metadata={"source": f"https://en.wikipedia.org/wiki/{title}"},
    )


sources = [
    get_wiki_data("Unix", False),
    get_wiki_data("Microsoft_Windows", False),
    get_wiki_data("Linux", False),
    get_wiki_data("Seinfeld", False),
    get_wiki_data("Matchbox_Twenty", False),
    get_wiki_data("Roman_Empire", False),
    get_wiki_data("London", False),
    get_wiki_data("Python_(programming_language)", False),
    get_wiki_data("Monty_Python", False),
]

source_chunks = []
splitter = CharacterTextSplitter(separator=" ", chunk_size=1024, chunk_overlap=0)
for source in sources:
    for chunk in splitter.split_text(source.page_content):
        source_chunks.append(Document(page_content=chunk, metadata=source.metadata))

search_index = FAISS.from_documents(source_chunks, OpenAIEmbeddings())

chain = load_qa_with_sources_chain(OpenAI(temperature=0))


def print_answer(question):
    print(
        chain(
            {
                "input_documents": search_index.similarity_search(question, k=4),
                "question": question,
            },
            return_only_outputs=True,
        )["output_text"]
    )
