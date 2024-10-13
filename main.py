from dotenv import load_dotenv
from equity_research_tool import EquityResearchTool
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st

load_dotenv()
tool = EquityResearchTool()

st.title("EquityBot: Finance and Equity News Research Tool 📈")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

if all([url == '' for url in urls]):
    urls = []

col1, col2, col3 = st.columns([1, 1, 10])
with col1:
    process_url_clicked = st.sidebar.button("Process URLs",
                                            use_container_width=True)
st.header("Question: ")
main_placeholder = st.empty()

if process_url_clicked:

    loader = UnstructuredURLLoader(urls)

    main_placeholder.text("Data Loading...Started...✅✅✅")

    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000,
        chunk_overlap=200)

    main_placeholder.text("Text Splitter...Started...✅✅✅")

    docs = text_splitter.split_documents(data)
    main_placeholder.text("Embedding Vector Started Building...✅✅✅")

    tool.add_documents(docs)

query = main_placeholder.text_input("")

if query:
    result = tool.answer(query=query, source_urls=urls)

    st.write(result["answer"])

    sources = result.get("source", "")
    if sources:
        st.subheader("Sources:")
        sources_list = sources.split("\n")
        for source in sources_list:
            st.write(source)

with col2:
    delete_memory_clicked = st.sidebar.button("Delete Memory",
                                              use_container_width=True)
if delete_memory_clicked:
    if len(urls) != 0:
        tool.delete_document(urls=urls)
        st.sidebar.markdown(
            "<span style='color: green'>Memory deletion successful...</span>✅✅✅",
            unsafe_allow_html=True)
    else:
        st.sidebar.markdown(
            "<span style='color: red'>You didn't provide any URLs for which the memory needs to be erased.</span>",
            unsafe_allow_html=True)
