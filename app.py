


import streamlit as st
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "google/flan-t5-base"
qa_tokenizer = AutoTokenizer.from_pretrained(model_name)
qa_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Extract text from uploaded PDFs
def extract_text_from_pdf(pdf_files):
    all_text = ""
    for pdf_file in pdf_files:
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    all_text += text
    return all_text

# Split text into chunks
def split_text_into_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)
    return text_splitter.split_text(text)

# Store text chunks as embeddings in FAISS
def store_embeddings(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(text_chunks, embeddings)
    return vectorstore

# Retrieve relevant chunks and generate a generative answer
def get_pdf_response(user_query, vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    docs = retriever.get_relevant_documents(user_query)
    context = " ".join([doc.page_content for doc in docs])

    if not context:
        return "❌ Sorry, I couldn't find any relevant context."

    # Generate answer using Flan-T5
    input_text = f"Answer the question based on the context.\nQuestion: {user_query}\nContext: {context}"
    input_ids = qa_tokenizer.encode(input_text, return_tensors="pt", truncation=True, max_length=512)
    output_ids = qa_model.generate(input_ids, max_length=256)
    response = qa_tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return response

# Streamlit UI
st.set_page_config(page_title="DocuChat", layout="wide")
st.title('📚 DOCUCHAT - Free Multiple PDF Chatter')

# Sidebar for PDF upload
st.sidebar.header('📂 Upload Your PDF Documents')
pdf_files = st.sidebar.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

# Main logic
if pdf_files:
    text_data = extract_text_from_pdf(pdf_files)
    text_chunks = split_text_into_chunks(text_data)
    vectorstore = store_embeddings(text_chunks)

    st.sidebar.success("✅ PDF Processed! You can now chat.")

    if st.button("📄 Show Extracted Text Chunks"):
        st.subheader("🔍 Extracted Text Chunks")
        for i, chunk in enumerate(text_chunks):
            st.text_area(f"Chunk {i+1}", chunk, height=100)

    user_input = st.text_input("💬 Ask a question about the PDFs:")
    if user_input:
        response = get_pdf_response(user_input, vectorstore)
        st.markdown(f"### 🤖 DocuChat Response:\n{response}")

# Footer for i, chunk in enumerate(chunks):
      #  print(f"\nChunk {i+1}:\n{chunk}\n")
st.markdown('<div style="padding: 20px;"></div>', unsafe_allow_html=True)
st.markdown("""<footer style="text-align:center; font-size:12px;">Made with ❤ by Group 6 | Multiple PDF Generative QA App</footer>""", unsafe_allow_html=True)
