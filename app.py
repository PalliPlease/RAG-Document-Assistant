import streamlit as st
from chat import build_qa_chain
from pypdf import PdfReader

@st.cache_resource(show_spinner=True)
def load():
    #Doing this to not build it again and again
    return build_qa_chain()

qa_chain, retriever = load()

st.title("Doc reader")
# file = st.file_uploader("Enter the document you want to read", type="pdf")
#
# if file is not None:
#     reader = PdfReader(file)
#

st.write("Ask your document")


question = st.text_input("Your question:")

if st.button("Ask") and question.strip():
    with st.spinner("Thinking.."):
        answer = qa_chain.invoke(question)

        st.write(answer)

#use streamlit run app.py