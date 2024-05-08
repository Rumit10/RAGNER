import streamlit as st
import final

def without_rag():
    st.title("Without RAG")
    query = st.text_input("Enter query", "")
    
    if st.button("Generate response"):
        # Add your logic for processing queries without RAG here
        
        st.write(final.without_RAG(query))
        
        if st.button("Enter another query"):
            without_rag()

def with_rag():
    st.title("With RAG")
    query = st.text_input("Enter query", "")
    
    if st.button("Generate response"):
        # Add your logic for processing queries with RAG here
        st.write(final.with_RAG(query))
        
        if st.button("Enter another query"):
            with_rag()

def rag_with_ner():
    st.title("RAG with NER")
    query = st.text_input("Enter query", "")
    
    if st.button("Generate response"):
        # Add your logic for processing queries with RAG and NER here
        st.write(final.with_NER(query))
        
        if st.button("Enter another query"):
            rag_with_ner()

# Sidebar selection
option = st.sidebar.selectbox(
    'Select mode:',
    ('Without RAG', 'With RAG', 'RAG with NER')
)

# Main app logic
if option == 'Without RAG':
    without_rag()
elif option == 'With RAG':
    with_rag()
elif option == 'RAG with NER':
    rag_with_ner()

