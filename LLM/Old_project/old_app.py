import streamlit as st
import os
import old_final

def save_file(file, filename):
    with open(filename, "wb") as f:
        f.write(file.getbuffer())
    st.success(f"File saved as {filename}")

def professor_mode():
    st.title("Professor Mode")
    uploaded_pdf = st.file_uploader("Upload PDF (Resource)", type="pdf")
    uploaded_json = st.file_uploader("Upload JSON (Database)")

    if uploaded_pdf is not None and uploaded_json is not None:
        if st.button("Save Files"):
            save_file(uploaded_pdf, "the_resource.pdf")
            save_file(uploaded_json, "the_database.json")

def student_mode():
    st.title("Student Mode")
    query_type = st.selectbox("Select Query Type", ["General", "Coursework", "Stats"])


    if query_type == "General":
        general_query = st.text_area("Enter your general query here:")
        if st.button("Submit"):
            # Process general query
            st.write(old_final.without_RAG(general_query))

    elif query_type == "Coursework":
        coursework_query = st.text_area("Enter your coursework query here:")
        if st.button("Submit"):
            # Process coursework query
            st.write(old_final.with_RAG(coursework_query))

    elif query_type == "Stats":
        stats_query = st.text_area("Enter your stats query here:")
        if st.button("Submit"):
            # Process stats query
            st.write(old_final.with_JSON(stats_query))

    

def main():
    st.sidebar.title("Mode Selection")
    mode = st.sidebar.radio("Select Mode", ("Professor", "Student"))

    if mode == "Professor":
        professor_mode()
    elif mode == "Student":
        student_mode()

if __name__ == "__main__":
    main()
