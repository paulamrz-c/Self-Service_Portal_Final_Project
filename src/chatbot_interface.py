import streamlit as st
from retriever import answer


# Set the page title
st.set_page_config(page_title="Student FAQ Chatbot", layout="centered")
st.title("🎓 Ask the Student Affairs Chatbot")

# Display a short instruction
st.write("Ask a question and I’ll suggest relevant student support resources.")

st.markdown("""
Type your question below. The chatbot will find the most relevant FAQ or student resource using vector similarity.
""")

query = st.text_input("💬 Your question:")

model_type = st.radio("Choose embedding model:", ["w2v", "glove"], horizontal=True)

if st.button("Get Answer") and query:
    with st.spinner("Searching for the best answer..."):
        response = answer(query, model_type=model_type)
    st.success("✅ Result:")
    st.markdown(response)


