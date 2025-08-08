import streamlit as st
import requests

API_URL = "http://localhost:8000/predict"

# Set page title
st.set_page_config(page_title="Student FAQ Chatbot", layout="centered")


# Start History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Title and instructions
st.title("🎓 AI Student Support Chatbot 🤖")
st.caption("Ask me anything. I’ll try to find the most relevant FAQ or student resource.")
st.markdown("""
Type your question below. The chatbot will find the most relevant FAQ or student resource using vector similarity.
""")

# Showing History
for sender, message in st.session_state.messages:
    if sender == "user":
        st.markdown(f"""
        <div style="text-align: right; margin-bottom: 10px;">
            <span style="background-color: #DDC7EF; color: ##3E1A5B; padding: 10px 15px; border-radius: 13px; display: inline-block; max-width: 80%;">
                {message}
            </span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="text-align: left; margin-bottom: 10px;">
            <span style="background-color: #B280DB; color: ##3E1A5B; padding: 10px 15px; border-radius: 13px; display: inline-block; max-width: 80%;">
                {message}
            </span>
        </div>
        """, unsafe_allow_html=True)

# User input
query = st.text_input("💬 Type your message:", key="chat_input")

# On button click
if st.button("Send") and query.strip():
    st.session_state.messages.append(("user", query))

    try:
        with st.spinner("Thinking..."):
            res = requests.post(API_URL, json={"question": query})
            if res.status_code == 200:
                out = res.json()
                reply = f"{out['answer']} )"
            else:
                reply = "❌ API error. Please try again."
    except:
        reply = "🚫 Unable to reach the backend."

    st.session_state.messages.append(("bot", reply))
    st.rerun()

