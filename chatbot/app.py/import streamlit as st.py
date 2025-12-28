import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="AI Tutor Chatbot", page_icon="🤖")

st.title("🤖 AI Tutor Chatbot")
st.caption("Ask me about AI, DBMS, Python, and more!")

@st.cache_resource
def load_model():
    return pipeline(
        "text-generation",
        model="microsoft/DialoGPT-medium"
    )

chatbot = load_model()

if "messages" not in st.session_state:
    st.session_state.messages = []

user_input = st.text_input("You:", placeholder="Type your message here...")

if st.button("Send") and user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    conversation = ""
    for msg in st.session_state.messages:
        conversation += f"{msg['role']}: {msg['content']}\n"
    conversation += "bot:"

    response = chatbot(
        conversation,
        max_new_tokens=80,
        temperature=0.6,
        pad_token_id=50256
    )

    bot_reply = response[0]["generated_text"].split("bot:")[-1].strip()
    st.session_state.messages.append({"role": "bot", "content": bot_reply})

# Display chat
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"**You:** {msg['content']}")
    else:
        st.markdown(f"**🤖 TutorBot:** {msg['content']}")
