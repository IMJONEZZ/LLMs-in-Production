# Start service with:
# streamlit run chapters/chapter_8/listing_8.2.streamlit.py

import streamlit as st
import requests
import json

url = "http://localhost:8000/generate"  # point to your model's API

st.title("Chatbot with History")

# Create a chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat from history
for chat in st.session_state.chat_history:
    with st.chat_message(chat["role"]):
        st.markdown(chat["content"])

# Respond to User
# Note: we use the walrus operator (:=) to assign the user's input while
# also ensuring it's not None at the same time
if user_input := st.chat_input("Your question here"):
    # Display user's input
    with st.chat_message("user"):
        st.markdown(user_input)

    # Add user input to chat history
    st.session_state.chat_history.append(
        {"role": "user", "content": user_input}
    )

    # Stream response
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        # Format prompt adding chat history for additional context
        prompt = (
            "You are an assistant who helps the user. "
            "Answer their questions as accurately as possible. Be concise. "
        )
        history = [
            f'{ch["role"]}: {ch["content"]}'
            for ch in st.session_state.chat_history
        ]
        prompt += " ".join(history)
        prompt += " assistant: "
        data = json.dumps({"prompt": prompt})
        print(data)

        # Send request
        with requests.post(url, data=data, stream=True) as r:
            for line in r.iter_lines(decode_unicode=True):
                full_response += line.decode("utf-8")
                # Add a blinking cursor to simulate typing
                placeholder.markdown(full_response + "▌")
        placeholder.markdown(full_response)

    # Add LLM response to chat history
    st.session_state.chat_history.append(
        {"role": "assistant", "content": full_response}
    )
