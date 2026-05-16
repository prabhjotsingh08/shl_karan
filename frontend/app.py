"""Streamlit chatbot UI for the SHL conversational assessment agent."""

from __future__ import annotations

import os
from typing import List

import requests
import streamlit as st


def _normalize_api_url(raw: str) -> str:
    value = raw.strip().rstrip("/")
    if value and not value.startswith(("http://", "https://")):
        return f"https://{value}"
    return value


API_URL = _normalize_api_url(os.environ.get("RECOMMENDER_API_URL", "http://localhost:8000"))
MAX_MESSAGES = 8  # user + assistant turns combined (assignment limit)


def fetch_chat_reply(messages: List[dict]) -> dict:
    """Call the stateless /chat endpoint with full conversation history."""
    response = requests.post(
        f"{API_URL}/chat",
        json={"messages": messages},
        timeout=90,
    )
    response.raise_for_status()
    return response.json()


def to_api_messages(history: List[dict]) -> List[dict]:
    return [{"role": entry["role"], "content": entry["content"]} for entry in history]


if "messages" not in st.session_state:
    st.session_state.messages = []


st.set_page_config(page_title="SHL Assessment Advisor", layout="wide", page_icon="💼")

st.title("💼 SHL Assessment Recommendation Chatbot")
st.markdown(
    """
    <div style='background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin-bottom: 20px;'>
        <p style='margin: 0; color: #1f2937;'>
            👋 <strong>Welcome!</strong> Describe a role or hiring need. I may ask clarifying questions,
            then recommend SHL catalog assessments, refine results, or compare options.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

message_count = len(st.session_state.messages)
st.caption(f"Conversation messages: {message_count} / {MAX_MESSAGES} (user + assistant)")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        for assessment in message.get("recommendations", []):
            with st.expander(f"📋 {assessment.get('name', 'Assessment')}"):
                if assessment.get("description"):
                    st.markdown(f"**Description:** {assessment['description']}")
                if assessment.get("test_type"):
                    st.markdown(f"**Test type:** {', '.join(assessment['test_type'])}")
                if assessment.get("duration"):
                    st.markdown(f"**Duration:** {assessment['duration']} minutes")
                if assessment.get("url"):
                    st.markdown(f"[View in SHL catalog]({assessment['url']})")


if prompt := st.chat_input("Describe the role or assessment needs..."):
    if message_count >= MAX_MESSAGES:
        st.warning("Maximum 8 messages reached. Clear chat history to start over.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    payload = to_api_messages(st.session_state.messages)
                    data = fetch_chat_reply(payload)
                    reply = data.get("reply", "")
                    recommendations = data.get("recommendations", [])
                    end_of_conversation = bool(data.get("end_of_conversation"))

                    st.markdown(reply)

                    if recommendations:
                        st.success(f"Found {len(recommendations)} catalog assessment(s)")
                        for assessment in recommendations:
                            with st.expander(f"📋 {assessment.get('name', 'Assessment')}"):
                                if assessment.get("description"):
                                    st.markdown(assessment["description"])
                                if assessment.get("test_type"):
                                    test_type = assessment["test_type"]
                                    label = test_type if isinstance(test_type, str) else ", ".join(test_type)
                                    st.markdown(f"**Type code(s):** {label}")
                                if assessment.get("url"):
                                    st.markdown(f"[Catalog link]({assessment['url']})")

                    if end_of_conversation:
                        st.info("Conversation ended. Clear history to start a new session.")

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": reply,
                            "recommendations": recommendations,
                        }
                    )
                except requests.RequestException as exc:
                    error_msg = (
                        f"Could not reach the API at {API_URL}/chat: {exc}\n\n"
                        "Ensure the backend is running (`uvicorn backend.main:app --reload`)."
                    )
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})


with st.sidebar:
    st.header("ℹ️ About")
    st.markdown(
        """
        Stateless conversational agent over the SHL individual assessment catalog.

        **Example flows**
        - Vague query → clarifying question
        - Detailed JD → recommendations
        - Follow-up → refined list
        - Compare named assessments
        """
    )

    st.header("🔧 Settings")
    api_url = st.text_input("API URL", value=API_URL)
    if api_url != API_URL:
        os.environ["RECOMMENDER_API_URL"] = _normalize_api_url(api_url)
        st.rerun()

    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
