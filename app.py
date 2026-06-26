"""
Streamlit UI for the gift recommender agent.
Run: streamlit run app.py
"""

import streamlit as st
from agent.agent import agent

st.set_page_config(
    page_title="Gift Recommender",
    page_icon="🎁",
    layout="wide",
)

st.title("🎁 Gift Recommender")
st.caption("Describe the person you're buying for — get 3 personalised picks from the real product catalogue.")

# Sidebar: agent reasoning trace
with st.sidebar:
    st.header("🤖 Agent Trace")
    st.caption("See which tools the agent called and why.")
    trace_placeholder = st.empty()

# Example prompts
with st.expander("💡 Try these examples"):
    examples = [
        "My brother is 24, obsessed with anime and PC gaming, already has a headset. Budget around $200.",
        "Something for my mum who just got into birdwatching. She's not very techy. Under $100.",
        "Dad who loves cooking, just bought a new TV. Looking for something for the kitchen. $50–$150.",
        "Secret Santa for a coworker I don't know well. Safe choice, $20–$50.",
        "Best friend turning 30, really into fitness and hates clutter. Under $120.",
    ]
    for ex in examples:
        if st.button(ex, key=ex):
            st.session_state["prefill"] = ex

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input
prefill = st.session_state.pop("prefill", "")
prompt = st.chat_input(
    "e.g. My sister is 28, loves yoga and true crime podcasts, just moved into her first apartment...",
    key="chat_input",
) or prefill

if prompt:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Build conversation history for agent
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages
    ]

    with st.chat_message("assistant"):
        with st.spinner("Searching catalogue..."):
            result = agent.invoke({"messages": history})

        response = result["messages"][-1].content
        st.markdown(response)

        # Show tool trace in sidebar
        tool_steps = [
            m for m in result["messages"]
            if hasattr(m, "tool_calls") and m.tool_calls
        ]
        if tool_steps:
            with trace_placeholder.container():
                for msg in tool_steps:
                    for call in msg.tool_calls:
                        with st.expander(f"🔧 {call['name']}"):
                            st.json(call["args"])

    st.session_state.messages.append({"role": "assistant", "content": response})
