import os
import time
import uuid
import streamlit as st
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
import psycopg2
import psycopg2.extras
import openai
import weaviate.classes as wvc
import weaviate
from weaviate.classes.init import Auth
from openai import OpenAI

load_dotenv(override=True)
st.set_page_config(page_title="Sales Playbook Chat Assistant", layout="wide")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
WEAVIATE_URL = os.getenv("WEAVIATE_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
POSTGRES_URL = os.getenv("DATABASE_URL")

weaviate_url = os.environ["WEAVIATE_URL"]
weaviate_api_key = os.environ["WEAVIATE_API_KEY"]
open_api_key = os.getenv("OPENAI_API_KEY")
client1 = OpenAI(api_key=open_api_key)





headers = {
    "X-OpenAI-Api-Key": open_api_key,
}


# Connect to Weaviate Cloud
client = weaviate.connect_to_weaviate_cloud(
    cluster_url=weaviate_url,
    auth_credentials=Auth.api_key(weaviate_api_key),
    headers = headers
)

# Connect to Postgres
def get_db_connection():
    conn = psycopg2.connect(POSTGRES_URL, sslmode="require")
    return conn

def init_db():
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute('''
        create table if not exists query_logs (
    id serial primary key,
    query text,
    response text,
    comment text,
    correct_response text,     
    llm_model text,
    response_time float,
    timestamp timestamptz default now()
)
        ''')
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        st.error(f"Database init error: {e}")

init_db()

def save_query_to_db(query, response, comment, model, response_time):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute('''
            insert into query_logs
              (query, response, comment, llm_model, response_time, correct_response)
            values (%s, %s, %s, %s, %s, %s)
        ''', (query, response, comment, model, response_time))
        conn.commit()
        cur.close()
        conn.close()
        st.success("Query saved to logs!")
    except Exception as e:
        st.error(f"Database insert error: {e}")

def get_all_logs():
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cur.execute('select * from query_logs order by timestamp desc')
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return rows
    except Exception as e:
        st.error(f"Database retrieval error: {e}")
        return []

# 4) Weaviate Hybrid Search with OpenAI Embeddings
def search_weaviate_hybrid(query, top_k=3, alpha=0.5):
    """
    1) Embed the user query with OpenAI.
    2) Perform a hybrid search in Weaviate with BM25 + vector using that query embedding.
    3) Return chunk-like dicts: source, title, page_number, content.
    """
    start_time = time.perf_counter()

    # 1) Embed the query with OpenAI
    embed_response = client1.embeddings.create(
        model="text-embedding-3-large",  # or your desired embedding model
        input=[query]
    )
    query_emb = embed_response.data[0].embedding
    embedding_time = time.perf_counter() - start_time

    # 2) Hybrid search in Weaviate
    start_weaviate = time.perf_counter()

    # Suppose your collection name is "PlaybookChunk"
    # If you have a different name, adjust below:
    playbook_collection = client.collections.get("PlaybookChunk")

    response = playbook_collection.query.hybrid(
        query=query,         # used for BM25
        vector=query_emb,    # used for vector portion
        alpha=alpha,
        limit=top_k
    )

    weaviate_time = time.perf_counter() - start_weaviate

    # 3) Convert results to chunk dicts
    chunks = []
    if response.objects:
        for obj in response.objects:
            props = obj.properties
            chunk = {
                "source": props.get("source", "unknown"),
                "title": props.get("heading", ""),
                "page_number": props.get("page", ""),
                "content": props.get("text", "no content available")
            }
            chunks.append(chunk)

    return chunks, embedding_time, weaviate_time

# 5) Build Prompt + Query OpenAI Chat
def build_prompt(query, chunks):
    context = "\n\n".join(
        [
            f"source {c['source']} title {c['title']} page {c['page_number']} content\n{c['content']}"
            for c in chunks
        ]
    )
    prompt = f"""
use the following document excerpts to answer the user's query:

{context}

---
user query: {query}

answer the query based only on the provided content. if the answer cannot be determined from the content, state 'not available from the provided excerpts'.
"""
    return prompt.strip()

def query_openai_chat(model, prompt):
    try:
        response = client1.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Use the provided document chunks to answer the user's question accurately."
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            timeout=15,
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"OpenAI API error: {e}")
        return "OpenAI API error, please try again."


# Streamlit UI

st.title("Sales Playbook Chat Assistant")
st.caption("Chat with your Sales Playbook using Weaviate Hybrid Search + OpenAI Embeddings/LLM")

model_option = st.sidebar.selectbox(
    "Select OpenAI Chat Model",
    options=["gpt-4o-mini", "gpt-4o"],  # or your custom model names
    index=0
)

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display existing messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Logs viewer
with st.expander("View Query Logs (Live)"):
    if st.button("Refresh Logs"):
        pass  # triggers a rerun

    logs = get_all_logs()
    if not logs:
        st.info("No query logs available.")
    else:
        df = pd.DataFrame([dict(r) for r in logs])
        # optional filter
        search_term = st.text_input("Filter logs by query text")
        if search_term:
            df = df[df["query"].str.contains(search_term, case=False, na=False)]
        st.dataframe(df)

# Main user input
if user_input := st.chat_input("Type your query..."):
    start_time = time.perf_counter()

    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 1) Retrieve chunks via Weaviate Hybrid
    with st.spinner("Searching Weaviate..."):
        retrieved_chunks, embedding_time, weaviate_time = search_weaviate_hybrid(user_input, top_k=3, alpha=0.5)

    # 2) Build prompt
    prompt_start = time.perf_counter()
    prompt = build_prompt(user_input, retrieved_chunks)
    prompt_build_time = time.perf_counter() - prompt_start

    # 3) Query OpenAI Chat
    model_start = time.perf_counter()
    assistant_response = query_openai_chat(model_option, prompt)
    model_time = time.perf_counter() - model_start

    overall_time = time.perf_counter() - start_time

    # Display assistant response
    st.session_state.messages.append({"role": "assistant", "content": assistant_response})
    with st.chat_message("assistant"):
        st.markdown(assistant_response)

    # Show retrieved chunks
    with st.expander("Retrieved Context from Weaviate"):
        for idx, chunk in enumerate(retrieved_chunks, start=1):
            st.write(f"Chunk {idx} - Title: {chunk['title']}, Page: {chunk['page_number']}")
            st.text_area("Content", chunk["content"], height=100)

    # Show timing details
    with st.expander("Timing Details"):
        st.write(f"Embedding time: {embedding_time:.2f} sec")
        st.write(f"Weaviate search time: {weaviate_time:.2f} sec")
        st.write(f"Prompt build time: {prompt_build_time:.2f} sec")
        st.write(f"LLM query time: {model_time:.2f} sec")
        st.write(f"Overall time: {overall_time:.2f} sec")

    # 4) Save to DB
    st.session_state.latest_query = user_input
    st.session_state.latest_response = assistant_response
    st.session_state.latest_model = model_option
    st.session_state.latest_response_time = overall_time
#feedbackform
def clear_feedback_text():
    st.session_state["feedback_text"] = ""
    st.session_state["correct_text"] = ""
if "latest_query" in st.session_state and "latest_response" in st.session_state:
    with st.expander("provide feedback on the latest response"):
    with st.form("feedback_form"):
        feedback = st.text_area("Your comment or feedback on this response", key="feedback_text")
        correct_resp = st.text_area("If the assistant was incorrect, provide the correct response", key="correct_text")
        submitted = st.form_submit_button("submit feedback", on_click=clear_feedback_text)
        if submitted:
            save_query_to_db(
                st.session_state.latest_query,
                st.session_state.latest_response,
                feedback,
                st.session_state.latest_model,
                st.session_state.latest_response_time,
                correct_response=correct_resp
            )
                st.success("feedback saved thank you for helping us improve")