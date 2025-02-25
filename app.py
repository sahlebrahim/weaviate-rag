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
import cohere
# 1) Load environment variables
load_dotenv(override=True)
st.set_page_config(page_title="Sales Playbook Chat Assistant", layout="wide")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
WEAVIATE_URL = os.getenv("WEAVIATE_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
POSTGRES_URL = os.getenv("DATABASE_URL")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")  # new

weaviate_url = os.environ["WEAVIATE_URL"]
weaviate_api_key = os.environ["WEAVIATE_API_KEY"]
open_api_key = os.getenv("OPENAI_API_KEY")
client1 = OpenAI(api_key=open_api_key)

headers = {
    "X-OpenAI-Api-Key": open_api_key,
}

# 2) Connect to Weaviate Cloud


RAG_APP_PASSWORD = os.getenv("RAG_APP_PASSWORD", "mysecret")

if "auth_passed" not in st.session_state:
    st.session_state.auth_passed = False

# We'll do the login check inline:
if not st.session_state.auth_passed:
    st.title("Please Log In")
    pw = st.text_input("Enter Password", type="password")
    login_clicked = st.button("Login")
    if login_clicked:
        if pw == RAG_APP_PASSWORD:
            st.session_state.auth_passed = True
        else:
            st.error("Incorrect password.")

    # if still not authed => stop
    if not st.session_state.auth_passed:
        st.stop()

client = weaviate.connect_to_weaviate_cloud(
    cluster_url=weaviate_url,
    auth_credentials=Auth.api_key(weaviate_api_key),
    headers=headers,
)
co = cohere.Client(COHERE_API_KEY) if COHERE_API_KEY else None


# 3) Initialize session state for feedback text + correct text + a submission flag
if "feedback_text" not in st.session_state:
    st.session_state["feedback_text"] = ""
if "correct_text" not in st.session_state:
    st.session_state["correct_text"] = ""
if "feedback_submitted" not in st.session_state:
    st.session_state["feedback_submitted"] = False

# If feedback_submitted is True from a previous run, clear fields now and reset flag
if st.session_state["feedback_submitted"]:
    st.session_state["feedback_text"] = ""
    st.session_state["correct_text"] = ""
    st.session_state["feedback_submitted"] = False

# 4) Connect to Postgres
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

def save_query_to_db(query, response, comment, model, response_time, correct_response):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute('''
            insert into query_logs
              (query, response, comment, llm_model, response_time, correct_response)
            values (%s, %s, %s, %s, %s, %s)
        ''', (query, response, comment, model, response_time, correct_response))
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

# 5) Weaviate Hybrid Search
def search_weaviate_hybrid(query, top_k=10, alpha=0.5):
    """
    1) Embed the user query with OpenAI.
    2) Perform a hybrid search in Weaviate with BM25 + vector using that query embedding.
    3) Return chunk-like dicts: source, title, page_number, content.
    """
    start_time = time.perf_counter()
    embed_response = client1.embeddings.create(
        model="text-embedding-3-large",
        input=[query]
    )
    query_emb = embed_response.data[0].embedding
    embedding_time = time.perf_counter() - start_time

    start_weaviate = time.perf_counter()
    playbook_collection = client.collections.get("Refinedchunk1")

    response = playbook_collection.query.hybrid(
        query=query,
        vector=query_emb,
        alpha=alpha,
        limit=top_k
    )
    weaviate_time = time.perf_counter() - start_weaviate

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

# new: cohere rerank on top of weaviate results
def weaviate_plus_cohere_rerank(query, final_top_k=3, alpha=0.5, cohere_fetch=5):
    """
    1) weaviate hybrid search for e.g. 20 results
    2) cohere rerank them
    3) slice top final_top_k
    """
    # ensure we have a cohere client
    if not co:
        # fallback: just call the original weaviate search
        return search_weaviate_hybrid(query, top_k=final_top_k, alpha=alpha)

    # 1) get weaviate results
    weaviate_results, emb_time, weav_time = search_weaviate_hybrid(query, top_k=cohere_fetch, alpha=alpha)

    if not weaviate_results:
        return [], emb_time, weav_time

    # 2) cohere rerank
    docs = [r["content"] for r in weaviate_results]
    try:
        # model can be e.g. "rerank-english-v2.0" or "rerank-multilingual-v2.0"
        rerank_resp = co.rerank(
            model="rerank-english-v2.0",
            query=query,
            documents=docs,
            top_n=len(docs)
        )
    except Exception as e:
        st.warning(f"Cohere rerank error: {e}")
        # fallback: just return weaviate_results top final_top_k
        return weaviate_results[:final_top_k], emb_time, weav_time

    # reorder weaviate_results by cohere's relevance scores
    indexed_results = {i: weaviate_results[i] for i in range(len(weaviate_results))}
    reranked = []
    for item in rerank_resp.results:
        i = item.index
        score = item.relevance_score
        chunk = indexed_results[i]
        chunk["cohere_score"] = score
        reranked.append(chunk)

    reranked.sort(key=lambda x: x["cohere_score"], reverse=True)
    final_results = reranked[:final_top_k]
    return final_results, emb_time, weav_time

# 6) Build Prompt
def build_prompt(query, chunks):
    context = "\n\n".join(
        [f"source {c['source']} title {c['title']} page {c['page_number']} content\n{c['content']}"
         for c in chunks]
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
                    "content": "You are a helpful assistant. Use the provided document chunks to answer the user's question accurately. dont hallucinate however also be smart in looking through the contexts and answering accordingly"
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

# 7) UI
st.title("Sales Playbook Chat Assistant")
st.caption("Chat with your Sales Playbook using Weaviate Hybrid Search + OpenAI Embeddings/LLM")

model_option = st.sidebar.selectbox(
    "Select OpenAI Chat Model",
    options=["gpt-4o-mini", "gpt-4o"],
    index=0
)

# Chat message history
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Logs viewer
with st.expander("View Query Logs (Live)"):
    if st.button("Refresh Logs"):
        pass  # triggers rerun

    logs = get_all_logs()
    if not logs:
        st.info("No query logs available.")
    else:
        df = pd.DataFrame([dict(r) for r in logs])
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

    with st.spinner("Searching Weaviate..."):
        retrieved_chunks, embedding_time, weaviate_time = search_weaviate_hybrid(user_input, top_k=3, alpha=0.5)

    prompt_start = time.perf_counter()
    prompt = build_prompt(user_input, retrieved_chunks)
    prompt_build_time = time.perf_counter() - prompt_start

    model_start = time.perf_counter()
    assistant_response = query_openai_chat(model_option, prompt)
    model_time = time.perf_counter() - model_start
    overall_time = time.perf_counter() - start_time

    st.session_state.messages.append({"role": "assistant", "content": assistant_response})
    with st.chat_message("assistant"):
        st.markdown(assistant_response)

    with st.expander("Retrieved Context from Weaviate"):
        for idx, chunk in enumerate(retrieved_chunks, start=1):
            st.write(f"Chunk {idx} - Title: {chunk['title']}, Page: {chunk['page_number']}")
            st.text_area("Content", chunk["content"], height=100)

    with st.expander("Timing Details"):
        st.write(f"Embedding time: {embedding_time:.2f} sec")
        st.write(f"Weaviate search time: {weaviate_time:.2f} sec")
        st.write(f"Prompt build time: {prompt_build_time:.2f} sec")
        st.write(f"LLM query time: {model_time:.2f} sec")
        st.write(f"Overall time: {overall_time:.2f} sec")

    st.session_state.latest_query = user_input
    st.session_state.latest_response = assistant_response
    st.session_state.latest_model = model_option
    st.session_state.latest_response_time = overall_time

# 8) Feedback form
if "latest_query" in st.session_state and "latest_response" in st.session_state:
    with st.expander("Provide feedback on the latest response"):
        with st.form("feedback_form"):
            feedback = st.text_area(
                "Your comment or feedback on this response",
                key="feedback_text"
            )
            correct_resp = st.text_area(
                "If the assistant was incorrect, provide the correct response",
                key="correct_text"
            )
            submitted = st.form_submit_button("Submit feedback")

            if submitted:
                

                # Save to DB
                save_query_to_db(
                    st.session_state.latest_query,
                    st.session_state.latest_response,
                    feedback,
                    st.session_state.latest_model,
                    st.session_state.latest_response_time,
                    correct_response=correct_resp
                )

                st.success("Feedback saved. Fields will be cleared next run!")

                # Mark feedback_submitted and rerun
                st.session_state["feedback_submitted"] = True
                st.rerun()




#### undo till this.