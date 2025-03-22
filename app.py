import os
import time
import uuid
import streamlit as st
import pandas as pd
import json
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
import io
import re
import base64
from PIL import Image
import anthropic
from azure.identity import ClientSecretCredential
from azure.storage.blob import BlobServiceClient

# 1) Load environment variables
load_dotenv(override=True)
st.set_page_config(page_title="Sales Playbook Chat Assistant", layout="wide")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
WEAVIATE_URL = os.getenv("WEAVIATE_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
POSTGRES_URL = os.getenv("DATABASE_URL")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Azure Storage configuration
AZURE_CLIENT_ID = os.getenv("AZURE_CLIENT_ID")
AZURE_TENANT_ID = os.getenv("AZURE_TENANT_ID")
AZURE_CLIENT_SECRET = os.getenv("AZURE_CLIENT_SECRET")
AZURE_STORAGE_URL = os.getenv("AZURE_STORAGE_URL")

# LLM pricing information ($ per million tokens)
ANTHROPIC_PRICING = {
    "claude-3-7-sonnet-20250219": {"input": 3.00, "output": 15.00},
    "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
    "claude-3-5-sonnet-20240620": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku-20241022": {"input": 0.80, "output": 4.00},
    "claude-3-opus-20240229": {"input": 15.00, "output": 75.00},
    "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25}
}

OPENAI_PRICING = {
    "gpt-4o": {"input": 5.00, "output": 15.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60}
}

weaviate_url = os.environ["WEAVIATE_URL"]
weaviate_api_key = os.environ["WEAVIATE_API_KEY"]
open_api_key = os.getenv("OPENAI_API_KEY")
client1 = OpenAI(api_key=open_api_key)

# Initialize Azure Blob Storage client (if credentials are available)
azure_images_available = all([AZURE_CLIENT_ID, AZURE_TENANT_ID, AZURE_CLIENT_SECRET, AZURE_STORAGE_URL])
container_client = None

if azure_images_available:
    try:
        # Set up Azure credentials
        credentials = ClientSecretCredential(
            client_id=AZURE_CLIENT_ID,
            client_secret=AZURE_CLIENT_SECRET,
            tenant_id=AZURE_TENANT_ID
        )
        
        # Create Blob Service Client
        blob_service_client = BlobServiceClient(account_url=AZURE_STORAGE_URL, credential=credentials)
        
        # Set container and folder name for images
        container_name = 'data'
        folder = 'sales playbook images/'
        
        # Get container client for fetching images
        container_client = blob_service_client.get_container_client(container=container_name)
        st.sidebar.success("✅ Azure Blob Storage connected for images")
    except Exception as e:
        st.sidebar.error(f"❌ Azure Storage connection error: {str(e)}")
        azure_images_available = False
else:
    st.sidebar.warning("⚠️ Azure Storage credentials not found. Image display will be unavailable.")

headers = {
    "X-OpenAI-Api-Key": open_api_key,
}

# Initialize Weaviate client
client = weaviate.connect_to_weaviate_cloud(
    cluster_url=weaviate_url,
    auth_credentials=Auth.api_key(weaviate_api_key),
    headers=headers,
)
co = cohere.Client(COHERE_API_KEY) if COHERE_API_KEY else None

# Initialize Anthropic client if key is available
anthropic_client = None
if ANTHROPIC_API_KEY:
    try:
        anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        st.sidebar.success("✅ Anthropic API connected")
    except Exception as e:
        st.sidebar.error(f"❌ Anthropic API error: {str(e)}")

# 2) Connect to Weaviate Cloud
RAG_APP_PASSWORD = os.getenv("RAG_APP_PASSWORD", "mysecret")

# Initialize session state
if "auth_passed" not in st.session_state:
    st.session_state.auth_passed = False

# Login handler function
def handle_login():
    if st.session_state.password_input == RAG_APP_PASSWORD:
        st.session_state.auth_passed = True
    else:
        st.session_state.auth_error = True

# Authentication check
if not st.session_state.auth_passed:
    st.title("Please Log In")
    
    # Using on_change callback with a key to prevent UI persistence
    st.text_input("Enter Password", type="password", key="password_input", on_change=handle_login)
    st.button("Login", on_click=handle_login)
    
    if "auth_error" in st.session_state and st.session_state.auth_error:
        st.error("Incorrect password.")
        st.session_state.auth_error = False
    
    # Stop the app if not authenticated
    st.stop()

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
        ''', (
            query, response, comment, model, response_time, correct_response
        ))
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
    3) Return chunk-like dicts with the correct schema fields.
    """
    start_time = time.perf_counter()
    embed_response = client1.embeddings.create(
        model="text-embedding-3-large",
        input=[query]
    )
    query_emb = embed_response.data[0].embedding
    embedding_time = time.perf_counter() - start_time

    start_weaviate = time.perf_counter()
    playbook_collection = client.collections.get("Refinedchunk3")

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
            # Create chunk with the correct schema keys
            chunk = {
                #"source": props.get("source", "unknown"),
                "heading": props.get("heading", "unknown"),
                "level": props.get("level", ""),
                "page_number": props.get("page", ""),  # Note the key is "page" not "page_number"
                #"content": props.get("text", "no content available"),
                "summary": props.get("summary", ""),
                "parent_heading": props.get("parent_heading", None),
                "children_headings": props.get("children_headings", []),
            }
            
            # Handle the images_included field - parse from JSON string if it exists
            images_included = props.get("images_included", "[]")
            try:
                if isinstance(images_included, str):
                    chunk["images_included"] = json.loads(images_included)
                else:
                    chunk["images_included"] = images_included
            except json.JSONDecodeError:
                chunk["images_included"] = []
            print(chunk["images_included"])
            chunks.append(chunk)

    return chunks, embedding_time, weaviate_time

# Extract images from chunks
def extract_image_references_from_chunks(chunks):
    """
    Extract all unique image references from chunks based on the images_included field
    
    Args:
        chunks (list): List of chunk dictionaries from Weaviate
        
    Returns:
        list: List of unique image filenames
    """
    all_images = []
    
    for chunk in chunks:
        # Extract images from the images_included field
        if "images_included" in chunk and chunk["images_included"]:
            if isinstance(chunk["images_included"], list):
                all_images.extend(chunk["images_included"])
            elif isinstance(chunk["images_included"], str):
                try:
                    # In case it's a JSON string that wasn't parsed properly
                    img_list = json.loads(chunk["images_included"])
                    if isinstance(img_list, list):
                        all_images.extend(img_list)
                except json.JSONDecodeError:
                    pass
    
    # Remove duplicates and return
    return list(set(all_images))

# Cohere rerank on top of weaviate results
def weaviate_plus_cohere_rerank(query, final_top_k=3, alpha=0.5, cohere_fetch=10):
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
    docs = [r["summary"] for r in weaviate_results]
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
    #print(final_results)
    return final_results, emb_time, weav_time

# 6) Build Prompt
def build_prompt(query, chunks):
    """
    Build a prompt that includes chunk text with better instructions for image inclusion
    """
    context_parts = []
    
    # Extract all images referenced in the chunks for context
    referenced_images = extract_image_references_from_chunks(chunks)
    
    for c in chunks:
        # If there's no cohere_score, default to 0.0
        score = c.get("cohere_score", 0.0)
        
        # Add images available in this chunk, if any
        images_info = ""
        if c.get("images_included") and len(c.get("images_included", [])) > 0:
            images = c.get("images_included")
            if isinstance(images, list) and images:
                images_info = f"Images in this chunk: {', '.join(images)}\n"
        
        context_parts.append(
            #f"Source: {c.get('source','unknown')}, "
            f"Title: {c.get('heading','')}, "
            f"Page: {c.get('page_number','')}, "
            f"Cohere Score (for reference only): {score:.2f}\n"
            f"{images_info}"
            f"{c.get('summary','')}"
        )
    
    context = "\n\n".join(context_parts)
    
    # Create a more detailed prompt about image handling
    prompt = f"""
Use the following document excerpts to answer the user's query.
Cohere scores are included only as a reference. If a chunk is not relevant, do not include it in your final answer.

{context}

---
User Query: {query}

Answer the query based only on the provided content.
If the answer cannot be determined from these excerpts, say 'not available from the provided excerpts'.

IMPORTANT INSTRUCTIONS FOR IMAGE HANDLING:
1. The document excerpts may contain references to images which are available for display.
2. Referenced images: {", ".join(referenced_images) if referenced_images else "None"}
3. When an image is relevant to your answer, include it by placing its exact filename in curly braces
   like this: {{img-72.jpeg}}
4. Only include images that are actually mentioned in the chunks above.
5. Place image references at logical points in your answer where they add value - typically after
   you've discussed the relevant information that the image illustrates.
6. DO NOT create or hallucinate image filenames that aren't explicitly mentioned in the chunks.
7. DO NOT modify the image filenames in any way.
8. Multiple images may be relevant - include each one where appropriate in your response.

Your answer should be comprehensive, accurate, and should intelligently incorporate relevant images
where they help illustrate your points.
"""
    return prompt.strip()

# 7) Image Display Function for Streamlit
def display_inline_images_streamlit(text_response):
    """
    Process a text response containing image references like {img-72.jpeg} and 
    replace them with actual images from Azure Blob Storage for Streamlit display.
    
    Args:
        text_response (str): Text containing image placeholders
    """
    if not azure_images_available or container_client is None:
        # If Azure Storage isn't configured, just return the text as is
        st.write(text_response)
        return
    
    # Extract image references
    img_pattern = r'\{([^{}]+\.(jpeg|jpg|png|gif|bmp))\}'
    
    # Find all matches and their positions
    matches = list(re.finditer(img_pattern, text_response))
    
    if not matches:
        # No image references found, return the original text
        st.write(text_response)
        return
    
    # Process text and display images inline
    last_end = 0
    for match in matches:
        img_name = match.group(1)  # The filename without brackets
        start, end = match.span()  # Position of the placeholder in text
        
        # Display text before the image
        if start > last_end:
            st.write(text_response[last_end:start])
        
        # Get and display the image
        try:
            # Construct the full blob path
            blob_path = f"{folder}{img_name}"
            
            # Get a blob client for this specific image
            blob_client = container_client.get_blob_client(blob=blob_path)
            
            # Download the blob content
            download_stream = blob_client.download_blob()
            image_data = download_stream.readall()
            
            # Create a PIL Image
            image = Image.open(io.BytesIO(image_data))
            
            # Display the image in Streamlit (using use_container_width instead of use_column_width)
            st.image(image, caption=f"Image: {img_name}", use_container_width=True)
            
        except Exception as e:
            # If image can't be loaded, show error message
            st.error(f"❌ Could not load image {img_name}: {str(e)}")
        
        last_end = end
    
    # Display any remaining text after the last image
    if last_end < len(text_response):
        st.write(text_response[last_end:])
# 8) LLM Cost Calculation
def calculate_openai_cost(model, response_obj):
    """
    Calculate the cost of an OpenAI API call using the usage data from response.
    """
    # Extract token counts from response
    if hasattr(response_obj, 'usage'):
        input_tokens = response_obj.usage.prompt_tokens
        output_tokens = response_obj.usage.completion_tokens
    else:
        # Fallback if structure is different
        st.warning("Could not extract token counts from OpenAI response")
        return {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "input_cost": 0,
            "output_cost": 0,
            "total_cost": 0
        }
    
    # Get pricing for this model
    pricing = OPENAI_PRICING.get(model, {"input": 0.15, "output": 0.60})
    
    # Calculate costs
    input_cost = (pricing["input"] * input_tokens) / 1_000_000
    output_cost = (pricing["output"] * output_tokens) / 1_000_000
    total_cost = input_cost + output_cost
    
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost
    }

def calculate_anthropic_cost(model, response_obj):
    """
    Calculate the cost of an Anthropic API call using the usage data from response.
    """
    # Extract token counts from response
    if hasattr(response_obj, 'usage'):
        input_tokens = response_obj.usage.input_tokens
        output_tokens = response_obj.usage.output_tokens
    else:
        # Fallback if structure is different
        st.warning("Could not extract token counts from Anthropic response")
        return {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "input_cost": 0,
            "output_cost": 0,
            "total_cost": 0
        }
    
    # Get pricing for this model
    pricing = ANTHROPIC_PRICING.get(model, {"input": 3.0, "output": 15.0})
    
    # Calculate costs
    input_cost = (pricing["input"] * input_tokens) / 1_000_000
    output_cost = (pricing["output"] * output_tokens) / 1_000_000
    total_cost = input_cost + output_cost
    
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost
    }

# 9) LLM Query Functions
def query_openai_chat(model, prompt):
    """Query OpenAI with improved instructions for image handling"""
    try:
        response = client1.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant answering questions based on provided document excerpts. "
                        "The excerpts may contain references to images that can be included in your response. "
                        "Follow these guidelines for image handling:\n\n"
                        "1. Only reference images that are explicitly mentioned in the excerpts.\n"
                        "2. Include relevant images by placing their exact filename in curly braces: {img-72.jpeg}\n"
                        "3. Place images at logical points in your response where they illustrate your answer.\n"
                        "4. Never hallucinate or create image filenames not mentioned in the excerpts.\n"
                        "5. If multiple images are relevant, include each one where appropriate.\n\n"
                        "The text that appears in curly braces must exactly match the image filenames from the excerpts."
                    )
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=2000,
            timeout=15,
        )
        return response.choices[0].message.content, response
    except Exception as e:
        st.error(f"OpenAI API error: {e}")
        return "OpenAI API error, please try again.", None

def query_anthropic_chat(model, prompt):
    """Query Anthropic Claude with instructions for proper image handling"""
    #print(prompt)
    if not anthropic_client:
        return "Anthropic API key not configured", None
    
    try:
        message = anthropic_client.messages.create(
            model=model,
            max_tokens=2000,
            temperature=0,
            system=(
                "You are a helpful RAG assistant answering questions based on provided document excerpts. "
                "Follow these guidelines for handling images in your responses:\n\n"
                "1. The document excerpts may mention image files that are available for display.\n"
                "2. When an image is relevant to your answer, include it by placing its exact filename in curly braces: {img-72.jpeg}\n"
                "3. Only include images that are explicitly mentioned in the provided excerpts.\n"
                "4. Position image references at logical points in your answer where they add value.\n"
                "5. Never modify image filenames or create new ones.\n"
                "6. Multiple images may be relevant - use each one appropriately.\n\n"
                "Your answer should be comprehensive, accurate, and should intelligently incorporate relevant images."
            ),
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        
        # Extract text from the response
        if hasattr(message.content[0], 'text'):
            response_text = message.content[0].text
        else:
            # Fallback if the structure is different
            response_text = str(message.content)
            
        return response_text, message
    except Exception as e:
        st.error(f"Anthropic API error: {e}")
        return "Anthropic API error, please try again.", None

# 10) UI
st.title("Sales Playbook Chat Assistant")
st.caption("Chat with your Sales Playbook using Weaviate Hybrid Search + LLM with image support")

# Add model selection for both OpenAI and Anthropic
model_provider = st.sidebar.radio(
    "Select Model Provider", 
    options=["OpenAI", "Anthropic"],
    index=0 if not anthropic_client else 0
)

if model_provider == "OpenAI":
    model_option = st.sidebar.selectbox(
        "Select OpenAI Model",
        options=["gpt-4o-mini", "gpt-4o"],
        index=0
    )
else:
    model_option = st.sidebar.selectbox(
        "Select Anthropic Model",
        options=[
            "claude-3-7-sonnet-20250219",
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022",
            "claude-3-opus-20240229",
            "claude-3-haiku-20240307"
        ],
        index=0
    )

# Chat message history
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        # Check if this is an assistant message that might contain image references
        if msg["role"] == "assistant" and azure_images_available:
            display_inline_images_streamlit(msg["content"])
        else:
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
        retrieved_chunks, embedding_time, weaviate_time = weaviate_plus_cohere_rerank(user_input, final_top_k=3, alpha=0.5)

    prompt_start = time.perf_counter()
    prompt = build_prompt(user_input, retrieved_chunks)
    prompt_build_time = time.perf_counter() - prompt_start

    model_start = time.perf_counter()
    if model_provider == "OpenAI":
        assistant_response, response_obj = query_openai_chat(model_option, prompt)
        if response_obj:
            cost_info = calculate_openai_cost(model_option, response_obj)
        else:
            # Fallback if no response object
            cost_info = {
                "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
                "input_cost": 0, "output_cost": 0, "total_cost": 0
            }
    else:
        assistant_response, response_obj = query_anthropic_chat(model_option, prompt)
        if response_obj:
            cost_info = calculate_anthropic_cost(model_option, response_obj)
        else:
            # Fallback if no response object
            cost_info = {
                "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
                "input_cost": 0, "output_cost": 0, "total_cost": 0
            }
    
    model_time = time.perf_counter() - model_start
    overall_time = time.perf_counter() - start_time

    st.session_state.messages.append({"role": "assistant", "content": assistant_response})
    with st.chat_message("assistant"):
        if azure_images_available:
            display_inline_images_streamlit(assistant_response)
        else:
            st.markdown(assistant_response)

    # Store pricing info in session state
    st.session_state.latest_cost_info = cost_info

    # Add detail expanders
    with st.expander("Retrieved Context from Weaviate"):
        if not retrieved_chunks:
            st.info("No context retrieved from Weaviate for this query.")
        else:
            for idx, chunk in enumerate(retrieved_chunks, start=1):
                st.markdown(f"### Chunk {idx} - Title: {chunk.get('heading', '')}, Page: {chunk.get('page_number', '')}")
            
            # Display images_included if available
                if chunk.get("images_included") and len(chunk.get("images_included", [])) > 0:
                    st.markdown(f"**Images referenced:** {', '.join(chunk['images_included'])}")
            
            # Use a read-only display for content instead of text_area
                st.markdown("**Content:**")
                content = chunk.get("summary", "No content available")
                st.markdown(f"""```
    {content}
    ```""")
            
            # Add a separator between chunks
                if idx < len(retrieved_chunks):
                    st.markdown("---")

    with st.expander("Timing and Cost Details"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### Timing")
            st.write(f"Embedding time: {embedding_time:.2f} sec")
            st.write(f"Weaviate search time: {weaviate_time:.2f} sec")
            st.write(f"Prompt build time: {prompt_build_time:.2f} sec")
            st.write(f"LLM query time: {model_time:.2f} sec")
            st.write(f"Overall time: {overall_time:.2f} sec")
            
        with col2:
            st.write("### Cost")
            st.write(f"Model: {model_option}")
            st.write(f"Input tokens: {cost_info['input_tokens']:,}")
            st.write(f"Output tokens: {cost_info['output_tokens']:,}")
            st.write(f"Total tokens: {cost_info['total_tokens']:,}")
            st.write(f"Input cost: ${cost_info['input_cost']:.6f}")
            st.write(f"Output cost: ${cost_info['output_cost']:.6f}")
            st.write(f"Total cost: ${cost_info['total_cost']:.6f}")

    # Add a visualization for cost breakdown
    with st.expander("Cost Visualization"):
        cost_data = {
            "Category": ["Input", "Output"],
            "Tokens": [cost_info["input_tokens"], cost_info["output_tokens"]],
            "Cost": [cost_info["input_cost"], cost_info["output_cost"]]
        }
        
        # Simple token count visualization
        st.write("### Token Usage")
        st.bar_chart(pd.DataFrame(cost_data).set_index("Category")["Tokens"])
        
        # Cost visualization
        st.write("### Cost Breakdown")
        st.bar_chart(pd.DataFrame(cost_data).set_index("Category")["Cost"])
        
        # Display pricing table
        st.write("### Model Pricing ($ per million tokens)")
        if model_provider == "Anthropic":
            pricing_df = pd.DataFrame(ANTHROPIC_PRICING).T
            pricing_df.index.name = "Model"
            st.dataframe(pricing_df)
        else:
            pricing_df = pd.DataFrame(OPENAI_PRICING).T
            pricing_df.index.name = "Model"
            st.dataframe(pricing_df)

    # Save feedback in session state
    st.session_state.latest_query = user_input
    st.session_state.latest_response = assistant_response
    st.session_state.latest_model = f"{model_provider}-{model_option}"
    st.session_state.latest_response_time = overall_time
    
# 11) Feedback form
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

# Properly close Weaviate connection when the app exits
def cleanup():
    if 'client' in globals():
        client.close()

# Register the cleanup handler
import atexit
atexit.register(cleanup)