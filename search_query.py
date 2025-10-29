
import base64
import json
import re
import os
import textwrap
from dotenv import load_dotenv
from quart import request, jsonify
import asyncpg

from azure.search.documents.aio import SearchClient as AsyncSearchClient
from azure.search.documents.models import VectorizableTextQuery
from azure.identity.aio import DefaultAzureCredential as AsyncDefaultAzureCredential
from openai import AsyncAzureOpenAI




load_dotenv()


# Async DB config
DB_CONFIG = {
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'database': os.getenv('DB_NAME'),
    'host': os.getenv('DB_HOST'),
    'port': os.getenv('DB_PORT')
}

async def connect_db():
    try:
        return await asyncpg.connect(**DB_CONFIG)
    except Exception as e:
        print(f"Database connection error: {e}")
        return None





async def load_fresh_config_from_db():
    """Load fresh configuration values from database on every call"""
    conn = None
    try:
        conn = await connect_db()
        if conn is None:
            print("Failed to connect to database for config loading")
            return None
            
        # Query to get the latest configuration
        query = """
        SELECT * FROM azaisearch_evolve_sandbox_settings 
        WHERE update_id = (SELECT MAX(update_id) FROM azaisearch_evolve_sandbox_settings)
        """
        
        row = await conn.fetchrow(query)
        
        if row:
            # Return configuration as dictionary instead of setting global variables
            config = {
                'azure_search_endpoint': row.get('azure_search_endpoint'),
                'azure_search_index_name': row.get('azure_search_index_name'),
                'current_prompt': row.get('current_prompt'),
                'openai_model_deployment_name': row.get('openai_model_deployment_name'),
                'openai_endpoint': row.get('openai_endpoint'),
                'openai_api_version': row.get('openai_api_version'),
                'openai_api_key': row.get('openai_api_key'),
                'azure_search_index_name_french': row.get('azure_search_index_name_french'),
                'current_prompt_french': row.get('current_prompt_french'),
                'semantic_configuration_name_english': row.get('semantic_configuration_name_english'),
                'semantic_configuration_name_french': row.get('semantic_configuration_name_french')
            }
            
            # Convert Decimal to float for temperature
            temp_value = row.get('openai_model_temperature')
            config['openai_model_temperature'] = float(temp_value) if temp_value is not None else None

            # DEBUG BOTH LANGUAGE CONFIGURATIONS
            print("=== FRESH CONFIG LOADED ===")
            print(f"azure_search_index_name: '{config['azure_search_index_name']}'")
            print(f"current_prompt length: {len(config['current_prompt']) if config['current_prompt'] else 'None'}")
            
            print("=== FRENCH CONFIG ===")  
            print(f"azure_search_index_name_french: '{config['azure_search_index_name_french']}'")
            print(f"current_prompt_french length: {len(config['current_prompt_french']) if config['current_prompt_french'] else 'None'}")
            
            print("Fresh configuration loaded successfully from database")
            return config
        else:
            print("No configuration found in database")
            return None
            
    except Exception as e:
        print(f"Error loading configuration from database: {e}")
        return None
    finally:
        if conn:
            await conn.close()














def safe_base64_decode(data):
    if data.startswith("https"):
        return data
    try:
        valid_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/="
        data = data.rstrip()
        while data and data[-1] not in valid_chars:
            data = data[:-1]
        while len(data) % 4 == 1:
            data = data[:-1]
        missing_padding = len(data) % 4
        if missing_padding:
            data += '=' * (4 - missing_padding)
        decoded = base64.b64decode(data).decode("utf-8", errors="ignore")
        decoded = decoded.strip().rstrip("\uFFFD").rstrip("?").strip()
        decoded = re.sub(r'\.(docx|pdf|pptx|xlsx)[0-9]+$', r'.\1', decoded, flags=re.IGNORECASE)
        return decoded
    except Exception as e:
        return f"[Invalid Base64] {data} - {str(e)}"

# -------------------------
# Async search and answer
# -------------------------
# -------------------------
# Async search and answer
# -------------------------
async def ask_query(user_query, user_id, conversation_store, clanguage="english"):





        # Load fresh configuration on every API call
    config = await load_fresh_config_from_db()
    if config is None:
        raise Exception("Failed to load configuration from database")









    # Async Azure credential
    credential = AsyncDefaultAzureCredential()

    AZURE_SEARCH_SERVICE = config['azure_search_endpoint']

    deployment_name = config['openai_model_deployment_name']


    # Language-based index and prompts
    if clanguage == "french_canadian":

        index_name = config['azure_search_index_name_french']
        semantic_config_name = config.get('semantic_configuration_name_french')
        answer_prompt_template = f"""{config['current_prompt_french']}"""


        followup_prompt_template = """En vous basant uniquement sur les extraits suivants, générez 3 questions de suivi que l’utilisateur pourrait poser.
N’utilisez que le contenu des sources. N’inventez pas de nouveaux faits.

Format :
Q1 : <question>
Q2 : <question>
Q3 : <question>

SOURCES :
{citations}

- Toutes les questions doivent être formulées en français canadien.
"""  
    else:
        index_name = config['azure_search_index_name']
        semantic_config_name = config.get('semantic_configuration_name_english')
        answer_prompt_template = f"""{config['current_prompt']}"""

        followup_prompt_template = """Based only on the following chunks of source material, generate 3 follow-up questions the user might ask.
Only use the content in the sources. Do not invent new facts.

Format:
Q1: <question>
Q2: <question>
Q3: <question>

SOURCES:
{citations}"""  

    openai_client = AsyncAzureOpenAI(
        api_version=config['openai_api_version'],

        azure_endpoint=config['openai_endpoint'],

        api_key=config['openai_api_key']

    )

    search_client = AsyncSearchClient(
        endpoint=AZURE_SEARCH_SERVICE,
        index_name=index_name,
        credential=credential
    )

    # -------------------------
    # Conversation tracking (MODIFIED)
    # -------------------------
    session_key = (user_id, clanguage)   # unique per user + language

    if session_key not in conversation_store:
        conversation_store[session_key] = {"history": [], "chat": ""}

    conversation_store[session_key]["history"].append(user_query)
    if len(conversation_store[session_key]["history"]) > 3:
        conversation_store[session_key]["history"] = conversation_store[session_key]["history"][-3:]

    history_queries = " ".join(conversation_store[session_key]["history"])
    conversation_history = conversation_store[session_key]["chat"]

    # -------------------------
    # Async fetch chunks
    # -------------------------
    async def fetch_chunks(query_text, k_value, start_index):
        vector_query = VectorizableTextQuery(text=query_text, k_nearest_neighbors=5, fields="text_vector")
        results = await search_client.search(
            search_text=query_text,
            vector_queries=[vector_query],
            select=["title", "chunk", "parent_id"],
            top=k_value,
            # semantic_configuration_name=f"{index_name}-semantic-configuration",
            semantic_configuration_name=semantic_config_name,
            query_type="semantic"
        )
        chunks, sources = [], []
        i = 0
        async for doc in results:
            title = doc.get("title", "N/A")
            chunk_content = doc.get("chunk", "N/A").replace("\n", " ").replace("\t", " ").strip()
            parent_id_decoded = safe_base64_decode(doc.get("parent_id", "Unknown Document"))
            chunk_id = start_index + i
            chunks.append({"id": chunk_id, "title": title, "chunk": chunk_content, "parent_id": parent_id_decoded})
            sources.append(f"Source ID: [{chunk_id}]\nContent: {chunk_content}\nDocument: {parent_id_decoded}")
            i += 1
        return chunks, sources

    history_chunks, history_sources = await fetch_chunks(history_queries, 5, 1)
    standalone_chunks, standalone_sources = await fetch_chunks(user_query, 5, 6)

    # Deduplicate chunks
    combined_chunks = history_chunks + standalone_chunks
    seen = set()
    all_chunks = []
    for chunk in combined_chunks:
        if chunk["chunk"] not in seen:
            seen.add(chunk["chunk"])
            all_chunks.append(chunk)
    sources_formatted = "\n\n---\n\n".join([f"Source ID: [{c['id']}]\nContent: {c['chunk']}\nDocument: {c['parent_id']}" for c in all_chunks])

    # -------------------------
    # Build AI prompt
    # -------------------------
    prompt = answer_prompt_template.format(conversation_history=conversation_history, sources=sources_formatted, query=user_query)

    response = await openai_client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=config['openai_model_deployment_name'],
        # temperature=0.7
        temperature= config['openai_model_temperature']

    )
    full_reply = response.choices[0].message.content.strip()

    # Standardize citations
    flat_ids = [int(n.strip()) for match in re.findall(r"\[(.*?)\]", full_reply) for n in match.split(",") if n.strip().isdigit()]
    unique_ids = []
    for i in flat_ids:
        if i not in unique_ids:
            unique_ids.append(i)
    id_mapping = {old_id: new_id+1 for new_id, old_id in enumerate(unique_ids)}

    def replace_citations(text, mapping):
        def repl(match):
            nums = [mapping.get(int(n.strip()), int(n.strip())) for n in match.group(1).split(",") if n.strip().isdigit()]
            return f"[{', '.join(map(str, sorted(set(nums))))}]"
        return re.sub(r"\[(.*?)\]", repl, text)

    ai_response = replace_citations(full_reply, id_mapping)

    # Update conversation and citations
    citations = []
    seen = set()
    for old_id in unique_ids:
        new_id = id_mapping[old_id]
        for chunk in all_chunks:
            if chunk["id"] == old_id and old_id not in seen:
                seen.add(old_id)
                updated_chunk = chunk.copy()
                updated_chunk["id"] = new_id
                citations.append(updated_chunk)

    conversation_store[session_key]["chat"] += f"\nUser: {user_query}\nAI: {ai_response}"

    # Follow-up questions
    follow_up_prompt = followup_prompt_template.format(citations=citations)
    follow_up_resp = await openai_client.chat.completions.create(
        messages=[{"role": "user", "content": follow_up_prompt}],
        model=deployment_name
    )
    follow_ups_raw = follow_up_resp.choices[0].message.content.strip()


        # Add this before the return statement
    print(f"=== FINAL DEBUG INFO ===")
    print(f"Using config azure_search_endpoint: {config['azure_search_endpoint']}")
    print(f"Using config openai_endpoint: {config['openai_endpoint']}")
    print(f"Using index_name: {index_name}")
    
    # Cleanup clients
    await search_client.close()
    await openai_client.close()
    await credential.close()

    return {"query": user_query, "ai_response": ai_response, "citations": citations, "follow_ups": follow_ups_raw}



