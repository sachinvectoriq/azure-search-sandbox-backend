# app.py
from quart import Quart, request, jsonify
# from saml import saml_login, saml_callback, extract_token
import os

# Import the refactored function
from search_query import ask_query # Renamed to avoid conflict with route name

# --- In-memory store for conversation history (TEMPORARY - NOT for production) ---
# This will not persist across restarts or multiple Flask processes/instances.
user_conversations = {}  # Define the single source of truth here

# Initialize Quart app
app = Quart(__name__)
# app.config["SAML_PATH"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saml")
# app.config["SECRET_KEY"] = os.getenv('JWT_SECRET_KEY')  # Replace with hardcoded key or securely read it, as you prefer.

# ---- Basic route ----
@app.route('/')
async def hello():
    return 'Hello!'

# # ---- SAML routes ----
# @app.route('/saml/login')
# async def login(): # Changed to async def
#     return await saml_login(app.config["SAML_PATH"]) # Added await

# @app.route('/saml/callback', methods=['POST'])
# async def login_callback(): # Changed to async def
#     return await saml_callback(app.config["SAML_PATH"]) # Added await

# @app.route('/saml/token/extract', methods=['POST'])
# async def func_get_data_from_token(): # Changed to async def
#     return await extract_token() # Added await

# ---- Async ask route ----
@app.route("/ask", methods=["POST"])
async def ask():
    data = await request.get_json()
    if not data or "query" not in data:
        return jsonify({"error": "Missing 'query' in request body"}), 400

    user_id = data.get("user_id", "default_user")
    clanguage = data.get("clanguage", "english").strip().lower()  # ✅ normalize

    try:
        result = await ask_query(data["query"], user_id, user_conversations, clanguage)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---- All other sync routes ----

# from user_login_log import log_user
# @app.route('/log/user', methods=['POST'])
# async def call_log_user():
#     return await log_user()


# from feedback import submit_feedback
# @app.route('/feedback', methods=['POST'])
# async def call_submit_feedback():
#     return await submit_feedback()


# from logging_chat import log_query
# @app.route('/log', methods=['POST'])
# async def call_log_query():
#     return await log_query()


from get_settings import get_settings
@app.route('/get_settings', methods=['GET'])
async def call_get_settings():
    return await get_settings()
    

from update_settings import update_settings
@app.route('/update_settings', methods=['POST'])
async def call_update_settings():
    return await update_settings()

# ---- Optional sync test route ----
#@app.route("/ping", methods=["GET"])
#def ping():
    #return "pong"

# ---- Main Entry Point ----
if __name__ == "__main__":
    # For local development, use uvicorn directly
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)  # Removed workers=1 for local dev simplicity
