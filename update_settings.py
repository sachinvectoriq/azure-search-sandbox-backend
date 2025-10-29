# update_settings.py

import os
import asyncpg
from quart import request, jsonify
from dotenv import load_dotenv

# Load env variables
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
        print(f"Error connecting to the database: {e}")
        return None

async def update_settings():
    # Read form data
    form = await request.form
    insert_fields = {}
    
    # Define field types for proper conversion
    field_types = {
        'azure_search_endpoint': str,
        'azure_search_index_name': str,
        'current_prompt': str,
        'openai_model_deployment_name': str,
        'openai_endpoint': str,
        'openai_api_version': str,
        'openai_model_temperature': float,  
        'semantic_configuration_name_english': str,
        'semantic_configuration_name_french': str,
        'openai_api_key': str,
        'user_name': str,
        'login_session_id': str,
        'azure_search_index_name_french' : str,
        'current_prompt_french' : str
    }

    for field, field_type in field_types.items():
        if form.get(field) is not None:
            try:
                if field_type == float:
                    insert_fields[field] = float(form.get(field))
                elif field_type == int:
                    insert_fields[field] = int(form.get(field))
                else:
                    insert_fields[field] = form.get(field)
            except ValueError:
                return jsonify({'error': f'Invalid {field_type.__name__} value for {field}'}), 400

    if not insert_fields:
        return jsonify({'error': 'No valid fields provided to insert'}), 400

    conn = await connect_db()
    if conn is None:
        return jsonify({'error': 'Database connection failed'}), 500

    try:
        # Build INSERT query dynamically
        columns = ', '.join(insert_fields.keys())
        placeholders = ', '.join(f"${i+1}" for i in range(len(insert_fields)))
        values = list(insert_fields.values())

        query = f"""
            INSERT INTO azaisearch_evolve__sandbox_settings ({columns})
            VALUES ({placeholders})
            RETURNING update_id
        """

        # Execute query and get the new update_id
        new_update_id = await conn.fetchval(query, *values)
        await conn.close()

        return jsonify({
            'message': f'New settings row created successfully with update_id={new_update_id}',
            'update_id': new_update_id
        })

    except Exception as e:
        print(f"Database error: {e}")
        return jsonify({'error': str(e)}), 500