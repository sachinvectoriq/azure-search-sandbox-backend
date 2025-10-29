import os
import asyncpg
from quart import request, jsonify
from dotenv import load_dotenv

# Load environment variables
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

async def get_settings():
    conn = await connect_db()
    if conn is None:
        return jsonify({'error': 'Database connection failed'}), 500

    try:
        # Query to get the row with maximum update_id (latest entry)
        query = """
            SELECT * FROM azaisearch_evolve_sandbox_settings
            WHERE update_id = (SELECT MAX(update_id) FROM azaisearch_evolve_sandbox_settings)
        """
        row = await conn.fetchrow(query)

        if row is None:
            return jsonify({'message': 'No settings found in the table'}), 404

        result = dict(row)
        return jsonify(result)

    except Exception as e:
        print(f"Database error: {e}")
        return jsonify({'error': str(e)}), 500

    finally:
        await conn.close()