import asyncio
import sys
import os

# Add the root directory to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.db.database import DatabaseManager

async def test_connection():
    print("Testing async database connection using DatabaseManager...")
    db = DatabaseManager()
    
    try:
        is_reachable = await db.health_check()
        
        if is_reachable:
            print("✅ Successfully connected to the database via asyncpg!")
        else:
            print("❌ Failed to reach the database. Check your DATABASE_URL in .env.")
            
    except Exception as e:
        print(f"❌ An error occurred while testing the connection: {e}")
    finally:
        await db.dispose()

if __name__ == "__main__":
    asyncio.run(test_connection())
import psycopg2

conn = psycopg2.connect(
    host="127.0.0.1",
    port=5432,
    database="llm_logs",
    user="llm_user",
    password="llm_pass"
)

cursor = conn.cursor()
try:
    cursor.execute("SELECT version();")
    version = cursor.fetchone()
    print("✅ Successfully connected as llm_user!")
    print("PostgreSQL version:", version[0])
    
    cursor.execute("SELECT current_database();")
    db_name = cursor.fetchone()
    print("Connected to database:", db_name[0])
    
except Exception as e:
    print(f"❌ Connection error: {e}")
finally:
    cursor.close()
    conn.close()