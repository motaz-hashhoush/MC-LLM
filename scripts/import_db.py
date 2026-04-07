"""
MC-LLM Database Import Script
─────────────────────────────
Imports a SQL dump into the production PostgreSQL instance.

Usage (on production server):
    python scripts/import_db.py --input llm_logs_backup_20260407.sql
"""

import subprocess
import sys
import os

# ── Configuration ────────────────────────────────────────────────────────────
DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")
DB_NAME = os.getenv("DB_NAME", "llm_logs")


def find_psql():
    """Find psql executable on Windows or Linux."""
    common_paths = [
        r"C:\Program Files\PostgreSQL\17\bin\psql.exe",
        r"C:\Program Files\PostgreSQL\16\bin\psql.exe",
        r"C:\Program Files\PostgreSQL\15\bin\psql.exe",
    ]

    try:
        result = subprocess.run(
            ["psql", "--version"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return "psql"
    except FileNotFoundError:
        pass

    for path in common_paths:
        if os.path.exists(path):
            return path

    return None


def setup_user_and_db(psql_path, env):
    """Create the llm_user and llm_logs database if needed."""
    print("🔧 Setting up user and database...")

    # Create user
    cmd = [
        psql_path, "-h", DB_HOST, "-p", DB_PORT, "-U", DB_USER,
        "-d", "postgres", "-c",
        "DO $$ BEGIN IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'llm_user') "
        "THEN CREATE ROLE llm_user WITH LOGIN PASSWORD 'llm_pass'; END IF; END $$;"
    ]
    subprocess.run(cmd, env=env, capture_output=True, text=True)

    # Create database
    cmd = [
        psql_path, "-h", DB_HOST, "-p", DB_PORT, "-U", DB_USER,
        "-d", "postgres", "-c",
        f"SELECT 'CREATE DATABASE {DB_NAME} OWNER llm_user' "
        f"WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = '{DB_NAME}')"
    ]
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)

    # Check if database needs to be created
    if "CREATE DATABASE" in result.stdout:
        cmd = [
            psql_path, "-h", DB_HOST, "-p", DB_PORT, "-U", DB_USER,
            "-d", "postgres", "-c",
            f"CREATE DATABASE {DB_NAME} OWNER llm_user;"
        ]
        subprocess.run(cmd, env=env, capture_output=True, text=True)

    print("✅ User 'llm_user' and database 'llm_logs' are ready.")


def import_database(input_file):
    """Import a SQL dump into PostgreSQL."""
    if not os.path.exists(input_file):
        print(f"❌ File not found: {input_file}")
        sys.exit(1)

    psql_path = find_psql()
    if psql_path is None:
        print("❌ psql not found!")
        print("   Please install PostgreSQL client tools.")
        print("   On Ubuntu: sudo apt install postgresql-client")
        sys.exit(1)

    env = os.environ.copy()
    env["PGPASSWORD"] = DB_PASSWORD

    # Setup user and database first
    setup_user_and_db(psql_path, env)

    size = os.path.getsize(input_file)
    print(f"📥 Importing '{input_file}' ({size / 1024:.1f} KB) into {DB_HOST}:{DB_PORT}/{DB_NAME}...")

    cmd = [
        psql_path,
        "-h", DB_HOST,
        "-p", DB_PORT,
        "-U", DB_USER,
        "-d", DB_NAME,
        "-f", input_file,
    ]

    try:
        result = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Import successful!")

            # Grant privileges
            grant_cmd = [
                psql_path, "-h", DB_HOST, "-p", DB_PORT, "-U", DB_USER,
                "-d", DB_NAME, "-c",
                "GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO llm_user; "
                "GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO llm_user;"
            ]
            subprocess.run(grant_cmd, env=env, capture_output=True, text=True)
            print("✅ Privileges granted to llm_user.")

            # Verify
            verify_cmd = [
                psql_path, "-h", DB_HOST, "-p", DB_PORT, "-U", DB_USER,
                "-d", DB_NAME, "-c",
                "SELECT count(*) as total_rows FROM request_logs;"
            ]
            verify = subprocess.run(verify_cmd, env=env, capture_output=True, text=True)
            print(f"\n📊 Verification:\n{verify.stdout}")
        else:
            print(f"❌ Import failed!")
            if result.stderr:
                # Filter out non-critical warnings
                errors = [l for l in result.stderr.split('\n') if l and 'NOTICE' not in l]
                if errors:
                    print(f"   Errors: {chr(10).join(errors)}")
    except Exception as e:
        print(f"❌ Import failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    if "--input" not in sys.argv:
        print("Usage: python scripts/import_db.py --input <backup_file.sql>")
        print()
        print("Environment variables:")
        print("  DB_HOST     (default: 127.0.0.1)")
        print("  DB_PORT     (default: 5432)")
        print("  DB_USER     (default: postgres)")
        print("  DB_PASSWORD (default: postgres)")
        print("  DB_NAME     (default: llm_logs)")
        sys.exit(1)

    idx = sys.argv.index("--input")
    if idx + 1 >= len(sys.argv):
        print("❌ Please provide the input file path.")
        sys.exit(1)

    input_file = sys.argv[idx + 1]
    import_database(input_file)
