"""
MC-LLM Database Export Script
─────────────────────────────
Exports the llm_logs database (schema + data) to a SQL dump file.

Usage:
    python scripts/export_db.py
    python scripts/export_db.py --output backup_2026-04-07.sql
"""

import subprocess
import sys
import os
from datetime import datetime

# ── Configuration ────────────────────────────────────────────────────────────
DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "llm_logs")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")


def find_pg_dump():
    """Find pg_dump executable on Windows or Linux."""
    # Common Windows PostgreSQL paths
    common_paths = [
        r"C:\Program Files\PostgreSQL\17\bin\pg_dump.exe",
        r"C:\Program Files\PostgreSQL\16\bin\pg_dump.exe",
        r"C:\Program Files\PostgreSQL\15\bin\pg_dump.exe",
        r"C:\Program Files\PostgreSQL\14\bin\pg_dump.exe",
    ]

    # Try system PATH first
    try:
        result = subprocess.run(
            ["pg_dump", "--version"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return "pg_dump"
    except FileNotFoundError:
        pass

    # Try common Windows paths
    for path in common_paths:
        if os.path.exists(path):
            return path

    return None


def export_database(output_file=None):
    """Export the llm_logs database to a SQL file."""
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"llm_logs_backup_{timestamp}.sql"

    pg_dump = find_pg_dump()
    if pg_dump is None:
        print("❌ pg_dump not found!")
        print("   Please install PostgreSQL client tools or add pg_dump to PATH.")
        print("   On Windows: Add C:\\Program Files\\PostgreSQL\\17\\bin to PATH")
        print("   On Linux:   sudo apt install postgresql-client")
        sys.exit(1)

    print(f"📦 Exporting database '{DB_NAME}' from {DB_HOST}:{DB_PORT}...")
    print(f"   Using: {pg_dump}")

    # Set password via environment variable
    env = os.environ.copy()
    env["PGPASSWORD"] = DB_PASSWORD

    cmd = [
        pg_dump,
        "-h", DB_HOST,
        "-p", DB_PORT,
        "-U", DB_USER,
        "-d", DB_NAME,
        "--no-owner",           # Don't include ownership commands
        "--no-privileges",      # Don't include GRANT/REVOKE
        "--clean",              # Include DROP statements before CREATE
        "--if-exists",          # Use IF EXISTS with DROP
        "--create",             # Include CREATE DATABASE
        "-f", output_file,
    ]

    try:
        result = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if result.returncode == 0:
            size = os.path.getsize(output_file)
            print(f"✅ Export successful!")
            print(f"   File: {os.path.abspath(output_file)}")
            print(f"   Size: {size / 1024:.1f} KB")
            print()
            print("📋 Next steps:")
            print(f"   1. Copy this file to your production server:")
            print(f"      scp {output_file} user@production-server:/tmp/")
            print(f"   2. On the production server, run:")
            print(f"      python scripts/import_db.py --input /tmp/{output_file}")
        else:
            print(f"❌ Export failed!")
            print(f"   Error: {result.stderr}")
    except Exception as e:
        print(f"❌ Export failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    output = None
    if "--output" in sys.argv:
        idx = sys.argv.index("--output")
        if idx + 1 < len(sys.argv):
            output = sys.argv[idx + 1]

    export_database(output)
