# app.py - Main Flask Application
from flask import Flask, send_file, jsonify, Response, request, render_template_string, render_template, send_from_directory
from flask_cors import CORS
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime, timedelta
import os
from pathlib import Path
import pandas as pd
import json
import logging
import shutil
import hashlib
import secrets
from functools import wraps

# Import from new package structure
from cleverly.services.pipeline.cold_email import main as run_pipeline, OUTPUT_ROOT, MondayConfig
from cleverly.core.config import Config

# Import Calendly data extraction modules
from cleverly.services.calendly.data_extractor import CCDataW, CalendlyAPIException
from cleverly.services.calendly.events import CCDWEvents, CalendlyAPIError as EventsAPIError
from cleverly.services.calendly.invitees import CCDWInvitees, CalendlyAPIError as InviteesAPIError

# Import API key management
from cleverly.core.security import (
    get_api_key_manager,
    APIKeyManager,
    KeyPermission,
    generate_curl_example,
    generate_python_example
)

# NEW imports for the non-blocking force-run
from threading import Lock
from uuid import uuid4

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)-8s %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("api")

# -----------------------------------------------------------------------------
# Flask app & CORS
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent  # Go up to project root
app = Flask(
    __name__,
    static_folder=str(PROJECT_ROOT / "static"),
    template_folder=str(PROJECT_ROOT / "templates")
)
CORS(
    app,
    origins=["*"],
    methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-API-Key"],
)

# -----------------------------------------------------------------------------
# Authentication Configuration
# -----------------------------------------------------------------------------
# API Keys for programmatic access (store securely in production)
API_KEYS = {
    os.getenv("CLEVERLY_API_KEY", "cleverly-default-api-key-2024"): "admin",
    os.getenv("CLEVERLY_API_KEY_READONLY", "cleverly-readonly-key-2024"): "readonly",
}

# User credentials for frontend login (use proper database in production)
USERS = {
    os.getenv("CLEVERLY_ADMIN_USER", "admin"): {
        "password_hash": hashlib.sha256(os.getenv("CLEVERLY_ADMIN_PASSWORD", "cleverly2024!").encode()).hexdigest(),
        "role": "admin"
    },
    os.getenv("CLEVERLY_USER", "user"): {
        "password_hash": hashlib.sha256(os.getenv("CLEVERLY_USER_PASSWORD", "userpass2024!").encode()).hexdigest(),
        "role": "user"
    }
}

# Session tokens (in production, use Redis or database)
active_sessions = {}

# Calendly API Token
CALENDLY_ACCESS_TOKEN = os.getenv("CALENDLY_ACCESS_TOKEN", "")

# Initialize API Key Manager
API_KEY_STORAGE = PROJECT_ROOT / "data" / "api_keys.json"
api_key_manager = get_api_key_manager(API_KEY_STORAGE)

# -----------------------------------------------------------------------------
# Authentication Decorators
# -----------------------------------------------------------------------------
def require_api_key(f):
    """Decorator to require API key authentication"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get("X-API-Key")
        if not api_key or api_key not in API_KEYS:
            logger.warning("Unauthorized API access attempt from %s", request.remote_addr)
            return jsonify({"error": "Invalid or missing API key"}), 401
        request.user_role = API_KEYS[api_key]
        return f(*args, **kwargs)
    return decorated_function

def require_session(f):
    """Decorator to require session token authentication"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
            if token in active_sessions:
                session = active_sessions[token]
                if session["expires"] > datetime.now():
                    request.user = session["user"]
                    request.user_role = session["role"]
                    return f(*args, **kwargs)
                else:
                    del active_sessions[token]

        logger.warning("Unauthorized session access attempt from %s", request.remote_addr)
        return jsonify({"error": "Invalid or expired session"}), 401
    return decorated_function

def require_auth(f):
    """Decorator that accepts either API key or session token"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Try new API key manager first
        api_key = request.headers.get("X-API-Key")
        if api_key:
            # Try new managed keys
            is_valid, key_obj, error = api_key_manager.validate_key(api_key, request.remote_addr)
            if is_valid and key_obj:
                request.user_role = key_obj.permission.value
                request.user = f"api_key:{key_obj.key_id}"
                request.api_key_id = key_obj.key_id
                return f(*args, **kwargs)

            # Fallback to legacy keys
            if api_key in API_KEYS:
                request.user_role = API_KEYS[api_key]
                request.user = "api_user"
                return f(*args, **kwargs)

        # Try session token
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
            if token in active_sessions:
                session = active_sessions[token]
                if session["expires"] > datetime.now():
                    request.user = session["user"]
                    request.user_role = session["role"]
                    return f(*args, **kwargs)
                else:
                    del active_sessions[token]

        logger.warning("Unauthorized access attempt from %s", request.remote_addr)
        return jsonify({"error": "Authentication required"}), 401
    return decorated_function

# -----------------------------------------------------------------------------
# In-memory cache for latest data
# -----------------------------------------------------------------------------
latest_data = {
    "combined_csv": None,
    "group_csvs": {},
    "json_data": None,
    "timestamp": None,
}

# Cache for Calendly data
calendly_cache = {
    "user_data": None,
    "events_data": None,
    "invitees_data": None,
    "last_updated": None,
}

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
def _refresh_latest_cache():
    """Populate latest_data by inspecting OUTPUT_ROOT and sessions/."""
    try:
        session_dir = OUTPUT_ROOT / "sessions"
        combined_csvs = list(OUTPUT_ROOT.glob("combined_output_*.csv"))
        latest_data["combined_csv"] = (max(combined_csvs, key=lambda p: p.stat().st_mtime)
                                       if combined_csvs else None)

        json_files = list(session_dir.glob("data_*.json")) if session_dir.exists() else []
        latest_data["json_data"] = (max(json_files, key=lambda p: p.stat().st_mtime)
                                    if json_files else None)

        group_map = {}
        for group in MondayConfig.GROUP_MAPPING.values():
            group_csvs = list(session_dir.glob(f"{group}_*.csv"))
            if group_csvs:
                group_map[group] = max(group_csvs, key=lambda p: p.stat().st_mtime)
        latest_data["group_csvs"] = group_map

        if latest_data["combined_csv"]:
            filename = latest_data["combined_csv"].name
            latest_data["timestamp"] = filename.replace("combined_output_", "").replace(".csv", "")
        else:
            latest_data["timestamp"] = None

        logger.info("Cache refreshed (timestamp=%s)", latest_data["timestamp"])
    except Exception as e:
        logger.error("Error refreshing cache: %s", e)


def run_scheduled_pipeline():
    """Run the pipeline (long job) and update the in-memory cache."""
    try:
        logger.info("Starting scheduled pipeline run")
        run_pipeline()
        _refresh_latest_cache()
        logger.info("Pipeline run completed, cache updated")
    except Exception as e:
        logger.error("Scheduled pipeline failed: %s", e)


def initialize_cache():
    """Initialize cache from existing files at startup (non-blocking)."""
    try:
        _refresh_latest_cache()
        if latest_data["timestamp"]:
            logger.info("Cache initialized with timestamp: %s", latest_data["timestamp"])
    except Exception as e:
        logger.error("Error initializing cache: %s", e)


# -----------------------------------------------------------------------------
# Background scheduler (single-process)
# -----------------------------------------------------------------------------
scheduler = BackgroundScheduler(job_defaults={"max_instances": 1, "coalesce": True})
_FORCE_RUN_LOCK = Lock()


def _force_run_job(reset_sessions: bool = True):
    """Background job scheduled by /force_start_pipeline."""
    if not _FORCE_RUN_LOCK.acquire(blocking=False):
        logger.warning("Force run requested but a run is already in progress.")
        return

    try:
        logger.info("Force run pipeline started (reset_sessions=%s)", reset_sessions)
        session_dir = OUTPUT_ROOT / "sessions"

        if reset_sessions:
            try:
                if session_dir.exists():
                    shutil.rmtree(session_dir)
                    logger.info("Sessions directory cleared")
                session_dir.mkdir(parents=True, exist_ok=True)
                logger.info("Sessions directory recreated")
            except Exception as e:
                logger.exception("Failed to reset sessions directory (continuing): %s", e)

        run_pipeline()
        _refresh_latest_cache()
        logger.info("Force run pipeline finished successfully")
    except Exception:
        logger.exception("Force run pipeline failed")
    finally:
        _FORCE_RUN_LOCK.release()


# -----------------------------------------------------------------------------
# Startup: cache + scheduler + periodic job
# -----------------------------------------------------------------------------
initialize_cache()
scheduler.add_job(
    run_scheduled_pipeline,
    "interval",
    hours=4,
    next_run_time=datetime.now().replace(second=0, microsecond=0) + timedelta(minutes=1),
)
scheduler.start()


# -----------------------------------------------------------------------------
# Authentication Routes
# -----------------------------------------------------------------------------
@app.route("/api/auth/login", methods=["POST"])
def login():
    """Authenticate user and return session token"""
    try:
        data = request.get_json()
        username = data.get("username", "")
        password = data.get("password", "")

        if not username or not password:
            return jsonify({"error": "Username and password required"}), 400

        if username not in USERS:
            logger.warning("Failed login attempt for user: %s from %s", username, request.remote_addr)
            return jsonify({"error": "Invalid credentials"}), 401

        password_hash = hashlib.sha256(password.encode()).hexdigest()
        if USERS[username]["password_hash"] != password_hash:
            logger.warning("Failed login attempt for user: %s from %s", username, request.remote_addr)
            return jsonify({"error": "Invalid credentials"}), 401

        # Generate session token
        token = secrets.token_urlsafe(32)
        expires = datetime.now() + timedelta(hours=24)

        active_sessions[token] = {
            "user": username,
            "role": USERS[username]["role"],
            "expires": expires,
            "created": datetime.now().isoformat()
        }

        logger.info("User %s logged in from %s", username, request.remote_addr)

        return jsonify({
            "token": token,
            "user": username,
            "role": USERS[username]["role"],
            "expires": expires.isoformat()
        })

    except Exception as e:
        logger.error("Login error: %s", e)
        return jsonify({"error": "Login failed"}), 500


@app.route("/api/auth/logout", methods=["POST"])
@require_session
def logout():
    """Invalidate session token"""
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:]
        if token in active_sessions:
            del active_sessions[token]

    logger.info("User %s logged out", request.user)
    return jsonify({"message": "Logged out successfully"})


@app.route("/api/auth/verify", methods=["GET"])
@require_auth
def verify_auth():
    """Verify authentication status"""
    return jsonify({
        "authenticated": True,
        "user": getattr(request, 'user', 'api_user'),
        "role": request.user_role
    })


# -----------------------------------------------------------------------------
# API Key Management Routes
# -----------------------------------------------------------------------------
@app.route("/api/keys", methods=["GET"])
@require_session
def list_api_keys():
    """List all API keys for the authenticated user"""
    include_revoked = request.args.get("include_revoked", "false").lower() == "true"
    keys = api_key_manager.list_keys(include_revoked=include_revoked)

    return jsonify({
        "keys": [k.to_dict() for k in keys],
        "total": len(keys)
    })


@app.route("/api/keys", methods=["POST"])
@require_session
def create_api_key():
    """Create a new API key"""
    try:
        data = request.get_json()

        name = data.get("name", "").strip()
        if not name:
            return jsonify({"error": "Key name is required"}), 400

        permission_str = data.get("permission", "standard")
        try:
            permission = KeyPermission(permission_str)
        except ValueError:
            return jsonify({"error": f"Invalid permission: {permission_str}"}), 400

        expires_in_days = data.get("expires_in_days")
        if expires_in_days:
            expires_in_days = int(expires_in_days)

        description = data.get("description", "")
        rate_limit = data.get("rate_limit", 100)

        plain_key, api_key = api_key_manager.create_key(
            name=name,
            permission=permission,
            created_by=request.user,
            expires_in_days=expires_in_days,
            description=description,
            rate_limit=rate_limit
        )

        logger.info("API key created: %s by %s", api_key.key_id, request.user)

        return jsonify({
            "key": plain_key,
            "key_id": api_key.key_id,
            "name": api_key.name,
            "permission": api_key.permission.value,
            "expires_at": api_key.expires_at.isoformat() if api_key.expires_at else None,
            "message": "Store this key securely - it will not be shown again!"
        }), 201

    except Exception as e:
        logger.error("Error creating API key: %s", e)
        return jsonify({"error": str(e)}), 500


@app.route("/api/keys/<key_id>", methods=["GET"])
@require_session
def get_api_key(key_id):
    """Get details for a specific API key"""
    stats = api_key_manager.get_key_stats(key_id)
    if not stats:
        return jsonify({"error": "Key not found"}), 404

    return jsonify(stats)


@app.route("/api/keys/<key_id>", methods=["PUT"])
@require_session
def update_api_key(key_id):
    """Update an API key's properties"""
    try:
        data = request.get_json()

        success = api_key_manager.update_key(
            key_id=key_id,
            name=data.get("name"),
            description=data.get("description"),
            rate_limit=data.get("rate_limit"),
            updated_by=request.user
        )

        if not success:
            return jsonify({"error": "Key not found"}), 404

        return jsonify({"message": "Key updated successfully"})

    except Exception as e:
        logger.error("Error updating API key: %s", e)
        return jsonify({"error": str(e)}), 500


@app.route("/api/keys/<key_id>", methods=["DELETE"])
@require_session
def revoke_api_key(key_id):
    """Revoke an API key"""
    reason = request.args.get("reason", "Revoked by user")

    success = api_key_manager.revoke_key(key_id, request.user, reason)
    if not success:
        return jsonify({"error": "Key not found"}), 404

    logger.info("API key revoked: %s by %s", key_id, request.user)
    return jsonify({"message": "Key revoked successfully"})


@app.route("/api/keys/<key_id>/audit", methods=["GET"])
@require_session
def get_key_audit_log(key_id):
    """Get audit log for a specific key"""
    limit = request.args.get("limit", 100, type=int)
    entries = api_key_manager.get_audit_log(key_id=key_id, limit=limit)

    return jsonify({
        "key_id": key_id,
        "entries": entries,
        "total": len(entries)
    })


@app.route("/api/keys/audit", methods=["GET"])
@require_session
def get_all_audit_log():
    """Get complete audit log"""
    limit = request.args.get("limit", 100, type=int)
    entries = api_key_manager.get_audit_log(limit=limit)

    return jsonify({
        "entries": entries,
        "total": len(entries)
    })


@app.route("/api/keys/examples", methods=["POST"])
@require_session
def generate_code_examples():
    """Generate code examples for API usage"""
    try:
        data = request.get_json()

        api_key = data.get("api_key", "YOUR_API_KEY")
        endpoint = data.get("endpoint", "/api/status")
        method = data.get("method", "GET")
        request_data = data.get("data")

        curl_example = generate_curl_example(endpoint, api_key, method, request_data)
        python_example = generate_python_example(endpoint, api_key, method, request_data)

        return jsonify({
            "curl": curl_example,
            "python": python_example
        })

    except Exception as e:
        logger.error("Error generating examples: %s", e)
        return jsonify({"error": str(e)}), 500


# -----------------------------------------------------------------------------
# Calendly Data Extraction Routes
# -----------------------------------------------------------------------------
@app.route("/api/calendly/user-info", methods=["GET"])
@require_auth
def get_calendly_user_info():
    """Extract and return Calendly user information"""
    logger.info("Calendly user info requested by %s from %s",
                getattr(request, 'user', 'api'), request.remote_addr)

    if not CALENDLY_ACCESS_TOKEN:
        return jsonify({"error": "Calendly access token not configured"}), 500

    try:
        extractor = CCDataW(access_token=CALENDLY_ACCESS_TOKEN)
        user_info = extractor.get_user_information()

        result = {
            "user_information": {
                "name": user_info.name,
                "email": user_info.email,
                "uri": user_info.uri,
                "timezone": user_info.timezone,
                "scheduling_url": user_info.scheduling_url,
                "created_at": user_info.created_at,
                "updated_at": user_info.updated_at,
                "current_organization": user_info.current_organization,
                "resource_type": user_info.resource_type
            },
            "metadata": {
                "extracted_at": datetime.now().isoformat(),
                "source": "CCDataW"
            }
        }

        calendly_cache["user_data"] = result
        calendly_cache["last_updated"] = datetime.now().isoformat()

        logger.info("Successfully extracted Calendly user info for: %s", user_info.name)
        return jsonify(result)

    except CalendlyAPIException as e:
        logger.error("Calendly API error: %s", e)
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        logger.error("Unexpected error: %s", e)
        return jsonify({"error": "Failed to extract user information"}), 500


@app.route("/api/calendly/organization-members", methods=["GET"])
@require_auth
def get_calendly_organization_members():
    """Extract and return Calendly organization members"""
    logger.info("Calendly org members requested by %s from %s",
                getattr(request, 'user', 'api'), request.remote_addr)

    if not CALENDLY_ACCESS_TOKEN:
        return jsonify({"error": "Calendly access token not configured"}), 500

    try:
        extractor = CCDataW(access_token=CALENDLY_ACCESS_TOKEN)
        members = extractor.get_organization_members()

        members_list = []
        for member in members:
            members_list.append({
                "name": member.name,
                "email": member.email,
                "user_uri": member.user_uri,
                "role": member.role,
                "status": member.status,
                "organization_uri": member.organization_uri,
                "membership_uri": member.membership_uri
            })

        result = {
            "organization_members": members_list,
            "metadata": {
                "total_members": len(members_list),
                "extracted_at": datetime.now().isoformat(),
                "source": "CCDataW"
            }
        }

        logger.info("Successfully extracted %d organization members", len(members_list))
        return jsonify(result)

    except CalendlyAPIException as e:
        logger.error("Calendly API error: %s", e)
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        logger.error("Unexpected error: %s", e)
        return jsonify({"error": "Failed to extract organization members"}), 500


@app.route("/api/calendly/events", methods=["GET"])
@require_auth
def get_calendly_events():
    """Extract and return Calendly event types"""
    logger.info("Calendly events requested by %s from %s",
                getattr(request, 'user', 'api'), request.remote_addr)

    if not CALENDLY_ACCESS_TOKEN:
        return jsonify({"error": "Calendly access token not configured"}), 500

    try:
        extractor = CCDWEvents(access_token=CALENDLY_ACCESS_TOKEN)
        event_data = extractor.extract_comprehensive_events()

        calendly_cache["events_data"] = event_data
        calendly_cache["last_updated"] = datetime.now().isoformat()

        extractor.close()

        logger.info("Successfully extracted Calendly events data")
        return jsonify(event_data)

    except EventsAPIError as e:
        logger.error("Calendly Events API error: %s", e)
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        logger.error("Unexpected error: %s", e)
        return jsonify({"error": "Failed to extract events data"}), 500


@app.route("/api/calendly/invitees", methods=["GET"])
@require_auth
def get_calendly_invitees():
    """Extract and return Calendly invitees data"""
    logger.info("Calendly invitees requested by %s from %s",
                getattr(request, 'user', 'api'), request.remote_addr)

    if not CALENDLY_ACCESS_TOKEN:
        return jsonify({"error": "Calendly access token not configured"}), 500

    # Get optional parameters
    days_back = request.args.get("days_back", default=365, type=int)

    try:
        extractor = CCDWInvitees(access_token=CALENDLY_ACCESS_TOKEN)
        invitee_data = extractor.extract_invitees_comprehensive(days_back=days_back)

        calendly_cache["invitees_data"] = invitee_data
        calendly_cache["last_updated"] = datetime.now().isoformat()

        extractor.close()

        logger.info("Successfully extracted Calendly invitees data")
        return jsonify(invitee_data)

    except InviteesAPIError as e:
        logger.error("Calendly Invitees API error: %s", e)
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        logger.error("Unexpected error: %s", e)
        return jsonify({"error": "Failed to extract invitees data"}), 500


@app.route("/api/calendly/comprehensive", methods=["GET"])
@require_auth
def get_calendly_comprehensive():
    """Extract comprehensive Calendly data including user, events, and invitees"""
    logger.info("Calendly comprehensive data requested by %s from %s",
                getattr(request, 'user', 'api'), request.remote_addr)

    if not CALENDLY_ACCESS_TOKEN:
        return jsonify({"error": "Calendly access token not configured"}), 500

    days_back = request.args.get("days_back", default=365, type=int)

    try:
        # Extract user data
        user_extractor = CCDataW(access_token=CALENDLY_ACCESS_TOKEN)
        comprehensive_data = user_extractor.comprehensive_extraction(export=False)

        # Extract events
        events_extractor = CCDWEvents(access_token=CALENDLY_ACCESS_TOKEN)
        events_data = events_extractor.extract_comprehensive_events()
        events_extractor.close()

        # Extract invitees
        invitees_extractor = CCDWInvitees(access_token=CALENDLY_ACCESS_TOKEN)
        invitees_data = invitees_extractor.extract_invitees_comprehensive(days_back=days_back)
        invitees_extractor.close()

        result = {
            "user_data": {
                "name": comprehensive_data["user_information"].name,
                "email": comprehensive_data["user_information"].email,
                "organization": comprehensive_data["organization_information"]
            },
            "organization_members": [
                {
                    "name": m.name,
                    "email": m.email,
                    "role": m.role,
                    "status": m.status
                }
                for m in comprehensive_data["organization_members"]
            ],
            "events": events_data,
            "invitees": invitees_data,
            "metadata": {
                "extracted_at": datetime.now().isoformat(),
                "days_back": days_back,
                "source": "Cleverly Calendly Data Warehouse"
            }
        }

        # Update cache
        calendly_cache["user_data"] = result["user_data"]
        calendly_cache["events_data"] = events_data
        calendly_cache["invitees_data"] = invitees_data
        calendly_cache["last_updated"] = datetime.now().isoformat()

        logger.info("Successfully extracted comprehensive Calendly data")
        return jsonify(result)

    except Exception as e:
        logger.error("Comprehensive extraction error: %s", e)
        return jsonify({"error": str(e)}), 500


@app.route("/api/calendly/export/<data_type>", methods=["GET"])
@require_auth
def export_calendly_data(data_type):
    """Export Calendly data as CSV file"""
    logger.info("Calendly %s export requested by %s from %s",
                data_type, getattr(request, 'user', 'api'), request.remote_addr)

    if data_type not in ["events", "invitees", "members"]:
        return jsonify({"error": "Invalid data type. Use: events, invitees, or members"}), 400

    cache_key = f"{data_type}_data" if data_type != "members" else "user_data"

    if not calendly_cache.get(cache_key):
        return jsonify({"error": f"No {data_type} data in cache. Please fetch data first."}), 404

    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"calendly_{data_type}_{timestamp}.csv"
        filepath = OUTPUT_ROOT / "exports" / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)

        if data_type == "events":
            data = calendly_cache["events_data"]
            events_list = data.get("organization_events", []) + data.get("user_events", [])
            df = pd.DataFrame(events_list)
        elif data_type == "invitees":
            data = calendly_cache["invitees_data"]
            df = pd.DataFrame(data.get("invitees", []))
        else:  # members
            data = calendly_cache["user_data"]
            if isinstance(data, dict) and "organization_members" in data:
                df = pd.DataFrame(data["organization_members"])
            else:
                return jsonify({"error": "Members data not available"}), 404

        df.to_csv(filepath, index=False)

        logger.info("Exported %s data to %s", data_type, filepath)
        return send_file(filepath, mimetype="text/csv", as_attachment=True, download_name=filename)

    except Exception as e:
        logger.error("Export error: %s", e)
        return jsonify({"error": str(e)}), 500


# -----------------------------------------------------------------------------
# Calendly Session Data Endpoints (from pipeline extraction)
# -----------------------------------------------------------------------------
@app.route("/api/calendly/session/<data_type>", methods=["GET"])
@require_auth
def get_calendly_session_data(data_type):
    """Get Calendly data from the latest pipeline session extraction"""
    logger.info("Calendly session %s requested by %s from %s",
                data_type, getattr(request, 'user', 'api'), request.remote_addr)

    if data_type not in ["user", "members", "events", "invitees", "summary"]:
        return jsonify({"error": "Invalid data type. Use: user, members, events, invitees, or summary"}), 400

    try:
        session_dir = OUTPUT_ROOT / "sessions" / "calendly"
        if not session_dir.exists():
            return jsonify({"error": "No Calendly session data available. Run the pipeline first."}), 404

        # Find the latest file for the requested type
        file_patterns = {
            "user": "calendly_user_data_*.json",
            "members": "calendly_members_*.csv",
            "events": "calendly_events_*.json",
            "invitees": "calendly_invitees_*.json",
            "summary": "calendly_extraction_summary_*.json"
        }

        pattern = file_patterns[data_type]
        files = list(session_dir.glob(pattern))

        if not files:
            return jsonify({"error": f"No {data_type} data found in session"}), 404

        latest_file = max(files, key=lambda p: p.stat().st_mtime)

        if latest_file.suffix == ".json":
            with open(latest_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return jsonify(data)
        else:  # CSV
            return send_file(
                latest_file,
                mimetype="text/csv",
                as_attachment=True,
                download_name=latest_file.name
            )

    except Exception as e:
        logger.error("Session data error: %s", e)
        return jsonify({"error": str(e)}), 500


@app.route("/api/calendly/session/csv/<data_type>", methods=["GET"])
@require_auth
def get_calendly_session_csv(data_type):
    """Get Calendly CSV data from the latest pipeline session extraction"""
    logger.info("Calendly session CSV %s requested by %s from %s",
                data_type, getattr(request, 'user', 'api'), request.remote_addr)

    if data_type not in ["members", "events", "invitees"]:
        return jsonify({"error": "Invalid data type. Use: members, events, or invitees"}), 400

    try:
        session_dir = OUTPUT_ROOT / "sessions" / "calendly"
        if not session_dir.exists():
            return jsonify({"error": "No Calendly session data available. Run the pipeline first."}), 404

        pattern = f"calendly_{data_type}_*.csv"
        files = list(session_dir.glob(pattern))

        if not files:
            return jsonify({"error": f"No {data_type} CSV found in session"}), 404

        latest_file = max(files, key=lambda p: p.stat().st_mtime)

        return send_file(
            latest_file,
            mimetype="text/csv",
            as_attachment=True,
            download_name=latest_file.name
        )

    except Exception as e:
        logger.error("Session CSV error: %s", e)
        return jsonify({"error": str(e)}), 500


@app.route("/api/calendly/session/status", methods=["GET"])
def get_calendly_session_status():
    """Get status of Calendly session data availability"""
    try:
        session_dir = OUTPUT_ROOT / "sessions" / "calendly"

        if not session_dir.exists():
            return jsonify({
                "available": False,
                "message": "No Calendly session data. Run the pipeline first."
            })

        # Check for files
        status = {
            "available": True,
            "directory": str(session_dir),
            "files": {}
        }

        file_types = {
            "user_data": "calendly_user_data_*.json",
            "members_csv": "calendly_members_*.csv",
            "events_json": "calendly_events_*.json",
            "events_csv": "calendly_events_*.csv",
            "invitees_json": "calendly_invitees_*.json",
            "invitees_csv": "calendly_invitees_*.csv",
            "summary": "calendly_extraction_summary_*.json"
        }

        for name, pattern in file_types.items():
            files = list(session_dir.glob(pattern))
            if files:
                latest = max(files, key=lambda p: p.stat().st_mtime)
                status["files"][name] = {
                    "exists": True,
                    "filename": latest.name,
                    "size_kb": round(latest.stat().st_size / 1024, 2),
                    "modified": datetime.fromtimestamp(latest.stat().st_mtime).isoformat()
                }
            else:
                status["files"][name] = {"exists": False}

        return jsonify(status)

    except Exception as e:
        logger.error("Session status error: %s", e)
        return jsonify({"error": str(e)}), 500


# -----------------------------------------------------------------------------
# Original Pipeline Routes
# -----------------------------------------------------------------------------
@app.route("/api/combined_csv", methods=["GET"])
def get_combined_csv():
    """Serve the latest combined CSV file."""
    logger.info("Combined CSV endpoint accessed from %s", request.remote_addr)

    if not latest_data["combined_csv"]:
        return jsonify({"error": "No combined CSV available", "debug": "Cache is empty"}), 404

    if not latest_data["combined_csv"].exists():
        available_files = list(OUTPUT_ROOT.glob("combined_output_*.csv"))
        return jsonify({
            "error": f"Combined CSV file not found: {latest_data['combined_csv']}",
            "available_files": [str(f) for f in available_files],
        }), 404

    return send_file(
        latest_data["combined_csv"],
        mimetype="text/csv",
        as_attachment=True,
        download_name=latest_data["combined_csv"].name,
    )


@app.route("/api/group_csv/<group>", methods=["GET"])
def get_group_csv(group):
    """Serve the latest CSV for a specific group."""
    logger.info("Group CSV endpoint accessed for group: %s from %s", group, request.remote_addr)

    if group not in latest_data["group_csvs"]:
        available_groups = list(latest_data["group_csvs"].keys())
        return jsonify({
            "error": f"Group {group} not found",
            "available_groups": available_groups,
        }), 404

    csv_path = latest_data["group_csvs"].get(group)
    if not csv_path or not csv_path.exists():
        session_dir = OUTPUT_ROOT / "sessions"
        available_files = list(session_dir.glob(f"{group}_*.csv")) if session_dir.exists() else []
        return jsonify({
            "error": f"No CSV available for group {group}",
            "expected_path": str(csv_path),
            "available_files": [str(f) for f in available_files],
        }), 404

    return send_file(csv_path, mimetype="text/csv", as_attachment=True, download_name=csv_path.name)


@app.route("/api/json", methods=["GET"])
def get_json():
    """Serve the latest JSON data."""
    _refresh_latest_cache()
    if not latest_data["json_data"]:
        return jsonify({"error": "No JSON data available"}), 404

    if not latest_data["json_data"].exists():
        return jsonify({"error": f"JSON file not found: {latest_data['json_data']}"}), 404

    try:
        with open(latest_data["json_data"], "r", encoding="utf-8") as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        logger.error("Error reading JSON file: %s", e)
        return jsonify({"error": "Error reading JSON data"}), 500


@app.route("/force_start_pipeline", methods=["GET"])
@require_auth
def force_start_pipeline():
    """Trigger an immediate pipeline run in the background."""
    logger.info("Force start pipeline endpoint accessed from %s", request.remote_addr)

    if _FORCE_RUN_LOCK.locked():
        return jsonify({
            "status": "busy",
            "message": "A pipeline run is already in progress",
        }), 409

    job_id = f"force-{uuid4()}"
    scheduler.add_job(
        _force_run_job,
        trigger="date",
        args=(True,),
        id=job_id,
        replace_existing=False,
    )
    return jsonify({"status": "scheduled", "job_id": job_id}), 202


@app.after_request
def after_request(response):
    """Add CORS headers to all responses."""
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization,X-API-Key")
    response.headers.add("Access-Control-Allow-Methods", "GET,PUT,POST,DELETE,OPTIONS")
    return response


@app.route("/", methods=["GET"])
def health_check():
    """Health check endpoint for deployment."""
    return jsonify({"status": "healthy", "message": "Cleverly API is running"})


@app.route("/favicon.ico", methods=["GET"])
def favicon():
    """Serve favicon to prevent 404s."""
    return "", 204


@app.route("/api/status", methods=["GET"])
def get_status():
    """Return the timestamp of the last pipeline run."""
    if not latest_data["timestamp"]:
        return jsonify({"status": "No data available", "timestamp": None})
    return jsonify({
        "status": "Data available",
        "timestamp": latest_data["timestamp"],
        "combined_csv": str(latest_data["combined_csv"]) if latest_data["combined_csv"] else None,
        "group_csvs": {k: str(v) for k, v in latest_data["group_csvs"].items()},
        "json_data": str(latest_data["json_data"]) if latest_data["json_data"] else None,
        "calendly_cache": {
            "last_updated": calendly_cache.get("last_updated"),
            "has_user_data": calendly_cache.get("user_data") is not None,
            "has_events_data": calendly_cache.get("events_data") is not None,
            "has_invitees_data": calendly_cache.get("invitees_data") is not None,
        }
    })


@app.before_request
def handle_preflight():
    """Handle CORS preflight OPTIONS requests."""
    if request.method == "OPTIONS":
        response = Response()
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "*")
        response.headers.add("Access-Control-Allow-Methods", "*")
        return response


# -----------------------------------------------------------------------------
# Modern Frontend Dashboard
# -----------------------------------------------------------------------------
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Cleverly Data Hub</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        :root {
            --primary: #6366f1;
            --primary-dark: #4f46e5;
            --secondary: #8b5cf6;
            --accent: #06b6d4;
            --success: #10b981;
            --warning: #f59e0b;
            --error: #ef4444;
            --bg-dark: #0f172a;
            --bg-card: #1e293b;
            --bg-hover: #334155;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --border: #334155;
        }

        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, var(--bg-dark) 0%, #1a1a2e 100%);
            min-height: 100vh;
            color: var(--text-primary);
        }

        .login-container {
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            padding: 20px;
        }

        .login-card {
            background: var(--bg-card);
            border-radius: 24px;
            padding: 48px;
            width: 100%;
            max-width: 420px;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
            border: 1px solid var(--border);
        }

        .login-logo {
            text-align: center;
            margin-bottom: 32px;
        }

        .login-logo h1 {
            font-size: 32px;
            font-weight: 700;
            background: linear-gradient(135deg, var(--primary) 0%, var(--accent) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .login-logo p {
            color: var(--text-secondary);
            margin-top: 8px;
        }

        .form-group {
            margin-bottom: 24px;
        }

        .form-group label {
            display: block;
            margin-bottom: 8px;
            font-weight: 500;
            color: var(--text-secondary);
        }

        .form-group input {
            width: 100%;
            padding: 14px 16px;
            border: 2px solid var(--border);
            border-radius: 12px;
            background: var(--bg-dark);
            color: var(--text-primary);
            font-size: 16px;
            transition: all 0.3s ease;
        }

        .form-group input:focus {
            outline: none;
            border-color: var(--primary);
            box-shadow: 0 0 0 4px rgba(99, 102, 241, 0.2);
        }

        .btn {
            width: 100%;
            padding: 14px 24px;
            border: none;
            border-radius: 12px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
        }

        .btn-primary {
            background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
            color: white;
        }

        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px -5px rgba(99, 102, 241, 0.5);
        }

        .btn-primary:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }

        .error-message {
            background: rgba(239, 68, 68, 0.1);
            border: 1px solid var(--error);
            color: var(--error);
            padding: 12px 16px;
            border-radius: 8px;
            margin-bottom: 16px;
            display: none;
        }

        /* Dashboard Styles */
        .dashboard-container {
            display: none;
            min-height: 100vh;
        }

        .dashboard-header {
            background: var(--bg-card);
            border-bottom: 1px solid var(--border);
            padding: 16px 32px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .dashboard-header h1 {
            font-size: 24px;
            font-weight: 700;
            background: linear-gradient(135deg, var(--primary) 0%, var(--accent) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .user-info {
            display: flex;
            align-items: center;
            gap: 16px;
        }

        .user-badge {
            background: var(--bg-dark);
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 14px;
        }

        .btn-logout {
            background: var(--bg-dark);
            color: var(--text-primary);
            padding: 8px 16px;
            border: 1px solid var(--border);
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s ease;
        }

        .btn-logout:hover {
            background: var(--error);
            border-color: var(--error);
        }

        .dashboard-content {
            padding: 32px;
            max-width: 1400px;
            margin: 0 auto;
        }

        .section-title {
            font-size: 20px;
            font-weight: 600;
            margin-bottom: 20px;
            color: var(--text-primary);
        }

        .cards-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 24px;
            margin-bottom: 40px;
        }

        .action-card {
            background: var(--bg-card);
            border-radius: 16px;
            padding: 24px;
            border: 1px solid var(--border);
            transition: all 0.3s ease;
        }

        .action-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 20px 40px -10px rgba(0, 0, 0, 0.3);
            border-color: var(--primary);
        }

        .card-icon {
            width: 48px;
            height: 48px;
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 24px;
            margin-bottom: 16px;
        }

        .icon-user { background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%); }
        .icon-events { background: linear-gradient(135deg, var(--accent) 0%, var(--success) 100%); }
        .icon-invitees { background: linear-gradient(135deg, var(--warning) 0%, var(--error) 100%); }
        .icon-comprehensive { background: linear-gradient(135deg, var(--secondary) 0%, var(--primary) 100%); }
        .icon-pipeline { background: linear-gradient(135deg, var(--success) 0%, var(--accent) 100%); }

        .card-title {
            font-size: 18px;
            font-weight: 600;
            margin-bottom: 8px;
        }

        .card-description {
            color: var(--text-secondary);
            font-size: 14px;
            margin-bottom: 20px;
            line-height: 1.5;
        }

        .card-actions {
            display: flex;
            gap: 12px;
        }

        .btn-action {
            flex: 1;
            padding: 10px 16px;
            border-radius: 8px;
            font-size: 14px;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.3s ease;
            border: none;
        }

        .btn-fetch {
            background: var(--primary);
            color: white;
        }

        .btn-fetch:hover {
            background: var(--primary-dark);
        }

        .btn-export {
            background: var(--bg-dark);
            color: var(--text-primary);
            border: 1px solid var(--border);
        }

        .btn-export:hover {
            background: var(--bg-hover);
        }

        .btn-action:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }

        /* Status Panel */
        .status-panel {
            background: var(--bg-card);
            border-radius: 16px;
            padding: 24px;
            border: 1px solid var(--border);
            margin-bottom: 40px;
        }

        .status-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
        }

        .status-item {
            background: var(--bg-dark);
            padding: 16px;
            border-radius: 12px;
        }

        .status-label {
            font-size: 12px;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 8px;
        }

        .status-value {
            font-size: 16px;
            font-weight: 600;
        }

        .status-online { color: var(--success); }
        .status-offline { color: var(--error); }

        /* Results Panel */
        .results-panel {
            background: var(--bg-card);
            border-radius: 16px;
            padding: 24px;
            border: 1px solid var(--border);
            display: none;
        }

        .results-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 16px;
        }

        .results-content {
            background: var(--bg-dark);
            border-radius: 12px;
            padding: 16px;
            max-height: 400px;
            overflow: auto;
        }

        .results-content pre {
            font-family: 'Fira Code', monospace;
            font-size: 13px;
            line-height: 1.6;
            white-space: pre-wrap;
            word-wrap: break-word;
        }

        .btn-close {
            background: transparent;
            border: none;
            color: var(--text-secondary);
            font-size: 24px;
            cursor: pointer;
            padding: 4px;
        }

        .btn-close:hover {
            color: var(--text-primary);
        }

        /* Loading Spinner */
        .spinner {
            display: inline-block;
            width: 16px;
            height: 16px;
            border: 2px solid rgba(255,255,255,0.3);
            border-radius: 50%;
            border-top-color: white;
            animation: spin 1s ease-in-out infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        /* Toast Notifications */
        .toast-container {
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 1000;
        }

        .toast {
            background: var(--bg-card);
            border-radius: 12px;
            padding: 16px 20px;
            margin-bottom: 12px;
            border-left: 4px solid;
            box-shadow: 0 10px 40px -10px rgba(0, 0, 0, 0.5);
            animation: slideIn 0.3s ease;
        }

        .toast-success { border-color: var(--success); }
        .toast-error { border-color: var(--error); }
        .toast-info { border-color: var(--accent); }

        @keyframes slideIn {
            from {
                transform: translateX(100%);
                opacity: 0;
            }
            to {
                transform: translateX(0);
                opacity: 1;
            }
        }

        /* Responsive */
        @media (max-width: 768px) {
            .dashboard-header {
                flex-direction: column;
                gap: 16px;
            }

            .dashboard-content {
                padding: 16px;
            }

            .cards-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <!-- Login Screen -->
    <div class="login-container" id="loginScreen">
        <div class="login-card">
            <div class="login-logo">
                <h1>Cleverly</h1>
                <p>Data Hub Dashboard</p>
            </div>
            <div class="error-message" id="loginError"></div>
            <form id="loginForm">
                <div class="form-group">
                    <label for="username">Username</label>
                    <input type="text" id="username" name="username" required autocomplete="username">
                </div>
                <div class="form-group">
                    <label for="password">Password</label>
                    <input type="password" id="password" name="password" required autocomplete="current-password">
                </div>
                <button type="submit" class="btn btn-primary" id="loginBtn">
                    Sign In
                </button>
            </form>
        </div>
    </div>

    <!-- Dashboard -->
    <div class="dashboard-container" id="dashboardScreen">
        <header class="dashboard-header">
            <h1>Cleverly Data Hub</h1>
            <div class="user-info">
                <span class="user-badge" id="userBadge"></span>
                <button class="btn-logout" onclick="logout()">Logout</button>
            </div>
        </header>

        <div class="dashboard-content">
            <!-- Status Panel -->
            <h2 class="section-title">System Status</h2>
            <div class="status-panel">
                <div class="status-grid">
                    <div class="status-item">
                        <div class="status-label">API Status</div>
                        <div class="status-value status-online" id="apiStatus">Online</div>
                    </div>
                    <div class="status-item">
                        <div class="status-label">Last Pipeline Run</div>
                        <div class="status-value" id="lastPipeline">-</div>
                    </div>
                    <div class="status-item">
                        <div class="status-label">Calendly Cache</div>
                        <div class="status-value" id="calendlyCache">Not loaded</div>
                    </div>
                    <div class="status-item">
                        <div class="status-label">Session Expires</div>
                        <div class="status-value" id="sessionExpires">-</div>
                    </div>
                </div>
            </div>

            <!-- Calendly Data Actions -->
            <h2 class="section-title">Calendly Data Extraction</h2>
            <div class="cards-grid">
                <div class="action-card">
                    <div class="card-icon icon-user">👤</div>
                    <h3 class="card-title">User Information</h3>
                    <p class="card-description">Extract your Calendly user profile and organization details.</p>
                    <div class="card-actions">
                        <button class="btn-action btn-fetch" onclick="fetchData('user-info')">Fetch Data</button>
                    </div>
                </div>

                <div class="action-card">
                    <div class="card-icon icon-events">📅</div>
                    <h3 class="card-title">Event Types</h3>
                    <p class="card-description">Extract all event types from organization and users.</p>
                    <div class="card-actions">
                        <button class="btn-action btn-fetch" onclick="fetchData('events')">Fetch Data</button>
                        <button class="btn-action btn-export" onclick="exportData('events')">Export CSV</button>
                    </div>
                </div>

                <div class="action-card">
                    <div class="card-icon icon-invitees">👥</div>
                    <h3 class="card-title">Invitees Data</h3>
                    <p class="card-description">Extract invitee information from scheduled events.</p>
                    <div class="card-actions">
                        <button class="btn-action btn-fetch" onclick="fetchData('invitees')">Fetch Data</button>
                        <button class="btn-action btn-export" onclick="exportData('invitees')">Export CSV</button>
                    </div>
                </div>

                <div class="action-card">
                    <div class="card-icon icon-comprehensive">🔄</div>
                    <h3 class="card-title">Comprehensive Export</h3>
                    <p class="card-description">Extract all Calendly data including users, events, and invitees.</p>
                    <div class="card-actions">
                        <button class="btn-action btn-fetch" onclick="fetchData('comprehensive')">Full Extraction</button>
                    </div>
                </div>

                <div class="action-card">
                    <div class="card-icon icon-pipeline">⚡</div>
                    <h3 class="card-title">Run Pipeline</h3>
                    <p class="card-description">Trigger the main data processing pipeline manually.</p>
                    <div class="card-actions">
                        <button class="btn-action btn-fetch" onclick="runPipeline()">Start Pipeline</button>
                    </div>
                </div>
            </div>

            <!-- Results Panel -->
            <div class="results-panel" id="resultsPanel">
                <div class="results-header">
                    <h3 class="section-title" id="resultsTitle">Results</h3>
                    <button class="btn-close" onclick="closeResults()">&times;</button>
                </div>
                <div class="results-content">
                    <pre id="resultsContent"></pre>
                </div>
            </div>
        </div>
    </div>

    <!-- Toast Container -->
    <div class="toast-container" id="toastContainer"></div>

    <script>
        // State
        let authToken = localStorage.getItem('authToken');
        let currentUser = localStorage.getItem('currentUser');
        let sessionExpires = localStorage.getItem('sessionExpires');

        // Initialize
        document.addEventListener('DOMContentLoaded', () => {
            if (authToken && sessionExpires && new Date(sessionExpires) > new Date()) {
                showDashboard();
                loadStatus();
            } else {
                showLogin();
            }
        });

        // Login Form Handler
        document.getElementById('loginForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const btn = document.getElementById('loginBtn');
            const errorEl = document.getElementById('loginError');

            btn.disabled = true;
            btn.innerHTML = '<span class="spinner"></span> Signing in...';
            errorEl.style.display = 'none';

            try {
                const response = await fetch('/api/auth/login', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        username: document.getElementById('username').value,
                        password: document.getElementById('password').value
                    })
                });

                const data = await response.json();

                if (response.ok) {
                    authToken = data.token;
                    currentUser = data.user;
                    sessionExpires = data.expires;

                    localStorage.setItem('authToken', authToken);
                    localStorage.setItem('currentUser', currentUser);
                    localStorage.setItem('sessionExpires', sessionExpires);

                    showDashboard();
                    loadStatus();
                    showToast('Welcome back, ' + currentUser + '!', 'success');
                } else {
                    errorEl.textContent = data.error || 'Login failed';
                    errorEl.style.display = 'block';
                }
            } catch (err) {
                errorEl.textContent = 'Connection error. Please try again.';
                errorEl.style.display = 'block';
            } finally {
                btn.disabled = false;
                btn.textContent = 'Sign In';
            }
        });

        // Show/Hide Screens
        function showLogin() {
            document.getElementById('loginScreen').style.display = 'flex';
            document.getElementById('dashboardScreen').style.display = 'none';
        }

        function showDashboard() {
            document.getElementById('loginScreen').style.display = 'none';
            document.getElementById('dashboardScreen').style.display = 'block';
            document.getElementById('userBadge').textContent = currentUser || 'User';

            if (sessionExpires) {
                const expDate = new Date(sessionExpires);
                document.getElementById('sessionExpires').textContent = expDate.toLocaleTimeString();
            }
        }

        // Logout
        async function logout() {
            try {
                await fetch('/api/auth/logout', {
                    method: 'POST',
                    headers: { 'Authorization': 'Bearer ' + authToken }
                });
            } catch (err) {
                console.error('Logout error:', err);
            }

            localStorage.removeItem('authToken');
            localStorage.removeItem('currentUser');
            localStorage.removeItem('sessionExpires');
            authToken = null;
            currentUser = null;

            showLogin();
            showToast('Logged out successfully', 'info');
        }

        // Load Status
        async function loadStatus() {
            try {
                const response = await fetch('/api/status');
                const data = await response.json();

                document.getElementById('lastPipeline').textContent = data.timestamp || 'Never';

                const cache = data.calendly_cache || {};
                if (cache.last_updated) {
                    document.getElementById('calendlyCache').textContent = 'Updated ' + new Date(cache.last_updated).toLocaleTimeString();
                }
            } catch (err) {
                document.getElementById('apiStatus').textContent = 'Error';
                document.getElementById('apiStatus').className = 'status-value status-offline';
            }
        }

        // Fetch Data
        async function fetchData(type) {
            const endpoints = {
                'user-info': '/api/calendly/user-info',
                'events': '/api/calendly/events',
                'invitees': '/api/calendly/invitees',
                'comprehensive': '/api/calendly/comprehensive'
            };

            showToast('Fetching ' + type + ' data...', 'info');

            try {
                const response = await fetch(endpoints[type], {
                    headers: { 'Authorization': 'Bearer ' + authToken }
                });

                if (response.status === 401) {
                    showToast('Session expired. Please login again.', 'error');
                    logout();
                    return;
                }

                const data = await response.json();

                if (response.ok) {
                    showResults(type + ' Data', data);
                    showToast('Data fetched successfully!', 'success');
                    loadStatus();
                } else {
                    showToast(data.error || 'Failed to fetch data', 'error');
                }
            } catch (err) {
                showToast('Network error: ' + err.message, 'error');
            }
        }

        // Export Data
        async function exportData(type) {
            showToast('Preparing export...', 'info');

            try {
                const response = await fetch('/api/calendly/export/' + type, {
                    headers: { 'Authorization': 'Bearer ' + authToken }
                });

                if (response.status === 401) {
                    showToast('Session expired. Please login again.', 'error');
                    logout();
                    return;
                }

                if (response.ok) {
                    const blob = await response.blob();
                    const url = window.URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = 'calendly_' + type + '_' + new Date().toISOString().slice(0,10) + '.csv';
                    document.body.appendChild(a);
                    a.click();
                    window.URL.revokeObjectURL(url);
                    a.remove();
                    showToast('Export downloaded!', 'success');
                } else {
                    const data = await response.json();
                    showToast(data.error || 'Export failed', 'error');
                }
            } catch (err) {
                showToast('Export error: ' + err.message, 'error');
            }
        }

        // Run Pipeline
        async function runPipeline() {
            showToast('Starting pipeline...', 'info');

            try {
                const response = await fetch('/force_start_pipeline', {
                    headers: { 'Authorization': 'Bearer ' + authToken }
                });

                const data = await response.json();

                if (response.ok || response.status === 202) {
                    showToast('Pipeline scheduled: ' + data.job_id, 'success');
                } else {
                    showToast(data.message || data.error || 'Failed to start pipeline', 'error');
                }
            } catch (err) {
                showToast('Error: ' + err.message, 'error');
            }
        }

        // Show Results
        function showResults(title, data) {
            document.getElementById('resultsTitle').textContent = title;
            document.getElementById('resultsContent').textContent = JSON.stringify(data, null, 2);
            document.getElementById('resultsPanel').style.display = 'block';
        }

        // Close Results
        function closeResults() {
            document.getElementById('resultsPanel').style.display = 'none';
        }

        // Toast Notifications
        function showToast(message, type = 'info') {
            const container = document.getElementById('toastContainer');
            const toast = document.createElement('div');
            toast.className = 'toast toast-' + type;
            toast.textContent = message;
            container.appendChild(toast);

            setTimeout(() => {
                toast.style.opacity = '0';
                setTimeout(() => toast.remove(), 300);
            }, 4000);
        }

        // Auto-refresh status every 30 seconds
        setInterval(loadStatus, 30000);
    </script>
</body>
</html>
"""


@app.route("/dashboard", methods=["GET"])
def dashboard():
    """Serve the modern frontend dashboard"""
    return render_template("dashboard.html")


@app.route("/static/<path:filename>")
def serve_static(filename):
    """Serve static files"""
    return send_from_directory(app.static_folder, filename)


@app.route("/docs", methods=["GET"])
def api_docs():
    """Serve API documentation."""
    docs_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Cleverly API Documentation</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; background: #0f172a; color: #f8fafc; }
            h1 { color: #6366f1; }
            h2 { color: #8b5cf6; margin-top: 30px; }
            .endpoint { background: #1e293b; padding: 15px; border-radius: 8px; margin: 10px 0; border-left: 4px solid #6366f1; }
            .method { font-weight: bold; color: #10b981; }
            .auth { color: #f59e0b; font-size: 12px; }
            code { background: #334155; padding: 2px 6px; border-radius: 4px; }
        </style>
    </head>
    <body>
        <h1>Cleverly API Documentation</h1>

        <h2>Authentication</h2>
        <div class="endpoint">
            <p><span class="method">POST</span> <code>/api/auth/login</code></p>
            <p>Authenticate and receive session token</p>
            <p>Body: <code>{"username": "...", "password": "..."}</code></p>
        </div>

        <h2>Calendly Data Extraction</h2>
        <div class="endpoint">
            <p><span class="method">GET</span> <code>/api/calendly/user-info</code> <span class="auth">[Auth Required]</span></p>
            <p>Get Calendly user information</p>
        </div>
        <div class="endpoint">
            <p><span class="method">GET</span> <code>/api/calendly/organization-members</code> <span class="auth">[Auth Required]</span></p>
            <p>Get organization members</p>
        </div>
        <div class="endpoint">
            <p><span class="method">GET</span> <code>/api/calendly/events</code> <span class="auth">[Auth Required]</span></p>
            <p>Get all event types</p>
        </div>
        <div class="endpoint">
            <p><span class="method">GET</span> <code>/api/calendly/invitees?days_back=365</code> <span class="auth">[Auth Required]</span></p>
            <p>Get invitees data</p>
        </div>
        <div class="endpoint">
            <p><span class="method">GET</span> <code>/api/calendly/comprehensive</code> <span class="auth">[Auth Required]</span></p>
            <p>Get all Calendly data</p>
        </div>
        <div class="endpoint">
            <p><span class="method">GET</span> <code>/api/calendly/export/{type}</code> <span class="auth">[Auth Required]</span></p>
            <p>Export data as CSV (type: events, invitees, members)</p>
        </div>

        <h2>Authentication Methods</h2>
        <p>Use either:</p>
        <ul>
            <li>Header: <code>Authorization: Bearer {token}</code></li>
            <li>Header: <code>X-API-Key: {api_key}</code></li>
        </ul>
    </body>
    </html>
    """
    return docs_html


@app.errorhandler(404)
def handle_404(error):
    """Handle 404 errors with helpful information."""
    logger.warning("404 error for path: %s - method: %s", request.path, request.method)
    available_endpoints = [
        "/", "/dashboard", "/docs", "/api/status",
        "/api/auth/login", "/api/auth/logout", "/api/auth/verify",
        "/api/calendly/user-info", "/api/calendly/organization-members",
        "/api/calendly/events", "/api/calendly/invitees",
        "/api/calendly/comprehensive", "/api/calendly/export/{type}",
        "/api/combined_csv", "/api/group_csv/<group>", "/api/json",
        "/force_start_pipeline",
    ]
    return jsonify({
        "error": "Endpoint not found",
        "path": request.path,
        "method": request.method,
        "available_endpoints": available_endpoints,
    }), 404


@app.errorhandler(Exception)
def handle_error(error):
    """Handle unexpected errors."""
    logger.error("API error: %s", error)
    return jsonify({"error": "Internal server error"}), 500


# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(host=Config.HOST, port=Config.PORT, debug=Config.DEBUG)
