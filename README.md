# Cleverly Data Hub

A production-grade enterprise data management platform for synchronizing and analyzing data from Monday.com and Calendly.

## Features

- **Data Pipeline**: Automated data extraction from Monday.com and Calendly
- **Modern Dashboard**: Beautiful, responsive UI with real-time log viewing
- **Authentication**: Session-based login and API key support
- **Multiple Data Formats**: Export to CSV, JSON, and Parquet
- **Calendly Integration**: Complete data extraction including users, events, and invitees
- **RESTful API**: Secure endpoints for programmatic access

## Project Structure

```
cleverly-server/
├── src/cleverly/              # Main application package
│   ├── api/                   # API layer
│   │   ├── routes/            # Endpoint handlers
│   │   └── middleware/        # Auth decorators
│   ├── services/              # Business logic
│   │   ├── calendly/          # Calendly data extraction
│   │   ├── monday/            # Monday.com integration
│   │   └── pipeline/          # Data pipeline runners
│   ├── core/                  # Core utilities
│   │   ├── config.py          # Configuration management
│   │   └── security.py        # API key management
│   └── app.py                 # Flask application
├── static/                    # CSS, JS assets
├── templates/                 # HTML templates
├── tests/                     # Test suite
├── data/                      # Runtime data
├── logs/                      # Application logs
├── docs/                      # Documentation
├── scripts/                   # Utility scripts
├── .env.example               # Environment template
├── pyproject.toml             # Python packaging config
├── Makefile                   # Build commands
├── gunicorn.conf.py           # Production server config
└── wsgi.py                    # WSGI entry point
```

## Quick Start

### Prerequisites

- Python 3.9+
- pip (Python package manager)
- Monday.com API key
- Calendly access token

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Cleverly-server
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   # Using make (recommended)
   make install

   # Or manually
   pip install -e .

   # For development
   make dev
   ```

4. **Configure environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your credentials
   ```

5. **Initialize directories**
   ```bash
   make init-data
   ```

6. **Run the application**
   ```bash
   # Development
   make run

   # Or directly
   PYTHONPATH=src python -m cleverly.app

   # Production
   make prod
   ```

7. **Access the dashboard**
   - Open http://localhost:5000/dashboard
   - Login with default credentials: `admin` / `cleverly2024!`

## Environment Variables

Create a `.env` file with the following variables:

```env
# Monday.com Configuration
MONDAY_API_KEY=your_monday_api_key
MONDAY_BOARD_ID=6942829967

# Calendly Configuration
CALENDLY_API_KEY=your_calendly_api_key
CALENDLY_ACCESS_TOKEN=your_calendly_access_token

# Authentication (Change these in production!)
CLEVERLY_ADMIN_USER=admin
CLEVERLY_ADMIN_PASSWORD=cleverly2024!
CLEVERLY_USER=user
CLEVERLY_USER_PASSWORD=userpass2024!
CLEVERLY_API_KEY=your-custom-api-key
CLEVERLY_API_KEY_READONLY=your-readonly-key

# Application Settings
LOG_LEVEL=INFO
HOST=0.0.0.0
PORT=5000
DEBUG=False
```

## API Documentation

### Authentication

All protected endpoints require authentication using one of these methods:

**Session Token (for web UI)**
```http
Authorization: Bearer {session_token}
```

**API Key (for programmatic access)**
```http
X-API-Key: {api_key}
```

### Core Endpoints

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| POST | `/api/auth/login` | No | Authenticate and get session token |
| POST | `/api/auth/logout` | Yes | Invalidate session |
| GET | `/api/status` | No | System status and data availability |
| GET | `/force_start_pipeline` | Yes | Trigger data pipeline |

### Monday.com Data

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/combined_csv` | Download combined data as CSV |
| GET | `/api/json` | Get data in JSON format |
| GET | `/api/group_csv/{group}` | Download specific group data |

**Available groups**: `scheduled`, `unqualified`, `won`, `cancelled`, `noshow`, `proposal`, `lost`

### Calendly Live Extraction

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/calendly/user-info` | Get user profile |
| GET | `/api/calendly/organization-members` | List organization members |
| GET | `/api/calendly/events` | Extract event types |
| GET | `/api/calendly/invitees?days_back=365` | Extract invitees data |
| GET | `/api/calendly/comprehensive` | Full data extraction |
| GET | `/api/calendly/export/{type}` | Export as CSV |

### Calendly Session Data (from pipeline)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/calendly/session/status` | Check session data availability |
| GET | `/api/calendly/session/{type}` | Get cached data (user/events/invitees/summary) |
| GET | `/api/calendly/session/csv/{type}` | Download cached CSV (members/events/invitees) |

## API Key Management

The platform includes a comprehensive API key management system with cryptographic security, rate limiting, and audit logging.

### API Key Endpoints

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| POST | `/api/keys` | Session | Create a new API key |
| GET | `/api/keys` | Session | List all API keys |
| GET | `/api/keys/{key_id}` | Session | Get specific key details |
| DELETE | `/api/keys/{key_id}` | Session | Revoke an API key |
| GET | `/api/keys/code-examples` | Session | Get code examples for integration |

### Permission Levels

- **read_only**: Can only read data (GET requests)
- **standard**: Can read and write data
- **admin**: Full access including key management

### Creating API Keys

**Via Dashboard:**
1. Navigate to the "API Keys" section
2. Enter a key name and select permission level
3. Optionally set expiration and rate limit
4. Click "Create Key"
5. Copy the key immediately (it won't be shown again)

**Via API (curl):**
```bash
# Login to get session token
TOKEN=$(curl -s -X POST http://localhost:5000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "cleverly2024!"}' \
  | jq -r '.token')

# Create a new API key
curl -X POST http://localhost:5000/api/keys \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Production Integration",
    "permission": "standard",
    "expires_in_days": 90,
    "rate_limit": 100,
    "description": "Key for production data sync"
  }'
```

**Via API (Python):**
```python
import requests

# Login to get session token
login_response = requests.post(
    "http://localhost:5000/api/auth/login",
    json={"username": "admin", "password": "cleverly2024!"}
)
token = login_response.json()["token"]

# Create a new API key
response = requests.post(
    "http://localhost:5000/api/keys",
    headers={"Authorization": f"Bearer {token}"},
    json={
        "name": "Production Integration",
        "permission": "standard",
        "expires_in_days": 90,
        "rate_limit": 100,
        "description": "Key for production data sync"
    }
)

key_data = response.json()
api_key = key_data["key"]  # Save this securely!
print(f"Created key: {api_key}")
```

### Using API Keys for Data Access

Once you have an API key, use it in the `X-API-Key` header to access protected endpoints.

**Fetch Monday.com Data (curl):**
```bash
# Get combined CSV data
curl -X GET http://localhost:5000/api/combined_csv \
  -H "X-API-Key: clv_your_api_key_here" \
  -o monday_data.csv

# Get JSON data
curl -X GET http://localhost:5000/api/json \
  -H "X-API-Key: clv_your_api_key_here"

# Get specific group data
curl -X GET http://localhost:5000/api/group_csv/scheduled \
  -H "X-API-Key: clv_your_api_key_here" \
  -o scheduled.csv
```

**Fetch Monday.com Data (Python):**
```python
import requests
import pandas as pd
from io import StringIO

API_KEY = "clv_your_api_key_here"
BASE_URL = "http://localhost:5000"

headers = {"X-API-Key": API_KEY}

# Get JSON data
response = requests.get(f"{BASE_URL}/api/json", headers=headers)
data = response.json()

# Get CSV and load into pandas
response = requests.get(f"{BASE_URL}/api/combined_csv", headers=headers)
df = pd.read_csv(StringIO(response.text))
print(f"Loaded {len(df)} rows")

# Get specific group
response = requests.get(f"{BASE_URL}/api/group_csv/won", headers=headers)
won_df = pd.read_csv(StringIO(response.text))
```

**Fetch Calendly Data (curl):**
```bash
# Get user info
curl -X GET http://localhost:5000/api/calendly/user-info \
  -H "X-API-Key: clv_your_api_key_here"

# Get all events
curl -X GET http://localhost:5000/api/calendly/events \
  -H "X-API-Key: clv_your_api_key_here"

# Get invitees (last 30 days)
curl -X GET "http://localhost:5000/api/calendly/invitees?days_back=30" \
  -H "X-API-Key: clv_your_api_key_here"

# Export invitees as CSV
curl -X GET http://localhost:5000/api/calendly/export/invitees \
  -H "X-API-Key: clv_your_api_key_here" \
  -o invitees.csv

# Get session data (from pipeline)
curl -X GET http://localhost:5000/api/calendly/session/invitees \
  -H "X-API-Key: clv_your_api_key_here"
```

**Fetch Calendly Data (Python):**
```python
import requests

API_KEY = "clv_your_api_key_here"
BASE_URL = "http://localhost:5000"

headers = {"X-API-Key": API_KEY}

# Get comprehensive Calendly data
response = requests.get(
    f"{BASE_URL}/api/calendly/comprehensive",
    headers=headers
)
calendly_data = response.json()

print(f"User: {calendly_data['user_info']['name']}")
print(f"Events: {calendly_data['summary']['event_types_count']}")
print(f"Invitees: {calendly_data['summary']['total_invitees']}")

# Get invitees with custom date range
response = requests.get(
    f"{BASE_URL}/api/calendly/invitees",
    headers=headers,
    params={"days_back": 90}
)
invitees = response.json()
```

**Trigger Pipeline (curl):**
```bash
curl -X GET http://localhost:5000/force_start_pipeline \
  -H "X-API-Key: clv_your_api_key_here"
```

**Trigger Pipeline (Python):**
```python
import requests

API_KEY = "clv_your_api_key_here"
BASE_URL = "http://localhost:5000"

response = requests.get(
    f"{BASE_URL}/force_start_pipeline",
    headers={"X-API-Key": API_KEY}
)

result = response.json()
print(f"Pipeline status: {result['message']}")
```

### Managing API Keys

**List all keys (curl):**
```bash
curl -X GET http://localhost:5000/api/keys \
  -H "Authorization: Bearer $TOKEN"
```

**Revoke a key (curl):**
```bash
curl -X DELETE http://localhost:5000/api/keys/{key_id} \
  -H "Authorization: Bearer $TOKEN"
```

**List and manage keys (Python):**
```python
import requests

# List all keys
response = requests.get(
    "http://localhost:5000/api/keys",
    headers={"Authorization": f"Bearer {token}"}
)
keys = response.json()["keys"]

for key in keys:
    print(f"{key['name']}: {key['permission']} - {key['status']}")
    print(f"  Created: {key['created_at']}")
    print(f"  Last used: {key.get('last_used', 'Never')}")

# Revoke a specific key
key_id = "key_id_to_revoke"
response = requests.delete(
    f"http://localhost:5000/api/keys/{key_id}",
    headers={"Authorization": f"Bearer {token}"}
)
```

### Rate Limiting

API keys have configurable rate limits (requests per minute). Default is 100 requests/minute. When exceeded, requests return 429 status with retry information.

```python
response = requests.get(url, headers=headers)
if response.status_code == 429:
    retry_after = response.json().get("retry_after", 60)
    time.sleep(retry_after)
    response = requests.get(url, headers=headers)
```

### Integration Example: Sync to External System

```python
import requests
import pandas as pd
from io import StringIO

class CleverlyClient:
    def __init__(self, base_url, api_key):
        self.base_url = base_url
        self.headers = {"X-API-Key": api_key}

    def get_monday_data(self):
        """Fetch Monday.com data as DataFrame"""
        response = requests.get(
            f"{self.base_url}/api/combined_csv",
            headers=self.headers
        )
        response.raise_for_status()
        return pd.read_csv(StringIO(response.text))

    def get_calendly_invitees(self, days_back=30):
        """Fetch Calendly invitees"""
        response = requests.get(
            f"{self.base_url}/api/calendly/invitees",
            headers=self.headers,
            params={"days_back": days_back}
        )
        response.raise_for_status()
        return response.json()

    def trigger_pipeline(self):
        """Start data extraction pipeline"""
        response = requests.get(
            f"{self.base_url}/force_start_pipeline",
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()

# Usage
client = CleverlyClient(
    base_url="http://localhost:5000",
    api_key="clv_your_api_key_here"
)

# Sync Monday.com data
monday_df = client.get_monday_data()
monday_df.to_sql("monday_leads", db_connection, if_exists="replace")

# Sync Calendly data
invitees = client.get_calendly_invitees(days_back=7)
for invitee in invitees.get("invitees", []):
    sync_to_crm(invitee)
```

## Production Deployment

### Using Gunicorn (Recommended)

1. **Install Gunicorn**
   ```bash
   pip install gunicorn
   ```

2. **Create Gunicorn config**
   ```python
   # gunicorn.conf.py
   bind = "0.0.0.0:5000"
   workers = 4
   worker_class = "sync"
   timeout = 300
   keepalive = 5
   max_requests = 1000
   max_requests_jitter = 100
   accesslog = "logs/access.log"
   errorlog = "logs/error.log"
   loglevel = "info"
   ```

3. **Run with Gunicorn**
   ```bash
   gunicorn -c gunicorn.conf.py main_app:app
   ```

### Using Docker

1. **Create Dockerfile**
   ```dockerfile
   FROM python:3.11-slim

   WORKDIR /app

   # Install system dependencies
   RUN apt-get update && apt-get install -y \
       gcc \
       && rm -rf /var/lib/apt/lists/*

   # Copy requirements and install
   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt
   RUN pip install gunicorn

   # Copy application
   COPY . .

   # Create necessary directories
   RUN mkdir -p logs output/sessions

   # Expose port
   EXPOSE 5000

   # Health check
   HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
       CMD curl -f http://localhost:5000/ || exit 1

   # Run with Gunicorn
   CMD ["gunicorn", "-c", "gunicorn.conf.py", "main_app:app"]
   ```

2. **Create docker-compose.yml**
   ```yaml
   version: '3.8'

   services:
     cleverly:
       build: .
       ports:
         - "5000:5000"
       environment:
         - MONDAY_API_KEY=${MONDAY_API_KEY}
         - MONDAY_BOARD_ID=${MONDAY_BOARD_ID}
         - CALENDLY_API_KEY=${CALENDLY_API_KEY}
         - CALENDLY_ACCESS_TOKEN=${CALENDLY_ACCESS_TOKEN}
         - CLEVERLY_ADMIN_USER=${CLEVERLY_ADMIN_USER}
         - CLEVERLY_ADMIN_PASSWORD=${CLEVERLY_ADMIN_PASSWORD}
         - CLEVERLY_API_KEY=${CLEVERLY_API_KEY}
       volumes:
         - ./output:/app/output
         - ./logs:/app/logs
       restart: unless-stopped

   volumes:
     output:
     logs:
   ```

3. **Deploy**
   ```bash
   docker-compose up -d --build
   ```

### Using systemd (Linux)

1. **Create service file**
   ```ini
   # /etc/systemd/system/cleverly.service
   [Unit]
   Description=Cleverly Data Hub
   After=network.target

   [Service]
   Type=simple
   User=www-data
   Group=www-data
   WorkingDirectory=/opt/cleverly
   Environment="PATH=/opt/cleverly/venv/bin"
   EnvironmentFile=/opt/cleverly/.env
   ExecStart=/opt/cleverly/venv/bin/gunicorn -c gunicorn.conf.py main_app:app
   Restart=always
   RestartSec=10

   [Install]
   WantedBy=multi-user.target
   ```

2. **Enable and start service**
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable cleverly
   sudo systemctl start cleverly
   ```

### Nginx Reverse Proxy

```nginx
server {
    listen 80;
    server_name cleverly.yourdomain.com;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_connect_timeout 300s;
        proxy_read_timeout 300s;
    }

    location /static {
        alias /opt/cleverly/static;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
}
```

### SSL/HTTPS with Certbot

```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d cleverly.yourdomain.com
```

## Security Best Practices

1. **Change default credentials** - Update all default passwords in production
2. **Use environment variables** - Never commit secrets to version control
3. **Enable HTTPS** - Use SSL/TLS for all production traffic
4. **Firewall configuration** - Only expose necessary ports
5. **Regular updates** - Keep dependencies updated for security patches
6. **Log monitoring** - Monitor access and error logs for anomalies
7. **Rate limiting** - Consider adding rate limiting for API endpoints
8. **Session management** - In production, use Redis or database for sessions

## Monitoring & Maintenance

### Log Files

- `logs/access.log` - HTTP access logs
- `logs/error.log` - Application errors
- `ccdw_events_debug.log` - Calendly events extraction logs
- `ccdw_invitees_debug.log` - Calendly invitees extraction logs

### Health Check

```bash
curl http://localhost:5000/
# Response: {"status": "healthy", "message": "Cleverly API is running"}
```

### Data Locations

- `output/` - Combined CSV exports
- `output/sessions/` - Session-specific data
- `output/sessions/calendly/` - Calendly extraction data
- `output/exports/` - User-requested exports

### Scheduled Tasks

The pipeline runs automatically every 4 hours. You can also trigger it manually:

```bash
# Via API
curl -H "X-API-Key: your-api-key" http://localhost:5000/force_start_pipeline

# Via Dashboard
# Click "Start Pipeline" button in the Pipeline Operations section
```

## Troubleshooting

### Common Issues

**API returns 401 Unauthorized**
- Check that your API key or session token is valid
- Verify credentials in environment variables

**Pipeline fails to run**
- Verify MONDAY_API_KEY and CALENDLY_ACCESS_TOKEN are set
- Check log files for detailed error messages
- Ensure network access to external APIs

**No data in exports**
- Run the pipeline first to populate data
- Check API credentials are valid
- Verify board ID and organization settings

**Dashboard won't load**
- Check that static files exist in `/static` directory
- Verify templates exist in `/templates` directory
- Check browser console for JavaScript errors

### Getting Help

For issues and feature requests, please:
1. Check the logs for error messages
2. Review the API documentation at `/docs`
3. Report issues at contact@synexian.com

## License

Copyright 2024 Cleverly. All rights reserved.

## Version History

- **2.0.0** - Production dashboard with comprehensive Calendly integration
- **1.0.0** - Initial release with Monday.com pipeline
