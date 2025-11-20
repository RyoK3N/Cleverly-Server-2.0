from __future__ import annotations
import os
import sys
import time
import json
import shutil
import logging
from pathlib import Path
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urljoin
from typing import List, Dict
import dotenv
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm.auto import tqdm
from cleverly.core.config import Config

# Logging set-up
dotenv.load_dotenv()
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)-8s %(name)s ─ %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("pipeline")

# Monday settings & helpers
class MondayConfig:
    BOARD_ID: str = os.getenv("MONDAY_BOARD_ID", "6942829967")
    ITEMS_LIMIT: int = int(os.getenv("MONDAY_ITEMS_LIMIT", "500"))
    GROUP_MAPPING: Dict[str, str] = {
        "topics": "scheduled",
        "new_group34578__1": "unqualified",
        "new_group27351__1": "won",
        "new_group54376__1": "cancelled",
        "new_group64021__1": "noshow",
        "new_group65903__1": "proposal",
        "new_group62617__1": "lost",
    }
    COLUMN_MAPPING: Dict[str, str] = {
        "name": "Name",
        "auto_number__1": "Auto number",
        "person": "Owner",
        "last_updated__1": "Last updated",
        "link__1": "Linkedin",
        "phone__1": "Phone",
        "email__1": "Email",
        "text7__1": "Company",
        "date4": "Sales Call Date",
        "status9__1": "Follow Up Tracker",
        "notes__1": "Notes",
        "interested_in__1": "Interested In",
        "status4__1": "Plan Type",
        "numbers__1": "Deal Value",
        "status6__1": "Email Template #1",
        "dup__of_email_template__1": "Email Template #2",
        "status__1": "Deal Status",
        "status2__1": "Send Panda Doc?",
        "utm_source__1": "UTM Source",
        "date__1": "Deal Status Date",
        "utm_campaign__1": "UTM Campaign",
        "utm_medium__1": "UTM Medium",
        "utm_content__1": "UTM Content",
        "link3__1": "UTM LINK",
        "lead_source8__1": "Lead Source",
        "color__1": "Channel FOR FUNNEL METRICS",
        "subitems__1": "Subitems",
        "date5__1": "Date Created",
    }

try:
    from cleverly.services.monday.extractor import fetch_items_recursive, fetch_groups
except ImportError as e:
    logger.error("Cannot import Monday helpers: %s", e)
    sys.exit(1)

class MondayDataProcessor:
    def __init__(self, cfg: MondayConfig):
        api_key = os.getenv("MONDAY_API_KEY")
        if not api_key:
            logger.critical("MONDAY_API_KEY not set in environment")
            sys.exit(1)
        self.key = api_key
        self.cfg = cfg

    def _items_to_df(self, items: List[dict]) -> pd.DataFrame:
        if not items or not items[0].get("column_values"):
            return pd.DataFrame()
        cols = [c["id"] for c in items[0]["column_values"]]
        rows: list[dict] = []
        for it in items:
            row = {"Item ID": it["id"], "Item Name": it["name"]}
            for col in it["column_values"]:
                row[col["id"]] = col.get("text", "")
            rows.append(row)
        df = pd.DataFrame(rows, columns=["Item ID", "Item Name"] + cols)
        return df.rename(columns=self.cfg.COLUMN_MAPPING)

    def fetch(self) -> pd.DataFrame:
        logger.info("Fetching Monday groups metadata…")
        groups = fetch_groups(self.cfg.BOARD_ID, self.key)
        logger.debug("Fetched %d groups from board %s", len(groups), self.cfg.BOARD_ID)

        segments: list[pd.DataFrame] = []
        for gid, nice_name in tqdm(self.cfg.GROUP_MAPPING.items(), desc="Monday groups", unit="grp"):
            grp_meta = next((g for g in groups if g["id"] == gid), None)
            if not grp_meta:
                logger.debug("Group %s not present on board – skipping", gid)
                continue
            logger.debug("Fetching items for group %s (%s)…", gid, nice_name)
            items = fetch_items_recursive(self.cfg.BOARD_ID, gid, self.key, self.cfg.ITEMS_LIMIT)
            logger.debug("Fetched %d items from %s", len(items), nice_name)
            df = self._items_to_df(items)
            if not df.empty:
                df["Group"] = nice_name
                segments.append(df)
        combined = pd.concat(segments, ignore_index=True) if segments else pd.DataFrame()
        logger.info("Monday data: %d rows, %d columns", *combined.shape)
        return combined

# Calendly settings & helpers
BASE = "https://api.calendly.com/"
TARGET_EVENT_TYPES = [
    "https://api.calendly.com/event_types/3f3b8e40-246e-4723-8690-d0de0419e231",
    "https://api.calendly.com/event_types/6b4aa5e3-b4a2-4ef2-b1b2-1405b02e9806",
]
CAL_MAX_WORKERS = int(os.getenv("CAL_MAX_WORKERS", 5))
CAL_THROTTLE = float(os.getenv("CAL_THROTTLE", 0.2))

CAL_KEY = os.getenv("CALENDLY_API_KEY")
if not CAL_KEY:
    logger.critical("CALENDLY_API_KEY not set in environment")
    sys.exit(1)

HEADERS_CAL = {
    "Authorization": f"Bearer {CAL_KEY}",
    "Content-Type": "application/json"
}

session = requests.Session()
session.mount(
    "https://",
    HTTPAdapter(max_retries=Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )),
)

def _get_cal(url: str, params: dict | None = None, retries: int = 3) -> dict:
    for i in range(retries):
        try:
            r = session.get(url, headers=HEADERS_CAL, params=params, timeout=30)
            if r.status_code == 429:
                delay = int(r.headers.get("Retry-After", CAL_THROTTLE))
                logger.debug("429 received. Sleeping %ds…", delay)
                time.sleep(delay)
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            logger.warning("Calendly GET failed (%s) – retry %d/%d", e, i + 1, retries)
            if i == retries - 1:
                return {"collection": []}
            time.sleep(2**i)
    return {"collection": []}

def _paginate_cal(url: str, params: dict) -> list[dict]:
    out: list[dict] = []
    while url:
        page = _get_cal(url, params)
        out.extend(page.get("collection", []))
        url = page.get("pagination", {}).get("next_page")
        params = None
    return out

def _org_uri() -> str:
    return _get_cal(urljoin(BASE, "users/me")).get("resource", {}).get("current_organization", "")

def list_events(status: str, cutoff_iso: str) -> list[dict]:
    url = urljoin(BASE, "scheduled_events")
    params = {
        "organization": _org_uri(),
        "status": status,
        "min_start_time": cutoff_iso,
        "count": 100,
    }
    return _paginate_cal(url, params)

def fetch_calendly() -> pd.DataFrame:
    cutoff_iso = (datetime.now(timezone.utc) - timedelta(days=120)).isoformat()
    logger.info("Fetching Calendly events (since %s)…", cutoff_iso[:10])
    raw = list_events("active", cutoff_iso) + list_events("canceled", cutoff_iso)
    events = [e for e in raw if e.get("event_type") in TARGET_EVENT_TYPES]
    logger.info("Calendly events matching target types: %d", len(events))

    df = pd.DataFrame(events)
    if df.empty:
        return df

    df.sort_values("start_time", ascending=False, inplace=True)
    df["invitee_name"] = None
    df["invitee_email"] = None

    def _fetch_invitees(uri: str):
        for st in ("active", "canceled"):
            j = _get_cal(f"{uri}/invitees", {"status": st, "count": 100})
            coll = j.get("collection") or []
            if coll:
                return coll[0].get("name"), coll[0].get("email")
        return None, None

    with ThreadPoolExecutor(max_workers=CAL_MAX_WORKERS) as ex:
        futures = {ex.submit(_fetch_invitees, u): i for i, u in df["uri"].items()}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Invitee look-ups", unit="evt"):
            i = futures[fut]
            try:
                name, email = fut.result()
            except Exception as e:
                logger.debug("Invitee fetch failed: %s", e)
                name = email = None
            df.at[i, "invitee_name"] = name
            df.at[i, "invitee_email"] = email
    logger.info("Calendly invitees enriched")
    return df

# Output folders
OUTPUT_ROOT = Path(os.getenv("OUTPUT_DIR", "sessions"))
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# Main pipeline
def main():
    logger.info("=== Combined pipeline start ===")

    with ThreadPoolExecutor(max_workers=1) as ex:
        mon_future = ex.submit(MondayDataProcessor(MondayConfig()).fetch)
        cal_future = ex.submit(fetch_calendly)

        monday_df = mon_future.result()
        calendly_df = cal_future.result()

    if monday_df.empty:
        logger.error("No Monday data – aborting")
        sys.exit(1)
    if calendly_df.empty:
        logger.error("No Calendly data – aborting")
        sys.exit(1)

    # Annotate origin
    invitee_emails = set(calendly_df["invitee_email"].dropna())
    monday_df["origin"] = "LinkedIn"
    monday_df.loc[monday_df["Email"].isin(invitee_emails), "origin"] = "cold-email"
    logger.info("Lead origins annotated")

    # Persist combined CSV
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    combined_csv = OUTPUT_ROOT / f"combined_output_{ts}.csv"
    monday_df.to_csv(combined_csv, index=False)
    logger.info("Combined CSV written → %s", combined_csv)

    # Build/refresh session dir
    session_dir = OUTPUT_ROOT / "sessions"
    if session_dir.exists():
        shutil.rmtree(session_dir)
    session_dir.mkdir(parents=True)
    logger.debug("Session dir prepared: %s", session_dir)

    # Split by Group
    grouped = {g: df for g, df in monday_df.groupby("Group", sort=False)}

    for grp, gdf in tqdm(grouped.items(), desc="Saving group CSVs", unit="grp"):
        grp_csv = session_dir / f"{grp}_{ts}.csv"
        gdf.to_csv(grp_csv, index=False)
        logger.debug("Group %s CSV → %s", grp, grp_csv)

    consolidated = {k: v.to_dict(orient="records") for k, v in grouped.items()}
    json_path = session_dir / f"data_{ts}.json"
    json_path.write_text(json.dumps(consolidated, indent=2), encoding="utf-8")
    logger.info("Grouped JSON written → %s", json_path)

    logger.info("=== Pipeline completed successfully ===")
