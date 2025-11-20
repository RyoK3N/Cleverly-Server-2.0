#!/usr/bin/env python3
"""
combined_pipeline_v3.py
───────────────────────
Enhanced pipeline with additional Calendly event type tracking and origin tagging:
1. Fetch Monday.com board data and Calendly scheduled events (including cold-calling events)
2. Annotate Monday data with origin: LinkedIn, cold-email, or cold-calling
3. Persist timestamped CSVs, group-specific CSVs, and JSON for dashboards
4. Rich progress bars and detailed console logging throughout
"""
from __future__ import annotations

# ─── STANDARD LIB ────────────────────────────────────────────────────────────
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

# ─── 3RD-PARTY ───────────────────────────────────────────────────────────────
import dotenv
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm.auto import tqdm
from cleverly.core.config import Config

# Import Calendly data extraction modules
from cleverly.services.calendly.data_extractor import CCDataW, CalendlyAPIException
from cleverly.services.calendly.events import CCDWEvents, CalendlyAPIError as EventsAPIError
from cleverly.services.calendly.invitees import CCDWInvitees, CalendlyAPIError as InviteesAPIError

# ─── LOGGING SET-UP ───────────────────────────────────────────────────────────
dotenv.load_dotenv()
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)-8s %(name)s ─ %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("pipeline")


# ─── MONDAY SETTINGS & HELPERS ───────────────────────────────────────────────
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
        logger.debug("MondayDataProcessor initialized with board ID: %s",
                     cfg.BOARD_ID)

    def _items_to_df(self, items: List[dict]) -> pd.DataFrame:
        if not items or not items[0].get("column_values"):
            logger.debug("No items or column values found")
            return pd.DataFrame()

        cols = [c["id"] for c in items[0]["column_values"]]
        rows: list[dict] = []

        for it in items:
            row = {"Item ID": it["id"], "Item Name": it["name"]}
            for col in it["column_values"]:
                row[col["id"]] = col.get("text", "")
            rows.append(row)

        df = pd.DataFrame(rows, columns=["Item ID", "Item Name"] + cols)
        logger.debug("Created DataFrame with %d rows and %d columns", len(df),
                     len(df.columns))
        return df.rename(columns=self.cfg.COLUMN_MAPPING)

    def fetch(self) -> pd.DataFrame:
        logger.info("=" * 60)
        logger.info("STARTING MONDAY.COM DATA FETCH")
        logger.info("=" * 60)

        logger.info("Fetching Monday groups metadata from board %s…",
                    self.cfg.BOARD_ID)
        groups = fetch_groups(self.cfg.BOARD_ID, self.key)
        logger.info("✓ Successfully fetched %d groups from board", len(groups))

        segments: list[pd.DataFrame] = []
        total_items = 0

        with tqdm(total=len(self.cfg.GROUP_MAPPING),
                  desc="Processing Monday groups",
                  unit="group") as pbar:
            for gid, nice_name in self.cfg.GROUP_MAPPING.items():
                grp_meta = next((g for g in groups if g["id"] == gid), None)
                if not grp_meta:
                    logger.warning(
                        "⚠ Group %s (%s) not present on board – skipping", gid,
                        nice_name)
                    pbar.update(1)
                    continue

                logger.info("Fetching items for group: %s (%s)", nice_name,
                            gid)
                items = fetch_items_recursive(self.cfg.BOARD_ID, gid, self.key,
                                              self.cfg.ITEMS_LIMIT)
                logger.info("✓ Retrieved %d items from group '%s'", len(items),
                            nice_name)
                total_items += len(items)

                df = self._items_to_df(items)
                if not df.empty:
                    df["Group"] = nice_name
                    segments.append(df)
                    logger.debug("Added %d rows to segments for group '%s'",
                                 len(df), nice_name)

                pbar.update(1)

        combined = pd.concat(
            segments, ignore_index=True) if segments else pd.DataFrame()
        logger.info("=" * 60)
        logger.info("MONDAY.COM FETCH COMPLETE")
        logger.info("Total items fetched: %d", total_items)
        logger.info("Combined DataFrame shape: %d rows × %d columns",
                    *combined.shape)
        logger.info("=" * 60)
        return combined


# ─── CALENDLY SETTINGS & HELPERS ──────────────────────────────────────────────
BASE = "https://api.calendly.com/"

# Enhanced event type configuration with cold-calling event
TARGET_EVENT_TYPES = {
    "cold-email": [
        "https://api.calendly.com/event_types/3f3b8e40-246e-4723-8690-d0de0419e231",
        "https://api.calendly.com/event_types/6b4aa5e3-b4a2-4ef2-b1b2-1405b02e9806",
    ],
    "cold-calling": [
        "https://api.calendly.com/event_types/29ebb518-0491-41e8-888d-157b993f2706",
    ]
}

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
            r = session.get(url,
                            headers=HEADERS_CAL,
                            params=params,
                            timeout=30)
            if r.status_code == 429:
                delay = int(r.headers.get("Retry-After", CAL_THROTTLE))
                logger.warning("⚠ Rate limit hit (429). Sleeping %ds…", delay)
                time.sleep(delay)
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            logger.warning("Calendly GET failed (%s) – retry %d/%d", e, i + 1,
                           retries)
            if i == retries - 1:
                logger.error("All retries exhausted for URL: %s", url)
                return {"collection": []}
            time.sleep(2**i)
    return {"collection": []}


def _paginate_cal(url: str, params: dict) -> list[dict]:
    out: list[dict] = []
    page_count = 0

    while url:
        page_count += 1
        logger.debug("Fetching page %d from Calendly API…", page_count)
        page = _get_cal(url, params)
        items = page.get("collection", [])
        out.extend(items)
        logger.debug("Retrieved %d items from page %d", len(items), page_count)

        url = page.get("pagination", {}).get("next_page")
        params = None

    logger.debug("Pagination complete. Total items: %d across %d pages",
                 len(out), page_count)
    return out


def _org_uri() -> str:
    logger.debug("Fetching organization URI from Calendly…")
    org = _get_cal(urljoin(BASE,
                           "users/me")).get("resource",
                                            {}).get("current_organization", "")
    logger.debug("Organization URI: %s", org)
    return org


def list_events(status: str, cutoff_iso: str) -> list[dict]:
    url = urljoin(BASE, "scheduled_events")
    params = {
        "organization": _org_uri(),
        "status": status,
        "min_start_time": cutoff_iso,
        "count": 100,
    }
    logger.info("Fetching %s events since %s…", status, cutoff_iso[:10])
    events = _paginate_cal(url, params)
    logger.info("✓ Retrieved %d %s events", len(events), status)
    return events


def fetch_calendly() -> pd.DataFrame:
    logger.info("=" * 60)
    logger.info("STARTING CALENDLY DATA FETCH")
    logger.info("=" * 60)

    cutoff_iso = (datetime.now(timezone.utc) - timedelta(days=120)).isoformat()
    logger.info("Cutoff date: %s", cutoff_iso[:10])

    # Fetch all events
    raw = list_events("active", cutoff_iso) + list_events(
        "canceled", cutoff_iso)
    logger.info("Total raw events fetched: %d", len(raw))

    # Categorize events by type
    all_target_types = []
    for origin, types in TARGET_EVENT_TYPES.items():
        all_target_types.extend(types)

    events = [e for e in raw if e.get("event_type") in all_target_types]
    logger.info("✓ Filtered to %d events matching target event types",
                len(events))

    # Log breakdown by origin
    for origin, types in TARGET_EVENT_TYPES.items():
        count = len([e for e in events if e.get("event_type") in types])
        logger.info("  • %s events: %d", origin, count)

    df = pd.DataFrame(events)
    if df.empty:
        logger.warning("⚠ No Calendly events found")
        return df

    df.sort_values("start_time", ascending=False, inplace=True)
    df["invitee_name"] = None
    df["invitee_email"] = None
    df["origin"] = None

    # Tag origin based on event type
    for origin, types in TARGET_EVENT_TYPES.items():
        df.loc[df["event_type"].isin(types), "origin"] = origin

    logger.info("Starting invitee data enrichment for %d events…", len(df))

    def _fetch_invitees(uri: str):
        for st in ("active", "canceled"):
            j = _get_cal(f"{uri}/invitees", {"status": st, "count": 100})
            coll = j.get("collection") or []
            if coll:
                return coll[0].get("name"), coll[0].get("email")
        return None, None

    with ThreadPoolExecutor(max_workers=CAL_MAX_WORKERS) as ex:
        futures = {
            ex.submit(_fetch_invitees, u): i
            for i, u in df["uri"].items()
        }

        with tqdm(total=len(futures), desc="Invitee look-ups",
                  unit="evt") as pbar:
            for fut in as_completed(futures):
                i = futures[fut]
                try:
                    name, email = fut.result()
                    df.at[i, "invitee_name"] = name
                    df.at[i, "invitee_email"] = email
                except Exception as e:
                    logger.debug("Invitee fetch failed for event index %d: %s",
                                 i, e)
                    df.at[i, "invitee_name"] = None
                    df.at[i, "invitee_email"] = None
                pbar.update(1)

    enriched_count = df["invitee_email"].notna().sum()
    logger.info("✓ Enriched %d/%d events with invitee data", enriched_count,
                len(df))
    logger.info("=" * 60)
    logger.info("CALENDLY FETCH COMPLETE")
    logger.info("=" * 60)
    return df


# ─── OUTPUT FOLDERS ───────────────────────────────────────────────────────────
OUTPUT_ROOT = Path(os.getenv("OUTPUT_DIR", "sessions"))
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


# ─── ORIGIN ANNOTATION ────────────────────────────────────────────────────────
def annotate_origins(monday_df: pd.DataFrame,
                     calendly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Annotate Monday data with origin tags based on Calendly invitee emails:
    - cold-email: Email matches cold-email event type invitees
    - cold-calling: Email matches cold-calling event type invitees
    - LinkedIn: Default for all others
    """
    logger.info("=" * 60)
    logger.info("ANNOTATING LEAD ORIGINS")
    logger.info("=" * 60)

    if calendly_df.empty:
        logger.warning("⚠ No Calendly data available for origin annotation")
        monday_df["origin"] = "LinkedIn"
        return monday_df

    # Separate emails by origin
    cold_email_emails = set(calendly_df[calendly_df["origin"] == "cold-email"]
                            ["invitee_email"].dropna())
    cold_calling_emails = set(calendly_df[
        calendly_df["origin"] == "cold-calling"]["invitee_email"].dropna())

    logger.info("Calendly invitee breakdown:")
    logger.info("  • Cold-email invitees: %d unique emails",
                len(cold_email_emails))
    logger.info("  • Cold-calling invitees: %d unique emails",
                len(cold_calling_emails))

    # Default to LinkedIn
    monday_df["origin"] = "LinkedIn"

    # Tag cold-calling first (more specific)
    cold_calling_mask = monday_df["Email"].isin(cold_calling_emails)
    monday_df.loc[cold_calling_mask, "origin"] = "cold-calling"

    # Then tag cold-email
    cold_email_mask = monday_df["Email"].isin(cold_email_emails)
    monday_df.loc[cold_email_mask, "origin"] = "cold-email"

    # Log results
    origin_counts = monday_df["origin"].value_counts()
    logger.info("=" * 60)
    logger.info("ORIGIN ANNOTATION COMPLETE")
    logger.info("Monday.com lead origin breakdown:")
    for origin, count in origin_counts.items():
        percentage = (count / len(monday_df)) * 100
        logger.info("  • %s: %d leads (%.1f%%)", origin, count, percentage)
    logger.info("=" * 60)

    return monday_df


# ─── MAIN PIPELINE ────────────────────────────────────────────────────────────
# ─── CALENDLY COMPREHENSIVE DATA EXTRACTION ─────────────────────────────────
def extract_calendly_comprehensive_data(session_dir: Path, timestamp: str) -> bool:
    """
    Extract comprehensive Calendly data using CCDataW, CCDWEvents, and CCDWInvitees.
    Saves all data to the session directory for API access.

    Args:
        session_dir: Directory to save extracted data
        timestamp: Timestamp string for file naming

    Returns:
        bool: True if extraction successful, False otherwise
    """
    logger.info("=" * 60)
    logger.info("STARTING CALENDLY COMPREHENSIVE DATA EXTRACTION")
    logger.info("=" * 60)

    calendly_token = os.getenv("CALENDLY_ACCESS_TOKEN", "")
    if not calendly_token:
        logger.warning("CALENDLY_ACCESS_TOKEN not set - skipping comprehensive extraction")
        return False

    calendly_dir = session_dir / "calendly"
    calendly_dir.mkdir(parents=True, exist_ok=True)

    extraction_results = {
        "user_data": None,
        "organization_members": None,
        "events_data": None,
        "invitees_data": None,
        "extraction_timestamp": datetime.now().isoformat(),
        "success": False
    }

    try:
        # 1. Extract User and Organization Data using CCDataW
        logger.info("Step 1/3: Extracting user and organization data...")
        try:
            user_extractor = CCDataW(access_token=calendly_token)
            user_info = user_extractor.get_user_information()
            org_members = user_extractor.get_organization_members()

            user_data = {
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
                "organization_members": [
                    {
                        "name": m.name,
                        "email": m.email,
                        "user_uri": m.user_uri,
                        "role": m.role,
                        "status": m.status,
                        "organization_uri": m.organization_uri,
                        "membership_uri": m.membership_uri
                    }
                    for m in org_members
                ],
                "metadata": {
                    "extracted_at": datetime.now().isoformat(),
                    "total_members": len(org_members)
                }
            }

            # Save user data
            user_json_path = calendly_dir / f"calendly_user_data_{timestamp}.json"
            user_json_path.write_text(json.dumps(user_data, indent=2), encoding="utf-8")
            logger.info("✓ User data saved → %s", user_json_path)

            # Save members as CSV
            if org_members:
                members_df = pd.DataFrame([
                    {
                        "name": m.name,
                        "email": m.email,
                        "user_uri": m.user_uri,
                        "role": m.role,
                        "status": m.status
                    }
                    for m in org_members
                ])
                members_csv_path = calendly_dir / f"calendly_members_{timestamp}.csv"
                members_df.to_csv(members_csv_path, index=False)
                logger.info("✓ Members CSV saved → %s (%d members)", members_csv_path, len(org_members))

            extraction_results["user_data"] = user_data["user_information"]
            extraction_results["organization_members"] = user_data["organization_members"]

        except CalendlyAPIException as e:
            logger.error("Failed to extract user data: %s", e)

        # 2. Extract Event Types using CCDWEvents
        logger.info("Step 2/3: Extracting event types data...")
        try:
            events_extractor = CCDWEvents(access_token=calendly_token)
            events_data = events_extractor.extract_comprehensive_events()

            # Save events JSON
            events_json_path = calendly_dir / f"calendly_events_{timestamp}.json"
            events_json_path.write_text(json.dumps(events_data, indent=2, default=str), encoding="utf-8")
            logger.info("✓ Events data saved → %s", events_json_path)

            # Save events as CSV
            all_events = events_data.get("organization_events", []) + events_data.get("user_events", [])
            if all_events:
                events_df = pd.DataFrame(all_events)
                events_csv_path = calendly_dir / f"calendly_events_{timestamp}.csv"
                events_df.to_csv(events_csv_path, index=False)
                logger.info("✓ Events CSV saved → %s (%d events)", events_csv_path, len(all_events))

            extraction_results["events_data"] = events_data.get("summary", {})
            events_extractor.close()

        except EventsAPIError as e:
            logger.error("Failed to extract events data: %s", e)

        # 3. Extract Invitees using CCDWInvitees
        logger.info("Step 3/3: Extracting invitees data...")
        try:
            invitees_extractor = CCDWInvitees(access_token=calendly_token)
            invitees_data = invitees_extractor.extract_invitees_comprehensive(days_back=365)

            # Save invitees JSON
            invitees_json_path = calendly_dir / f"calendly_invitees_{timestamp}.json"
            invitees_json_path.write_text(json.dumps(invitees_data, indent=2, default=str), encoding="utf-8")
            logger.info("✓ Invitees data saved → %s", invitees_json_path)

            # Save invitees as CSV
            invitees_list = invitees_data.get("invitees", [])
            if invitees_list:
                invitees_df = pd.DataFrame(invitees_list)
                invitees_csv_path = calendly_dir / f"calendly_invitees_{timestamp}.csv"
                invitees_df.to_csv(invitees_csv_path, index=False)
                logger.info("✓ Invitees CSV saved → %s (%d invitees)", invitees_csv_path, len(invitees_list))

            extraction_results["invitees_data"] = invitees_data.get("summary", {})
            invitees_extractor.close()

        except InviteesAPIError as e:
            logger.error("Failed to extract invitees data: %s", e)

        # Save comprehensive extraction summary
        extraction_results["success"] = True
        summary_path = calendly_dir / f"calendly_extraction_summary_{timestamp}.json"
        summary_path.write_text(json.dumps(extraction_results, indent=2, default=str), encoding="utf-8")
        logger.info("✓ Extraction summary saved → %s", summary_path)

        logger.info("=" * 60)
        logger.info("CALENDLY COMPREHENSIVE EXTRACTION COMPLETE")
        logger.info("=" * 60)

        return True

    except Exception as e:
        logger.error("Calendly comprehensive extraction failed: %s", e)
        extraction_results["error"] = str(e)

        # Save error summary
        error_path = calendly_dir / f"calendly_extraction_error_{timestamp}.json"
        error_path.write_text(json.dumps(extraction_results, indent=2, default=str), encoding="utf-8")

        return False


def main():
    logger.info("=== Combined pipeline start ===")

    # Fetch data concurrently
    logger.info(
        "Initiating concurrent data fetch from Monday.com and Calendly…")
    with ThreadPoolExecutor(max_workers=2) as ex:
        mon_future = ex.submit(MondayDataProcessor(MondayConfig()).fetch)
        cal_future = ex.submit(fetch_calendly)

        monday_df = mon_future.result()
        calendly_df = cal_future.result()

    # Validation
    if monday_df.empty:
        logger.error("No Monday data – aborting")
        sys.exit(1)
    if calendly_df.empty:
        logger.error("No Calendly data – aborting")
        sys.exit(1)

    # Annotate origins
    monday_df = annotate_origins(monday_df, calendly_df)

    # Persist combined CSV
    logger.info("=" * 60)
    logger.info("PERSISTING DATA TO DISK")
    logger.info("=" * 60)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    combined_csv = OUTPUT_ROOT / f"combined_output_{ts}.csv"
    monday_df.to_csv(combined_csv, index=False)
    logger.info("Combined CSV written → %s", combined_csv)

    # Build/refresh session dir
    session_dir = OUTPUT_ROOT / "sessions"
    if session_dir.exists():
        logger.info("Cleaning existing sessions directory…")
        shutil.rmtree(session_dir)
    session_dir.mkdir(parents=True)
    logger.debug("Session dir prepared: %s", session_dir)

    # Split by Group
    grouped = {g: df for g, df in monday_df.groupby("Group", sort=False)}
    logger.info("Grouped data into %d distinct groups", len(grouped))

    for grp, gdf in tqdm(grouped.items(), desc="Saving group CSVs",
                         unit="grp"):
        grp_csv = session_dir / f"{grp}_{ts}.csv"
        gdf.to_csv(grp_csv, index=False)
        logger.debug("Group %s CSV → %s", grp, grp_csv)

    logger.info("✓ All group CSVs saved to %s", session_dir)

    # Create consolidated JSON
    consolidated = {k: v.to_dict(orient="records") for k, v in grouped.items()}
    json_path = session_dir / f"data_{ts}.json"
    json_path.write_text(json.dumps(consolidated, indent=2), encoding="utf-8")
    logger.info("Grouped JSON written → %s", json_path)

    # Extract comprehensive Calendly data using all three extraction scripts
    logger.info("Starting comprehensive Calendly data extraction...")
    calendly_success = extract_calendly_comprehensive_data(session_dir, ts)
    if calendly_success:
        logger.info("✓ Calendly comprehensive extraction completed successfully")
    else:
        logger.warning("⚠ Calendly comprehensive extraction completed with warnings or errors")

    logger.info("=== Pipeline completed successfully ===")


# ─── CLI ENTRY-POINT ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
