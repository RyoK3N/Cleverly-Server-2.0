"""
CCDWInvitees - Cleverly Calendly Data Warehouse Invitees Module
Invitee data extraction pipeline for Calendly API
Author: Data Engineering Team
Version: 1.0.0
"""

import requests
import json
import pandas as pd
import logging
from typing import List, Dict, Optional, Any, Union, Iterator
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import time
import os
from pathlib import Path
import gzip
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(funcName)s - %(message)s',
    handlers=[
        logging.FileHandler('ccdw_invitees_debug.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class Invitee:
    """Data class for Calendly invitee information"""
    uri: str
    email: str
    name: str
    status: str
    questions_and_answers: List[Dict]
    timezone: str
    text_reminder_number: Optional[str]
    created_at: str
    updated_at: str
    event_uri: str
    event_name: str
    event_type_uri: str
    event_type_name: str
    scheduled_event_uri: str
    scheduling_url: str
    cancel_url: str
    reschedule_url: str
    tracking: Dict

@dataclass
class ScheduledEvent:
    """Data class for Calendly scheduled event information"""
    uri: str
    name: str
    status: str
    start_time: str
    end_time: str
    event_type: str
    location: Dict
    invitees_counter: Dict
    created_at: str
    updated_at: str
    event_memberships: List[Dict]
    event_guests: List[Dict]

@dataclass
class InviteeExtractionSummary:
    """Data class for invitee extraction summary statistics"""
    total_events_processed: int
    total_scheduled_events: int
    total_invitees: int
    total_invitee_requests: int
    successful_invitee_requests: int
    failed_invitee_requests: int
    extraction_timestamp: str
    duration_seconds: float
    data_size_mb: float

class CalendlyAPIError(Exception):
    """Custom exception for Calendly API errors"""
    pass

class CCDWInvitees:
    """
    Cleverly Calendly Data Warehouse Invitees - Invitee data extraction pipeline
    
    Features:
    - Comprehensive invitee data extraction from scheduled events
    - Multi-threaded data extraction for performance
    - Incremental data saving to handle large datasets
    - Advanced error handling and retry mechanisms
    - Multiple output formats (JSON, CSV, Parquet, Pickle)
    - Data partitioning for large datasets
    - Comprehensive analytics and monitoring
    """
    
    BASE_URL = "https://api.calendly.com"
    DEFAULT_TIMEOUT = 30
    MAX_RETRIES = 3
    RETRY_BACKOFF_FACTOR = 1.5
    MAX_THREADS = 5  # Conservative threading to avoid rate limits
    CHUNK_SIZE = 10000  # Number of records per file chunk
    
    def __init__(self, access_token: str, organization_uri: Optional[str] = None):
        """
        Initialize CCDWInvitees with Calendly access token
        
        Args:
            access_token (str): Calendly personal access token
            organization_uri (str, optional): Organization URI
            
        Raises:
            ValueError: If access token is invalid
            CalendlyAPIError: If initial connection test fails
        """
        self._setup_logging()
        logger.info("🚀 Initializing CCDWInvitees...")
        
        if not access_token or access_token == "your_personal_access_token_here":
            error_msg = "Valid Calendly access token is required"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        self.access_token = access_token
        self.organization_uri = organization_uri
        self.headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
            "User-Agent": "CCDWInvitees/1.0.0"
        }
        
        self._session = requests.Session()
        self._session.headers.update(self.headers)
        
        # Initialize metrics and state
        self._extraction_metrics: Dict[str, Any] = {
            'start_time': None,
            'end_time': None,
            'total_invitees': 0,
            'total_events': 0,
            'success': False
        }
        self._api_metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'rate_limited_requests': 0,
            'average_response_time': 0.0
        }
        
        self._thread_lock = threading.Lock()
        self._extracted_invitees: List[Invitee] = []
        self._extracted_events: List[ScheduledEvent] = []
        
        self._validate_connection()
        logger.info("✅ CCDWInvitees initialized successfully")

    def _setup_logging(self):
        """Setup comprehensive logging configuration"""
        logger.setLevel(logging.DEBUG)

    def _validate_connection(self) -> None:
        """
        Validate API connection and permissions
        
        Raises:
            CalendlyAPIError: If connection test fails
        """
        logger.info("🔍 Validating API connection...")
        try:
            response = self._session.get(
                f"{self.BASE_URL}/users/me",
                timeout=self.DEFAULT_TIMEOUT
            )
            response.raise_for_status()
            
            user_data = response.json()['resource']
            
            # Set organization URI if not provided
            if not self.organization_uri:
                self.organization_uri = user_data.get('current_organization')
                if not self.organization_uri:
                    raise CalendlyAPIError("No organization URI found and none provided")
            
            logger.info(f"✅ Connected as: {user_data.get('name')}")
            logger.info(f"🏢 Organization: {self.organization_uri}")
            
        except requests.exceptions.RequestException as e:
            error_msg = f"❌ Connection validation failed: {str(e)}"
            logger.error(error_msg)
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response status: {e.response.status_code}")
                logger.error(f"Response body: {e.response.text}")
            raise CalendlyAPIError(error_msg)

    def _build_url(self, endpoint: str) -> str:
        """
        Build proper URL for API requests
        
        Args:
            endpoint (str): API endpoint
            
        Returns:
            str: Properly formatted URL
        """
        if endpoint.startswith('http'):
            return endpoint
        if endpoint.startswith(self.BASE_URL):
            return endpoint
        if endpoint.startswith('/'):
            return f"{self.BASE_URL}{endpoint}"
        return f"{self.BASE_URL}/{endpoint}"

    def _make_api_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Make authenticated API request with advanced retry logic
        
        Args:
            endpoint (str): API endpoint
            params (Dict): Query parameters
            
        Returns:
            Dict: API response data
            
        Raises:
            CalendlyAPIError: If API request fails after retries
        """
        start_time = time.time()
        url = self._build_url(endpoint)
        
        logger.debug(f"📡 API Request - URL: {url}, Params: {params}")
        
        for attempt in range(self.MAX_RETRIES):
            try:
                with self._thread_lock:
                    self._api_metrics['total_requests'] += 1
                
                logger.debug(f"Attempt {attempt + 1}/{self.MAX_RETRIES} for {url}")
                
                response = self._session.get(
                    url,
                    params=params,
                    timeout=self.DEFAULT_TIMEOUT
                )
                
                response_time = time.time() - start_time
                
                if response.status_code == 429:
                    with self._thread_lock:
                        self._api_metrics['rate_limited_requests'] += 1
                    retry_after = int(response.headers.get('Retry-After', 
                                        self.RETRY_BACKOFF_FACTOR * (attempt + 1)))
                    logger.warning(f"⏳ Rate limited. Retrying after {retry_after} seconds")
                    time.sleep(retry_after)
                    continue
                
                response.raise_for_status()
                
                with self._thread_lock:
                    self._api_metrics['successful_requests'] += 1
                    current_avg = self._api_metrics['average_response_time']
                    successful = self._api_metrics['successful_requests']
                    self._api_metrics['average_response_time'] = (
                        (current_avg * (successful - 1) + response_time) / successful
                    )
                
                logger.debug(f"✅ API request successful: {response.status_code} in {response_time:.2f}s")
                return response.json()
                
            except requests.exceptions.RequestException as e:
                with self._thread_lock:
                    self._api_metrics['failed_requests'] += 1
                response_time = time.time() - start_time
                
                logger.error(f"❌ API request failed (attempt {attempt + 1}): {str(e)}")
                logger.error(f"Failed URL: {url}")
                
                if hasattr(e, 'response') and e.response is not None:
                    logger.error(f"Status code: {e.response.status_code}")
                    if e.response.status_code < 500:
                        try:
                            error_body = e.response.json()
                            logger.error(f"Error response: {json.dumps(error_body, indent=2)}")
                        except:
                            logger.error(f"Error response text: {e.response.text[:500]}...")
                
                if attempt == self.MAX_RETRIES - 1:
                    error_msg = f"🚨 API request failed after {self.MAX_RETRIES} attempts: {str(e)}"
                    logger.error(error_msg)
                    raise CalendlyAPIError(error_msg)
                
                sleep_time = self.RETRY_BACKOFF_FACTOR * (attempt + 1)
                logger.info(f"💤 Retrying in {sleep_time:.1f} seconds...")
                time.sleep(sleep_time)
        
        raise CalendlyAPIError("Unexpected error in API request")

    def _get_paginated_data(self, endpoint: str, params: Optional[Dict] = None, data_type: str = "items") -> List[Dict]:
        """
        Handle paginated API responses and return all data
        
        Args:
            endpoint (str): API endpoint
            params (Dict): Query parameters
            data_type (str): Description of data being fetched
            
        Returns:
            List[Dict]: All paginated data
        """
        all_data = []
        url = endpoint
        current_params = params.copy() if params else {}
        page_count = 0
        total_items = 0
        
        logger.info(f"📄 Starting paginated data extraction for {data_type}")
        
        while url:
            try:
                page_count += 1
                logger.info(f"📖 Fetching page {page_count} for {data_type}...")
                
                data = self._make_api_request(url, current_params)
                page_data = data.get('collection', [])
                items_count = len(page_data)
                total_items += items_count
                
                all_data.extend(page_data)
                
                logger.info(f"✅ Page {page_count}: {items_count} {data_type} (Total: {total_items})")
                
                # Check for next page
                pagination = data.get('pagination', {})
                next_page = pagination.get('next_page')
                
                if next_page:
                    url = next_page
                    current_params = {}
                else:
                    logger.info(f"📚 No more pages. Completed {data_type} extraction.")
                    url = None
                
            except CalendlyAPIError as e:
                logger.error(f"🚨 Pagination error at page {page_count} for {data_type}: {str(e)}")
                break
        
        logger.info(f"🎉 Completed {data_type} extraction: {total_items} items across {page_count} pages")
        return all_data

    def get_scheduled_events(self, days_back: int = 365, user_uri: Optional[str] = None) -> List[ScheduledEvent]:
        """
        Get scheduled events with date range filtering
        
        Args:
            days_back (int): Number of days to look back for events
            user_uri (str): Specific user URI to filter events
            
        Returns:
            List[ScheduledEvent]: List of scheduled events
        """
        logger.info(f"📅 Fetching scheduled events for the past {days_back} days...")
        
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            params = {
                "organization": self.organization_uri,
                "count": 100,
                "min_start_time": start_date.isoformat() + 'Z',
                "max_start_time": end_date.isoformat() + 'Z'
            }
            
            if user_uri:
                params["user"] = user_uri
            
            raw_events = self._get_paginated_data("/scheduled_events", params, "scheduled events")
            
            scheduled_events = []
            for event_data in raw_events:
                scheduled_event = ScheduledEvent(
                    uri=event_data.get('uri', ''),
                    name=event_data.get('name', ''),
                    status=event_data.get('status', ''),
                    start_time=event_data.get('start_time', ''),
                    end_time=event_data.get('end_time', ''),
                    event_type=event_data.get('event_type', ''),
                    location=event_data.get('location', {}),
                    invitees_counter=event_data.get('invitees_counter', {}),
                    created_at=event_data.get('created_at', ''),
                    updated_at=event_data.get('updated_at', ''),
                    event_memberships=event_data.get('event_memberships', []),
                    event_guests=event_data.get('event_guests', [])
                )
                scheduled_events.append(scheduled_event)
            
            logger.info(f"✅ Retrieved {len(scheduled_events)} scheduled events")
            return scheduled_events
            
        except Exception as e:
            error_msg = f"🚨 Scheduled events extraction failed: {str(e)}"
            logger.error(error_msg)
            raise CalendlyAPIError(error_msg)

    def get_invitees_for_event(self, event_uri: str, event_name: str = "Unknown") -> List[Invitee]:
        """
        Get invitees for a specific scheduled event
        
        Args:
            event_uri (str): Scheduled event URI
            event_name (str): Event name for logging
            
        Returns:
            List[Invitee]: List of invitees for the event
        """
        try:
            logger.debug(f"👥 Fetching invitees for event: {event_name}")
            
            params = {
                "count": 100
            }
            
            # Extract event type information from event URI if possible
            event_type_uri = ""
            event_type_name = "Unknown"
            
            raw_invitees = self._get_paginated_data(f"{event_uri}/invitees", params, f"invitees for {event_name}")
            
            invitees = []
            for invitee_data in raw_invitees:
                invitee = Invitee(
                    uri=invitee_data.get('uri', ''),
                    email=invitee_data.get('email', ''),
                    name=invitee_data.get('name', ''),
                    status=invitee_data.get('status', ''),
                    questions_and_answers=invitee_data.get('questions_and_answers', []),
                    timezone=invitee_data.get('timezone', ''),
                    text_reminder_number=invitee_data.get('text_reminder_number'),
                    created_at=invitee_data.get('created_at', ''),
                    updated_at=invitee_data.get('updated_at', ''),
                    event_uri=event_uri,
                    event_name=event_name,
                    event_type_uri=event_type_uri,
                    event_type_name=event_type_name,
                    scheduled_event_uri=invitee_data.get('scheduled_event', ''),
                    scheduling_url=invitee_data.get('scheduling_url', ''),
                    cancel_url=invitee_data.get('cancel_url', ''),
                    reschedule_url=invitee_data.get('reschedule_url', ''),
                    tracking=invitee_data.get('tracking', {})
                )
                invitees.append(invitee)
            
            logger.debug(f"✅ Retrieved {len(invitees)} invitees for event: {event_name}")
            return invitees
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to extract invitees for event {event_name}: {str(e)}")
            return []

    def _process_event_invitees(self, event: ScheduledEvent) -> Dict[str, Any]:
        """
        Process a single event to extract invitees (for threading)
        
        Args:
            event (ScheduledEvent): Scheduled event to process
            
        Returns:
            Dict: Processing results
        """
        try:
            invitees = self.get_invitees_for_event(event.uri, event.name)
            
            with self._thread_lock:
                self._extracted_invitees.extend(invitees)
                self._extracted_events.append(event)
            
            return {
                'event_uri': event.uri,
                'event_name': event.name,
                'invitees_count': len(invitees),
                'success': True
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to process event {event.name}: {str(e)}")
            return {
                'event_uri': event.uri,
                'event_name': event.name,
                'invitees_count': 0,
                'success': False,
                'error': str(e)
            }

    def extract_invitees_comprehensive(self, days_back: int = 365, max_workers: Optional[int] = None) -> Dict[str, Any]:
        """
        Comprehensive invitee data extraction with multi-threading
        
        Args:
            days_back (int): Number of days to look back for events
            max_workers (int): Maximum number of threads to use
            
        Returns:
            Dict: Comprehensive invitee data
        """
        start_time = time.time()
        self._extraction_metrics['start_time'] = datetime.now().isoformat()
        
        if max_workers is None:
            max_workers = self.MAX_THREADS
        
        logger.info(f"🚀 Starting comprehensive invitee data extraction (past {days_back} days, {max_workers} threads)...")
        
        try:
            # Get all scheduled events
            scheduled_events = self.get_scheduled_events(days_back)
            
            if not scheduled_events:
                logger.warning("⚠️ No scheduled events found for the specified period")
                return {}
            
            # Reset storage
            self._extracted_invitees = []
            self._extracted_events = []
            
            # Process events with threading
            logger.info(f"🔧 Processing {len(scheduled_events)} events with {max_workers} threads...")
            
            successful_events = 0
            failed_events = 0
            total_invitees = 0
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_event = {
                    executor.submit(self._process_event_invitees, event): event 
                    for event in scheduled_events
                }
                
                # Process completed tasks
                for i, future in enumerate(as_completed(future_to_event), 1):
                    event = future_to_event[future]
                    try:
                        result = future.result()
                        if result['success']:
                            successful_events += 1
                            total_invitees += result['invitees_count']
                            logger.info(f"✅ Processed {i}/{len(scheduled_events)}: {event.name} - {result['invitees_count']} invitees")
                        else:
                            failed_events += 1
                            logger.warning(f"⚠️ Failed {i}/{len(scheduled_events)}: {event.name}")
                    except Exception as e:
                        failed_events += 1
                        logger.error(f"❌ Exception processing {event.name}: {str(e)}")
            
            extraction_time = time.time() - start_time
            
            # Calculate data size
            data_size = len(json.dumps([asdict(inv) for inv in self._extracted_invitees])) / (1024 * 1024)  # MB
            
            summary = InviteeExtractionSummary(
                total_events_processed=len(scheduled_events),
                total_scheduled_events=len(self._extracted_events),
                total_invitees=len(self._extracted_invitees),
                total_invitee_requests=self._api_metrics['total_requests'],
                successful_invitee_requests=self._api_metrics['successful_requests'],
                failed_invitee_requests=self._api_metrics['failed_requests'],
                extraction_timestamp=datetime.now().isoformat(),
                duration_seconds=round(extraction_time, 2),
                data_size_mb=round(data_size, 2)
            )
            
            comprehensive_data = {
                'invitees': [asdict(inv) for inv in self._extracted_invitees],
                'scheduled_events': [asdict(event) for event in self._extracted_events],
                'summary': asdict(summary),
                'api_metrics': self._api_metrics.copy(),
                'metadata': {
                    'extraction_version': '1.0.0',
                    'source': 'CCDWInvitees',
                    'days_back': days_back,
                    'organization_uri': self.organization_uri
                }
            }
            
            # Update metrics
            self._extraction_metrics.update({
                'total_invitees': len(self._extracted_invitees),
                'total_events': len(self._extracted_events),
                'extraction_time_seconds': extraction_time,
                'success': True,
                'end_time': datetime.now().isoformat()
            })
            
            # Log comprehensive results
            logger.info("🎉 COMPREHENSIVE INVITEE EXTRACTION COMPLETED SUCCESSFULLY!")
            logger.info("=" * 70)
            logger.info(f"📊 EXTRACTION SUMMARY:")
            logger.info(f"   • Total Events Processed: {len(scheduled_events)}")
            logger.info(f"   • Successful Events: {successful_events}")
            logger.info(f"   • Failed Events: {failed_events}")
            logger.info(f"   • Total Invitees: {len(self._extracted_invitees)}")
            logger.info(f"   • Extraction Time: {extraction_time:.2f}s")
            logger.info(f"   • Data Size: {data_size:.2f} MB")
            logger.info(f"   • API Success Rate: {(self._api_metrics['successful_requests']/self._api_metrics['total_requests']*100):.1f}%")
            logger.info("=" * 70)
            
            return comprehensive_data
            
        except Exception as e:
            extraction_time = time.time() - start_time
            self._extraction_metrics.update({
                'success': False,
                'end_time': datetime.now().isoformat(),
                'error': str(e)
            })
            
            error_msg = f"🚨 Comprehensive invitee extraction failed after {extraction_time:.2f}s: {str(e)}"
            logger.error(error_msg)
            raise CalendlyAPIError(error_msg)

    def save_invitees_chunked(self, data: Dict[str, Any], base_filename: str, chunk_size: Optional[int] = None) -> List[str]:
        """
        Save invitee data in chunks to handle large datasets
        
        Args:
            data (Dict): Invitee data to save
            base_filename (str): Base filename for chunks
            chunk_size (int): Number of records per chunk
            
        Returns:
            List[str]: List of saved file paths
        """
        if chunk_size is None:
            chunk_size = self.CHUNK_SIZE
        
        invitees = data.get('invitees', [])
        total_invitees = len(invitees)
        
        logger.info(f"💾 Saving {total_invitees} invitees in chunks of {chunk_size}...")
        
        if total_invitees == 0:
            logger.warning("⚠️ No invitee data to save")
            return []
        
        saved_files = []
        
        # Create directory if it doesn't exist
        Path(base_filename).parent.mkdir(parents=True, exist_ok=True)
        
        # Save in chunks
        for i in range(0, total_invitees, chunk_size):
            chunk = invitees[i:i + chunk_size]
            chunk_num = i // chunk_size + 1
            total_chunks = (total_invitees + chunk_size - 1) // chunk_size
            
            chunk_filename = f"{base_filename}_chunk_{chunk_num:03d}_of_{total_chunks:03d}.json"
            
            chunk_data = {
                'invitees': chunk,
                'metadata': {
                    'chunk_number': chunk_num,
                    'total_chunks': total_chunks,
                    'chunk_size': len(chunk),
                    'total_invitees': total_invitees,
                    'extraction_timestamp': data.get('summary', {}).get('extraction_timestamp'),
                    'base_filename': base_filename
                }
            }
            
            try:
                with open(chunk_filename, 'w', encoding='utf-8') as f:
                    json.dump(chunk_data, f, indent=2, ensure_ascii=False, default=str)
                
                file_size = len(json.dumps(chunk_data, default=str).encode('utf-8')) / 1024  # KB
                logger.info(f"✅ Saved chunk {chunk_num}/{total_chunks}: {chunk_filename} ({file_size:.1f} KB, {len(chunk)} invitees)")
                saved_files.append(chunk_filename)
                
            except Exception as e:
                logger.error(f"❌ Failed to save chunk {chunk_num}: {str(e)}")
        
        # Save summary separately
        summary_filename = f"{base_filename}_summary.json"
        try:
            with open(summary_filename, 'w', encoding='utf-8') as f:
                json.dump({
                    'summary': data.get('summary', {}),
                    'api_metrics': data.get('api_metrics', {}),
                    'metadata': data.get('metadata', {})
                }, f, indent=2, ensure_ascii=False, default=str)
            
            saved_files.append(summary_filename)
            logger.info(f"✅ Saved summary: {summary_filename}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save summary: {str(e)}")
        
        logger.info(f"🎉 Saved {len(saved_files)} files total")
        return saved_files

    def to_dataframe(self, data: Dict[str, Any]) -> pd.DataFrame:
        """
        Convert invitee data to pandas DataFrame for analysis
        
        Args:
            data (Dict): Comprehensive invitee data
            
        Returns:
            pd.DataFrame: Invitee data as DataFrame
        """
        try:
            logger.info("🔄 Converting invitee data to DataFrame...")
            
            invitees_data = []
            
            for invitee_dict in data.get('invitees', []):
                # Flatten the invitee data for better analysis
                flat_invitee = invitee_dict.copy()
                
                # Extract questions and answers as separate columns
                qa_list = invitee_dict.get('questions_and_answers', [])
                for j, qa in enumerate(qa_list):
                    flat_invitee[f'question_{j+1}'] = qa.get('question', '')
                    flat_invitee[f'answer_{j+1}'] = qa.get('answer', '')
                
                # Extract tracking information
                tracking = invitee_dict.get('tracking', {})
                flat_invitee['utm_source'] = tracking.get('utm_source', '')
                flat_invitee['utm_medium'] = tracking.get('utm_medium', '')
                flat_invitee['utm_campaign'] = tracking.get('utm_campaign', '')
                flat_invitee['utm_term'] = tracking.get('utm_term', '')
                flat_invitee['utm_content'] = tracking.get('utm_content', '')
                flat_invitee['salesforce_uuid'] = tracking.get('salesforce_uuid', '')
                
                invitees_data.append(flat_invitee)
            
            df = pd.DataFrame(invitees_data)
            logger.info(f"✅ Created DataFrame with {len(df)} invitees and {len(df.columns)} columns")
            
            # Clean and optimize DataFrame
            if not df.empty:
                # Convert datetime columns
                datetime_columns = ['created_at', 'updated_at']
                for col in datetime_columns:
                    if col in df.columns:
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                
                # Optimize data types
                categorical_columns = ['status', 'timezone', 'event_name', 'event_type_name']
                for col in categorical_columns:
                    if col in df.columns:
                        df[col] = df[col].astype('category')
                
                logger.debug("Optimized DataFrame data types")
            
            return df
            
        except Exception as e:
            error_msg = f"🚨 DataFrame conversion failed: {str(e)}"
            logger.error(error_msg)
            raise CalendlyAPIError(error_msg)

    def export_dataframe_optimized(self, df: pd.DataFrame, base_filename: str) -> Dict[str, str]:
        """
        Export DataFrame in multiple optimized formats
        
        Args:
            df (pd.DataFrame): Invitee data DataFrame
            base_filename (str): Base filename for exports
            
        Returns:
            Dict: Paths to exported files
        """
        exported_files = {}
        
        try:
            # CSV Export
            csv_filename = f"{base_filename}.csv"
            df.to_csv(csv_filename, index=False, encoding='utf-8')
            csv_size = len(df.to_csv(index=False).encode('utf-8')) / (1024 * 1024)  # MB
            exported_files['csv'] = csv_filename
            logger.info(f"✅ CSV exported: {csv_filename} ({csv_size:.2f} MB)")
            
            # Parquet Export (better for large datasets)
            parquet_filename = f"{base_filename}.parquet"
            df.to_parquet(parquet_filename, index=False, compression='snappy')
            parquet_size = Path(parquet_filename).stat().st_size / (1024 * 1024)  # MB
            exported_files['parquet'] = parquet_filename
            logger.info(f"✅ Parquet exported: {parquet_filename} ({parquet_size:.2f} MB)")
            
            # Pickle Export (for Python analysis)
            pickle_filename = f"{base_filename}.pkl"
            with open(pickle_filename, 'wb') as f:
                pickle.dump(df, f)
            pickle_size = Path(pickle_filename).stat().st_size / (1024 * 1024)  # MB
            exported_files['pickle'] = pickle_filename
            logger.info(f"✅ Pickle exported: {pickle_filename} ({pickle_size:.2f} MB)")
            
            return exported_files
            
        except Exception as e:
            error_msg = f"🚨 DataFrame export failed: {str(e)}"
            logger.error(error_msg)
            raise CalendlyAPIError(error_msg)

    def get_detailed_analytics(self, data: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive analytics for invitee data
        
        Args:
            data (Dict): Comprehensive invitee data
            df (pd.DataFrame): Invitee data DataFrame
            
        Returns:
            Dict: Comprehensive analytics
        """
        try:
            logger.info("📈 Generating detailed invitee analytics...")
            
            if df.empty:
                return {}
            
            analytics = {
                'summary_stats': {
                    'total_invitees': len(df),
                    'unique_events': df['event_uri'].nunique(),
                    'unique_invitees': df['email'].nunique(),
                    'date_range': {
                        'earliest_booking': df['created_at'].min().isoformat() if pd.notna(df['created_at'].min()) else None,
                        'latest_booking': df['created_at'].max().isoformat() if pd.notna(df['created_at'].max()) else None
                    }
                },
                'status_distribution': df['status'].value_counts().to_dict(),
                'event_popularity': df['event_name'].value_counts().head(20).to_dict(),
                'timezone_distribution': df['timezone'].value_counts().head(10).to_dict(),
                'booking_trends': {
                    'by_month': df['created_at'].dt.to_period('M').value_counts().sort_index().to_dict(),
                    'by_weekday': df['created_at'].dt.day_name().value_counts().to_dict()
                },
                'question_analysis': self._analyze_questions(df),
                'utm_analysis': {
                    'sources': df['utm_source'].value_counts().to_dict(),
                    'campaigns': df['utm_campaign'].value_counts().to_dict(),
                    'mediums': df['utm_medium'].value_counts().to_dict()
                }
            }
            
            logger.info("✅ Detailed analytics generated successfully")
            return analytics
            
        except Exception as e:
            logger.warning(f"⚠️ Analytics generation failed: {str(e)}")
            return {}

    def _analyze_questions(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze questions and answers from invitees
        
        Args:
            df (pd.DataFrame): Invitee data DataFrame
            
        Returns:
            Dict: Question analysis
        """
        try:
            # Extract all questions
            all_questions = {}
            question_columns = [col for col in df.columns if col.startswith('question_')]
            
            for col in question_columns:
                answer_col = col.replace('question_', 'answer_')
                if answer_col in df.columns:
                    # Get unique questions and their answer distributions
                    unique_questions = df[col].value_counts().to_dict()
                    for question, count in unique_questions.items():
                        if question and question not in all_questions:
                            all_questions[question] = {
                                'count': count,
                                'answer_distribution': df[df[col] == question][answer_col].value_counts().to_dict()
                            }
            
            return {
                'total_unique_questions': len(all_questions),
                'questions': all_questions
            }
            
        except Exception as e:
            logger.warning(f"⚠️ Question analysis failed: {str(e)}")
            return {}

    def print_extraction_report(self, data: Dict[str, Any], analytics: Dict[str, Any]):
        """
        Print comprehensive extraction report
        
        Args:
            data (Dict): Comprehensive invitee data
            analytics (Dict): Analytics data
        """
        summary = data.get('summary', {})
        
        logger.info("📊 COMPREHENSIVE INVITEE EXTRACTION REPORT")
        logger.info("=" * 80)
        logger.info("🎯 EXTRACTION OVERVIEW:")
        logger.info(f"   • Total Invitees: {summary.get('total_invitees', 0)}")
        logger.info(f"   • Events Processed: {summary.get('total_events_processed', 0)}")
        logger.info(f"   • Scheduled Events: {summary.get('total_scheduled_events', 0)}")
        logger.info(f"   • Extraction Time: {summary.get('duration_seconds', 0):.2f}s")
        logger.info(f"   • Data Size: {summary.get('data_size_mb', 0):.2f} MB")
        
        if analytics:
            stats = analytics.get('summary_stats', {})
            logger.info("📈 DATA INSIGHTS:")
            logger.info(f"   • Unique Events: {stats.get('unique_events', 0)}")
            logger.info(f"   • Unique Invitees: {stats.get('unique_invitees', 0)}")
            logger.info(f"   • Date Range: {stats.get('date_range', {}).get('earliest_booking', 'N/A')} to {stats.get('date_range', {}).get('latest_booking', 'N/A')}")
            
            status_dist = analytics.get('status_distribution', {})
            logger.info("📋 STATUS DISTRIBUTION:")
            for status, count in status_dist.items():
                logger.info(f"   • {status}: {count}")
        
        logger.info("=" * 80)

    def close(self):
        """Clean up resources"""
        logger.info("🧹 Cleaning up CCDWInvitees resources...")
        if hasattr(self, '_session'):
            self._session.close()
            logger.info("✅ HTTP session closed")
        logger.info("🎉 CCDWInvitees shutdown complete")


# Factory function
def create_calendly_invitees_extractor(access_token: str, organization_uri: Optional[str] = None) -> CCDWInvitees:
    """
    Factory function to create CCDWInvitees instance
    
    Args:
        access_token (str): Calendly access token
        organization_uri (str, optional): Organization URI
        
    Returns:
        CCDWInvitees: Initialized invitees extractor
    """
    return CCDWInvitees(access_token, organization_uri)
