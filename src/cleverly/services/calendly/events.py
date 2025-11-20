"""
CCDWEvents - Cleverly Calendly Data Warehouse Events Module
Event types data extraction for Calendly API
Author: Synexian Team
Version: 1.0.0 
"""

import requests
import json
import pandas as pd
import logging
from typing import List, Dict, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import time
from abc import ABC, abstractmethod
import sys
import inspect

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler('ccdw_events_debug.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class EventType:
    """Data class for Calendly event type information"""
    uri: str
    name: str
    active: bool
    type: str
    kind: str
    slug: str
    duration: int
    description_plain: Optional[str]
    scheduling_url: str
    color: Optional[str]
    internal_note: Optional[str]
    created_at: str
    updated_at: str
    owner_uri: str
    owner_type: str  # 'organization' or 'user'
    owner_name: Optional[str] = None
    owner_email: Optional[str] = None

@dataclass
class EventExtractionSummary:
    """Data class for extraction summary statistics"""
    total_organization_events: int
    total_user_events: int
    total_members: int
    active_events: int
    inactive_events: int
    extraction_timestamp: str
    duration_seconds: float
    success_rate: float

@dataclass
class APIRequestMetrics:
    """Data class for API request metrics"""
    total_requests: int
    successful_requests: int
    failed_requests: int
    rate_limited_requests: int
    average_response_time: float

class CalendlyAPIError(Exception):
    """Custom exception for Calendly API errors"""
    pass

class EventExtractionStrategy(ABC):
    """Abstract base class for event extraction strategies"""
    
    @abstractmethod
    def extract(self, extractor: 'CCDWEvents') -> List[EventType]:
        pass

    @abstractmethod
    def get_strategy_name(self) -> str:
        pass

class OrganizationEventExtraction(EventExtractionStrategy):
    """Strategy for organization-wide event types extraction"""
    
    def extract(self, extractor: 'CCDWEvents') -> List[EventType]:
        return extractor._get_organization_event_types()
    
    def get_strategy_name(self) -> str:
        return "OrganizationEventExtraction"

class UserEventExtraction(EventExtractionStrategy):
    """Strategy for user-specific event types extraction"""
    
    def extract(self, extractor: 'CCDWEvents') -> List[EventType]:
        return extractor._get_all_user_event_types()
    
    def get_strategy_name(self) -> str:
        return "UserEventExtraction"

class CCDWEvents:
    """
    Cleverly Calendly Data Warehouse Events - Event types data extraction
    
    Features:
    - Comprehensive event types extraction from organization and users
    - Advanced error handling and retry mechanisms
    - Detailed logging for production debugging
    - Multiple output formats (JSON, DataFrame, CSV)
    - Configurable extraction strategies
    - Performance monitoring and metrics
    - Real-time analytics and progress tracking
    """
    
    BASE_URL = "https://api.calendly.com"
    DEFAULT_TIMEOUT = 30
    MAX_RETRIES = 3
    RETRY_BACKOFF_FACTOR = 1.5
    
    def __init__(self, access_token: str, organization_uri: Optional[str] = None):
        """
        Initialize CCDWEvents with Calendly access token
        
        Args:
            access_token (str): Calendly personal access token
            organization_uri (str, optional): Organization URI. If None, will be auto-detected
            
        Raises:
            ValueError: If access token is invalid
            CalendlyAPIError: If initial connection test fails
        """
        self._setup_logging()
        logger.info("🚀 Initializing CCDWEvents...")
        
        if not access_token or access_token == "your_personal_access_token_here":
            error_msg = "Valid Calendly access token is required"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        self.access_token = access_token
        self.organization_uri = organization_uri
        self.headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
            "User-Agent": "CCDWEvents/1.1.0"
        }
        
        self._session = requests.Session()
        self._session.headers.update(self.headers)
        
        # Initialize metrics and state
        self._user_info: Optional[Dict] = None
        self._organization_members: List[Dict] = []
        self._extraction_metrics: Dict[str, Any] = {
            'start_time': None,
            'end_time': None,
            'total_events': 0,
            'success': False
        }
        self._api_metrics = APIRequestMetrics(0, 0, 0, 0, 0.0)
        
        self._validate_connection()
        logger.info("✅ CCDWEvents initialized successfully")

    def _setup_logging(self):
        """Setup comprehensive logging configuration"""
        logger.setLevel(logging.DEBUG)
        
        # Prevent duplicate handlers
        if not logger.handlers:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(funcName)s - %(message)s'
            )
            
            # File handler for detailed debugging
            file_handler = logging.FileHandler('ccdw_events_detailed.log', encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            
            # Console handler for important messages
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(formatter)
            
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)

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
            self._user_info = user_data
            
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
        Build proper URL for API requests (FIXED: Prevent double base URL)
        
        Args:
            endpoint (str): API endpoint or full URL
            
        Returns:
            str: Properly formatted URL
        """
        # If endpoint is already a full URL, use it as is
        if endpoint.startswith('http'):
            logger.debug(f"Using full URL: {endpoint}")
            return endpoint
        
        # If endpoint starts with base URL, use it as is (prevent duplication)
        if endpoint.startswith(self.BASE_URL):
            logger.debug(f"Endpoint already contains base URL: {endpoint}")
            return endpoint
        
        # Build proper URL
        if endpoint.startswith('/'):
            url = f"{self.BASE_URL}{endpoint}"
        else:
            url = f"{self.BASE_URL}/{endpoint}"
        
        logger.debug(f"Built URL: {url}")
        return url

    def _make_api_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Make authenticated API request with advanced retry logic and comprehensive logging
        
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
                self._api_metrics.total_requests += 1
                
                logger.debug(f"Attempt {attempt + 1}/{self.MAX_RETRIES} for {url}")
                
                response = self._session.get(
                    url,
                    params=params,
                    timeout=self.DEFAULT_TIMEOUT
                )
                
                response_time = time.time() - start_time
                
                if response.status_code == 429:  # Rate limiting
                    self._api_metrics.rate_limited_requests += 1
                    retry_after = int(response.headers.get('Retry-After', 
                                        self.RETRY_BACKOFF_FACTOR * (attempt + 1)))
                    logger.warning(f"⏳ Rate limited. Retrying after {retry_after} seconds")
                    time.sleep(retry_after)
                    continue
                
                response.raise_for_status()
                
                self._api_metrics.successful_requests += 1
                self._api_metrics.average_response_time = (
                    (self._api_metrics.average_response_time * (self._api_metrics.successful_requests - 1) + response_time) 
                    / self._api_metrics.successful_requests
                )
                
                logger.debug(f"✅ API request successful: {response.status_code} in {response_time:.2f}s")
                logger.debug(f"Response size: {len(response.content)} bytes")
                
                return response.json()
                
            except requests.exceptions.RequestException as e:
                self._api_metrics.failed_requests += 1
                response_time = time.time() - start_time
                
                logger.error(f"❌ API request failed (attempt {attempt + 1}): {str(e)}")
                logger.error(f"Failed URL: {url}")
                logger.error(f"Request duration: {response_time:.2f}s")
                
                if hasattr(e, 'response') and e.response is not None:
                    logger.error(f"Status code: {e.response.status_code}")
                    logger.error(f"Response headers: {dict(e.response.headers)}")
                    if e.response.status_code < 500:  # Client error
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
        Handle paginated API responses and return all data with progress tracking
        
        Args:
            endpoint (str): API endpoint
            params (Dict): Query parameters
            data_type (str): Description of data being fetched for logging
            
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
                    logger.debug(f"Next page available: {next_page}")
                    url = next_page
                    current_params = {}  # Next page URL includes params
                else:
                    logger.info(f"📚 No more pages. Completed {data_type} extraction.")
                    url = None
                
            except CalendlyAPIError as e:
                logger.error(f"🚨 Pagination error at page {page_count} for {data_type}: {str(e)}")
                break
        
        logger.info(f"🎉 Completed {data_type} extraction: {total_items} items across {page_count} pages")
        return all_data

    def _get_organization_event_types(self) -> List[EventType]:
        """
        Extract organization-wide event types
        
        Returns:
            List[EventType]: Organization event types
            
        Raises:
            CalendlyAPIError: If extraction fails
        """
        logger.info("🏢 Starting organization event types extraction...")
        
        try:
            params = {
                "organization": self.organization_uri,
                "count": 100
            }
            
            raw_events = self._get_paginated_data("/event_types", params, "organization event types")
            event_types = []
            
            logger.info(f"🔄 Processing {len(raw_events)} organization event types...")
            
            for i, event_data in enumerate(raw_events, 1):
                if i % 10 == 0:
                    logger.debug(f"Processed {i}/{len(raw_events)} organization events")
                
                event_type = EventType(
                    uri=event_data.get('uri', ''),
                    name=event_data.get('name', ''),
                    active=event_data.get('active', False),
                    type=event_data.get('type', ''),
                    kind=event_data.get('kind', ''),
                    slug=event_data.get('slug', ''),
                    duration=event_data.get('duration', 0),
                    description_plain=event_data.get('description_plain'),
                    scheduling_url=event_data.get('scheduling_url', ''),
                    color=event_data.get('color'),
                    internal_note=event_data.get('internal_note'),
                    created_at=event_data.get('created_at', ''),
                    updated_at=event_data.get('updated_at', ''),
                    owner_uri=self.organization_uri or '',
                    owner_type='organization',
                    owner_name=None,
                    owner_email=None
                )
                event_types.append(event_type)
            
            logger.info(f"✅ Successfully extracted {len(event_types)} organization event types")
            return event_types
            
        except Exception as e:
            error_msg = f"🚨 Organization event types extraction failed: {str(e)}"
            logger.error(error_msg)
            raise CalendlyAPIError(error_msg)

    def _get_user_event_types(self, user_uri: str, user_name: str, user_email: str) -> List[EventType]:
        """
        Extract event types for a specific user
        
        Args:
            user_uri (str): User URI
            user_name (str): User name
            user_email (str): User email
            
        Returns:
            List[EventType]: User event types
        """
        try:
            logger.debug(f"👤 Extracting event types for user: {user_name}")
            
            params = {
                "user": user_uri,
                "count": 100
            }
            
            raw_events = self._get_paginated_data("/event_types", params, f"event types for {user_name}")
            event_types = []
            
            for event_data in raw_events:
                event_type = EventType(
                    uri=event_data.get('uri', ''),
                    name=event_data.get('name', ''),
                    active=event_data.get('active', False),
                    type=event_data.get('type', ''),
                    kind=event_data.get('kind', ''),
                    slug=event_data.get('slug', ''),
                    duration=event_data.get('duration', 0),
                    description_plain=event_data.get('description_plain'),
                    scheduling_url=event_data.get('scheduling_url', ''),
                    color=event_data.get('color'),
                    internal_note=event_data.get('internal_note'),
                    created_at=event_data.get('created_at', ''),
                    updated_at=event_data.get('updated_at', ''),
                    owner_uri=user_uri,
                    owner_type='user',
                    owner_name=user_name,
                    owner_email=user_email
                )
                event_types.append(event_type)
            
            logger.debug(f"✅ Extracted {len(event_types)} event types for user: {user_name}")
            return event_types
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to extract event types for user {user_name}: {str(e)}")
            return []

    def _get_organization_members(self) -> List[Dict]:
        """
        Get all organization members
        
        Returns:
            List[Dict]: Organization members data
        """
        logger.info("👥 Fetching organization members...")
        
        try:
            params = {
                "organization": self.organization_uri,
                "count": 100
            }
            
            memberships = self._get_paginated_data("/organization_memberships", params, "organization members")
            
            # Extract user information from memberships
            members = []
            for membership in memberships:
                user_data = membership.get('user', {})
                member_info = {
                    'name': user_data.get('name', ''),
                    'email': user_data.get('email', ''),
                    'uri': user_data.get('uri', ''),
                    'role': membership.get('role', ''),
                    'status': membership.get('status', ''),
                    'membership_uri': membership.get('uri', '')
                }
                members.append(member_info)
            
            logger.info(f"✅ Retrieved {len(members)} organization members")
            return members
            
        except Exception as e:
            error_msg = f"🚨 Organization members extraction failed: {str(e)}"
            logger.error(error_msg)
            raise CalendlyAPIError(error_msg)

    def _get_all_user_event_types(self) -> List[EventType]:
        """
        Extract event types for all organization users
        
        Returns:
            List[EventType]: All user event types
        """
        logger.info("🔍 Starting user event types extraction for all members...")
        
        try:
            members = self._get_organization_members()
            all_user_events = []
            successful_extractions = 0
            
            logger.info(f"🔄 Processing {len(members)} members...")
            
            for i, member in enumerate(members, 1):
                logger.info(f"👤 Processing member {i}/{len(members)}: {member['name']}")
                
                user_events = self._get_user_event_types(
                    user_uri=member['uri'],
                    user_name=member['name'],
                    user_email=member['email']
                )
                
                if user_events:
                    successful_extractions += 1
                    all_user_events.extend(user_events)
                    logger.info(f"✅ {member['name']}: {len(user_events)} event types")
                else:
                    logger.warning(f"⚠️ {member['name']}: No event types found or failed to extract")
                
                # Brief pause to avoid rate limiting
                if i < len(members):
                    time.sleep(0.2)
            
            success_rate = (successful_extractions / len(members)) * 100 if members else 0
            logger.info(f"🎉 User event types extraction completed: {len(all_user_events)} events from {successful_extractions}/{len(members)} members ({success_rate:.1f}% success rate)")
            
            return all_user_events
            
        except Exception as e:
            error_msg = f"🚨 User event types extraction failed: {str(e)}"
            logger.error(error_msg)
            raise CalendlyAPIError(error_msg)

    def extract_events(self, strategy: EventExtractionStrategy) -> List[EventType]:
        """
        Extract events using specified strategy
        
        Args:
            strategy (EventExtractionStrategy): Extraction strategy
            
        Returns:
            List[EventType]: Extracted event types
        """
        strategy_name = strategy.get_strategy_name()
        logger.info(f"🎯 Executing event extraction strategy: {strategy_name}")
        
        start_time = time.time()
        try:
            result = strategy.extract(self)
            duration = time.time() - start_time
            
            logger.info(f"✅ Strategy {strategy_name} completed in {duration:.2f}s: {len(result)} events")
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"🚨 Strategy {strategy_name} failed after {duration:.2f}s: {str(e)}")
            raise

    def extract_comprehensive_events(self) -> Dict[str, Any]:
        """
        Perform comprehensive event types extraction including organization and user events
        
        Returns:
            Dict: Comprehensive event data with summary
            
        Raises:
            CalendlyAPIError: If extraction fails
        """
        start_time = time.time()
        self._extraction_metrics['start_time'] = datetime.now().isoformat()
        
        logger.info("🚀 Starting comprehensive event types extraction...")
        
        try:
            # Extract organization events
            org_strategy = OrganizationEventExtraction()
            organization_events = self.extract_events(org_strategy)
            
            # Extract user events
            user_strategy = UserEventExtraction()
            user_events = self.extract_events(user_strategy)
            
            # Get organization members
            members = self._get_organization_members()
            
            # Calculate summary statistics
            all_events = organization_events + user_events
            active_events = sum(1 for event in all_events if event.active)
            inactive_events = len(all_events) - active_events
            success_rate = (self._api_metrics.successful_requests / self._api_metrics.total_requests * 100) if self._api_metrics.total_requests > 0 else 0
            
            extraction_time = time.time() - start_time
            
            summary = EventExtractionSummary(
                total_organization_events=len(organization_events),
                total_user_events=len(user_events),
                total_members=len(members),
                active_events=active_events,
                inactive_events=inactive_events,
                extraction_timestamp=datetime.now().isoformat(),
                duration_seconds=round(extraction_time, 2),
                success_rate=round(success_rate, 2)
            )
            
            comprehensive_data = {
                'user_info': self._user_info,
                'organization_uri': self.organization_uri,
                'organization_events': [asdict(event) for event in organization_events],
                'user_events': [asdict(event) for event in user_events],
                'members': members,
                'summary': asdict(summary),
                'api_metrics': asdict(self._api_metrics),
                'metadata': {
                    'extraction_version': '1.1.0',
                    'source': 'CCDWEvents',
                    'total_events': len(all_events),
                    'extraction_duration_seconds': round(extraction_time, 2)
                }
            }
            
            # Store metrics
            self._extraction_metrics.update({
                'total_events': len(all_events),
                'extraction_time_seconds': extraction_time,
                'success': True,
                'end_time': datetime.now().isoformat()
            })
            
            # Log comprehensive results
            logger.info("🎉 COMPREHENSIVE EXTRACTION COMPLETED SUCCESSFULLY!")
            logger.info("=" * 60)
            logger.info(f"📊 EXTRACTION SUMMARY:")
            logger.info(f"   • Organization Events: {len(organization_events)}")
            logger.info(f"   • User Events: {len(user_events)}")
            logger.info(f"   • Total Members: {len(members)}")
            logger.info(f"   • Active Events: {active_events}")
            logger.info(f"   • Inactive Events: {inactive_events}")
            logger.info(f"   • Total Events: {len(all_events)}")
            logger.info(f"   • Extraction Time: {extraction_time:.2f}s")
            logger.info(f"   • API Success Rate: {success_rate:.1f}%")
            logger.info(f"   • Total API Requests: {self._api_metrics.total_requests}")
            logger.info(f"   • Average Response Time: {self._api_metrics.average_response_time:.2f}s")
            logger.info("=" * 60)
            
            return comprehensive_data
            
        except Exception as e:
            extraction_time = time.time() - start_time
            self._extraction_metrics.update({
                'success': False,
                'end_time': datetime.now().isoformat(),
                'error': str(e)
            })
            
            error_msg = f"🚨 Comprehensive extraction failed after {extraction_time:.2f}s: {str(e)}"
            logger.error(error_msg)
            raise CalendlyAPIError(error_msg)

    def export_to_json(self, data: Dict[str, Any], filename: Optional[str] = None) -> str:
        """
        Export event data to JSON file
        
        Args:
            data (Dict): Event data to export
            filename (str): Output filename
            
        Returns:
            str: Path to exported file
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"calendly_events_comprehensive_{timestamp}.json"
        
        try:
            logger.info(f"💾 Exporting data to JSON file: {filename}")
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            
            file_size = len(json.dumps(data, default=str).encode('utf-8')) / 1024  # KB
            logger.info(f"✅ Event data exported to: {filename} ({file_size:.1f} KB)")
            return filename
            
        except Exception as e:
            error_msg = f"🚨 JSON export failed: {str(e)}"
            logger.error(error_msg)
            raise CalendlyAPIError(error_msg)

    def to_dataframe(self, data: Dict[str, Any]) -> pd.DataFrame:
        """
        Convert event data to pandas DataFrame for analysis
        
        Args:
            data (Dict): Comprehensive event data
            
        Returns:
            pd.DataFrame: Event data as DataFrame
        """
        try:
            logger.info("🔄 Converting event data to DataFrame...")
            
            events_data = []
            
            # Process organization events
            org_events_count = len(data.get('organization_events', []))
            logger.debug(f"Processing {org_events_count} organization events")
            
            for event_dict in data.get('organization_events', []):
                event_data = event_dict.copy()
                event_data['source'] = 'organization'
                events_data.append(event_data)
            
            # Process user events
            user_events_count = len(data.get('user_events', []))
            logger.debug(f"Processing {user_events_count} user events")
            
            for event_dict in data.get('user_events', []):
                event_data = event_dict.copy()
                event_data['source'] = 'user'
                events_data.append(event_data)
            
            df = pd.DataFrame(events_data)
            logger.info(f"✅ Created DataFrame with {len(df)} events")
            
            # Clean and optimize DataFrame
            if not df.empty:
                # Convert datetime columns
                datetime_columns = ['created_at', 'updated_at']
                for col in datetime_columns:
                    if col in df.columns:
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                        logger.debug(f"Converted {col} to datetime")
                
                # Optimize data types
                df['active'] = df['active'].astype(bool)
                df['duration'] = pd.to_numeric(df['duration'], errors='coerce').fillna(0).astype(int)
                
                logger.debug("Optimized DataFrame data types")
            
            return df
            
        except Exception as e:
            error_msg = f"🚨 DataFrame conversion failed: {str(e)}"
            logger.error(error_msg)
            raise CalendlyAPIError(error_msg)

    def export_dataframe_to_csv(self, df: pd.DataFrame, filename: Optional[str] = None) -> str:
        """
        Export DataFrame to CSV file
        
        Args:
            df (pd.DataFrame): Event data DataFrame
            filename (str): Output filename
            
        Returns:
            str: Path to exported file
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"calendly_events_analysis_{timestamp}.csv"
        
        try:
            logger.info(f"💾 Exporting DataFrame to CSV: {filename}")
            
            df.to_csv(filename, index=False, encoding='utf-8')
            
            file_size = len(df.to_csv(index=False).encode('utf-8')) / 1024  # KB
            logger.info(f"✅ DataFrame exported to CSV: {filename} ({file_size:.1f} KB, {len(df)} rows)")
            return filename
            
        except Exception as e:
            error_msg = f"🚨 CSV export failed: {str(e)}"
            logger.error(error_msg)
            raise CalendlyAPIError(error_msg)

    def get_extraction_metrics(self) -> Dict[str, Any]:
        """
        Get extraction performance metrics
        
        Returns:
            Dict: Extraction metrics
        """
        metrics = self._extraction_metrics.copy()
        metrics['api_metrics'] = asdict(self._api_metrics)
        return metrics

    def get_event_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive event statistics
        
        Args:
            df (pd.DataFrame): Event data DataFrame
            
        Returns:
            Dict: Event statistics
        """
        try:
            logger.info("📈 Generating event statistics...")
            
            if df.empty:
                logger.warning("No data available for statistics generation")
                return {}
            
            stats = {
                'total_events': len(df),
                'active_events': df['active'].sum(),
                'inactive_events': len(df) - df['active'].sum(),
                'organization_events': len(df[df['source'] == 'organization']),
                'user_events': len(df[df['source'] == 'user']),
                'unique_users': df['owner_name'].nunique(),
                'duration_stats': {
                    'min': df['duration'].min(),
                    'max': df['duration'].max(),
                    'mean': df['duration'].mean(),
                    'median': df['duration'].median(),
                    'std': df['duration'].std()
                },
                'kind_distribution': df['kind'].value_counts().to_dict(),
                'type_distribution': df['type'].value_counts().to_dict(),
                'source_distribution': df['source'].value_counts().to_dict(),
                'top_users': df['owner_name'].value_counts().head(10).to_dict(),
                'activity_timeline': {
                    'earliest_created': df['created_at'].min().isoformat() if pd.notna(df['created_at'].min()) else None,
                    'latest_created': df['created_at'].max().isoformat() if pd.notna(df['created_at'].max()) else None,
                    'earliest_updated': df['updated_at'].min().isoformat() if pd.notna(df['updated_at'].min()) else None,
                    'latest_updated': df['updated_at'].max().isoformat() if pd.notna(df['updated_at'].max()) else None
                }
            }
            
            logger.info("✅ Event statistics generated successfully")
            return stats
            
        except Exception as e:
            logger.warning(f"⚠️ Statistics generation failed: {str(e)}")
            return {}

    def print_detailed_analytics(self, data: Dict[str, Any], df: pd.DataFrame):
        """
        Print detailed analytics report to console
        
        Args:
            data (Dict): Comprehensive event data
            df (pd.DataFrame): Event data DataFrame
        """
        logger.info("📊 GENERATING DETAILED ANALYTICS REPORT")
        logger.info("=" * 80)
        
        # Basic metrics
        summary = data.get('summary', {})
        api_metrics = data.get('api_metrics', {})
        
        logger.info("🎯 EXTRACTION OVERVIEW:")
        logger.info(f"   • Total Events: {summary.get('total_organization_events', 0) + summary.get('total_user_events', 0)}")
        logger.info(f"   • Organization Events: {summary.get('total_organization_events', 0)}")
        logger.info(f"   • User Events: {summary.get('total_user_events', 0)}")
        logger.info(f"   • Organization Members: {summary.get('total_members', 0)}")
        logger.info(f"   • Active Events: {summary.get('active_events', 0)}")
        logger.info(f"   • Inactive Events: {summary.get('inactive_events', 0)}")
        logger.info(f"   • Extraction Duration: {summary.get('duration_seconds', 0):.2f}s")
        logger.info(f"   • API Success Rate: {summary.get('success_rate', 0):.1f}%")
        
        logger.info("🔧 API PERFORMANCE:")
        logger.info(f"   • Total Requests: {api_metrics.get('total_requests', 0)}")
        logger.info(f"   • Successful: {api_metrics.get('successful_requests', 0)}")
        logger.info(f"   • Failed: {api_metrics.get('failed_requests', 0)}")
        logger.info(f"   • Rate Limited: {api_metrics.get('rate_limited_requests', 0)}")
        logger.info(f"   • Avg Response Time: {api_metrics.get('average_response_time', 0):.2f}s")
        
        if not df.empty:
            stats = self.get_event_statistics(df)
            logger.info("📈 EVENT STATISTICS:")
            logger.info(f"   • Unique Users with Events: {stats.get('unique_users', 0)}")
            logger.info(f"   • Event Duration Range: {stats['duration_stats']['min']} - {stats['duration_stats']['max']} min")
            logger.info(f"   • Average Duration: {stats['duration_stats']['mean']:.1f} min")
            
            logger.info("🏷️  EVENT TYPE DISTRIBUTION:")
            for kind, count in stats.get('kind_distribution', {}).items():
                logger.info(f"   • {kind}: {count}")
        
        logger.info("=" * 80)

    def close(self):
        """Clean up resources"""
        logger.info("🧹 Cleaning up CCDWEvents resources...")
        if hasattr(self, '_session'):
            self._session.close()
            logger.info("✅ HTTP session closed")
        logger.info("🎉 CCDWEvents shutdown complete")


# Factory function for easy initialization
def create_calendly_events_extractor(access_token: str, organization_uri: Optional[str] = None) -> CCDWEvents:
    """
    Factory function to create CCDWEvents instance
    
    Args:
        access_token (str): Calendly access token
        organization_uri (str, optional): Organization URI
        
    Returns:
        CCDWEvents: Initialized events extractor
    """
    return CCDWEvents(access_token, organization_uri)


# Notebook usage example
if __name__ == "__main__":
    import os
    
    # Example usage in notebook
    access_token = os.getenv('CALENDLY_ACCESS_TOKEN', 'your_personal_access_token_here')
    
    if access_token == 'your_personal_access_token_here':
        print("Please set CALENDLY_ACCESS_TOKEN environment variable")
    else:
        extractor = create_calendly_events_extractor(access_token)
        
        try:
            # Comprehensive extraction
            event_data = extractor.extract_comprehensive_events()
            
            # Export to JSON
            json_file = extractor.export_to_json(event_data)
            
            # Convert to DataFrame
            df = extractor.to_dataframe(event_data)
            
            # Export to CSV
            csv_file = extractor.export_dataframe_to_csv(df)
            
            # Get statistics and analytics
            stats = extractor.get_event_statistics(df)
            extractor.print_detailed_analytics(event_data, df)
            
            # Display some basic info
            print(f"\n🎉 Extraction completed successfully!")
            print(f"📁 JSON file: {json_file}")
            print(f"📁 CSV file: {csv_file}")
            print(f"📊 Total events: {len(df)}")
            
        except CalendlyAPIError as e:
            print(f"❌ Extraction failed: {e}")
        finally:
            extractor.close()
