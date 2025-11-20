"""
Cleverly Calendly Data Warehouse (CCDataW) - Data extraction module
Author: Synexian Team
Version: 1.0.0
"""

import requests
import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('calendly_data_extraction.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

@dataclass
class UserInfo:
    """Data class for user information"""
    name: str
    email: str
    uri: str
    timezone: str
    scheduling_url: str
    created_at: str
    updated_at: str
    current_organization: str
    resource_type: str

@dataclass
class OrganizationMember:
    """Data class for organization member information"""
    name: str
    email: str
    user_uri: str
    role: str
    status: str
    organization_uri: str
    membership_uri: str

class CalendlyAPIException(Exception):
    """Custom exception for Calendly API errors"""
    pass

class CCDataW:
    """
    Calendly Data Warehouse - Enterprise-grade data extraction class
    Provides comprehensive methods to extract USER, ORGANIZATION, and MEMBER data
    """
    
    BASE_URL = "https://api.calendly.com"
    MAX_RETRIES = 3
    RETRY_DELAY = 1  # seconds
    
    def __init__(self, access_token: str):
        """
        Initialize CCDataW with Calendly access token
        
        Args:
            access_token (str): Calendly personal access token
        """
        if not access_token or access_token == "your_personal_access_token_here":
            raise ValueError("Valid Calendly access token is required")
        
        self.access_token = access_token
        self.headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
            "User-Agent": "CCDataW-Client/1.0.0"
        }
        
        logger.info("CCDataW initialized successfully")

    def _make_api_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Make authenticated API request with retry logic and error handling
        
        Args:
            endpoint (str): API endpoint
            params (Dict): Query parameters
            
        Returns:
            Dict: API response data
            
        Raises:
            CalendlyAPIException: If API request fails
        """
        url = f"{self.BASE_URL}{endpoint}" if endpoint.startswith('/') else f"{self.BASE_URL}/{endpoint}"
        
        for attempt in range(self.MAX_RETRIES):
            try:
                logger.debug(f"Making API request to {url} (attempt {attempt + 1})")
                
                response = requests.get(
                    url, 
                    headers=self.headers, 
                    params=params,
                    timeout=30
                )
                
                if response.status_code == 429:  # Rate limiting
                    retry_after = int(response.headers.get('Retry-After', self.RETRY_DELAY * (attempt + 1)))
                    logger.warning(f"Rate limited. Retrying after {retry_after} seconds")
                    time.sleep(retry_after)
                    continue
                
                response.raise_for_status()
                
                logger.debug(f"API request successful: {response.status_code}")
                return response.json()
                
            except requests.exceptions.RequestException as e:
                logger.error(f"API request failed (attempt {attempt + 1}): {str(e)}")
                
                if attempt == self.MAX_RETRIES - 1:
                    error_msg = f"API request failed after {self.MAX_RETRIES} attempts: {str(e)}"
                    if hasattr(e, 'response') and e.response is not None:
                        error_msg += f" | Status: {e.response.status_code} | Response: {e.response.text}"
                    raise CalendlyAPIException(error_msg)
                
                time.sleep(self.RETRY_DELAY * (attempt + 1))
        
        raise CalendlyAPIException("Unexpected error in API request")

    def _get_paginated_data(self, endpoint: str, params: Optional[Dict] = None) -> List[Dict]:
        """
        Handle paginated API responses and return all data
        
        Args:
            endpoint (str): API endpoint
            params (Dict): Query parameters
            
        Returns:
            List[Dict]: All paginated data
        """
        all_data = []
        url = endpoint
        current_params = params.copy() if params else {}
        
        while url:
            try:
                data = self._make_api_request(url, current_params)
                all_data.extend(data.get('collection', []))
                
                # Check for next page
                pagination = data.get('pagination', {})
                url = pagination.get('next_page')
                current_params = {}  # Next page URL includes params
                
                logger.debug(f"Retrieved {len(data.get('collection', []))} items. Total: {len(all_data)}")
                
            except CalendlyAPIException as e:
                logger.error(f"Error in pagination: {str(e)}")
                break
        
        return all_data

    def get_user_information(self) -> UserInfo:
        """
        Extract comprehensive user information from Calendly
        
        Returns:
            UserInfo: Structured user information
            
        Raises:
            CalendlyAPIException: If user data extraction fails
        """
        logger.info("Starting user information extraction")
        
        try:
            data = self._make_api_request("/users/me")
            resource = data['resource']
            
            user_info = UserInfo(
                name=resource.get('name', ''),
                email=resource.get('email', ''),
                uri=resource.get('uri', ''),
                timezone=resource.get('timezone', ''),
                scheduling_url=resource.get('scheduling_url', ''),
                created_at=resource.get('created_at', ''),
                updated_at=resource.get('updated_at', ''),
                current_organization=resource.get('current_organization', ''),
                resource_type=resource.get('resource_type', '')
            )
            
            logger.info(f"Successfully extracted user information for: {user_info.name}")
            return user_info
            
        except KeyError as e:
            error_msg = f"Missing expected field in user data: {str(e)}"
            logger.error(error_msg)
            raise CalendlyAPIException(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error extracting user information: {str(e)}"
            logger.error(error_msg)
            raise CalendlyAPIException(error_msg)

    def get_organization_information(self, user_info: Optional[UserInfo] = None) -> Dict[str, Any]:
        """
        Extract organization information
        
        Args:
            user_info (UserInfo): User information object
            
        Returns:
            Dict: Organization information
            
        Raises:
            CalendlyAPIException: If organization data extraction fails
        """
        logger.info("Starting organization information extraction")
        
        try:
            if not user_info:
                user_info = self.get_user_information()
            
            org_uri = user_info.current_organization
            if not org_uri:
                raise CalendlyAPIException("No organization URI found in user data")
            
            # Extract organization details from URI
            org_id = org_uri.split('/')[-1]
            
            organization_info = {
                'uri': org_uri,
                'id': org_id,
                'user_uri': user_info.uri,
                'extracted_at': datetime.now().isoformat()
            }
            
            logger.info(f"Successfully extracted organization information: {org_id}")
            return organization_info
            
        except Exception as e:
            error_msg = f"Error extracting organization information: {str(e)}"
            logger.error(error_msg)
            raise CalendlyAPIException(error_msg)

    def get_organization_members(self, organization_uri: Optional[str] = None) -> List[OrganizationMember]:
        """
        Extract all organization members with detailed information
        
        Args:
            organization_uri (str): Organization URI
            
        Returns:
            List[OrganizationMember]: List of organization members
            
        Raises:
            CalendlyAPIException: If member data extraction fails
        """
        logger.info("Starting organization members extraction")
        
        try:
            if not organization_uri:
                user_info = self.get_user_information()
                organization_uri = user_info.current_organization
            
            if not organization_uri:
                raise CalendlyAPIException("Organization URI is required")
            
            params = {
                "organization": organization_uri,
                "count": 100
            }
            
            members_data = self._get_paginated_data("/organization_memberships", params)
            
            organization_members = []
            for membership in members_data:
                user = membership.get('user', {})
                organization_members.append(OrganizationMember(
                    name=user.get('name', ''),
                    email=user.get('email', ''),
                    user_uri=user.get('uri', ''),
                    role=membership.get('role', ''),
                    status=membership.get('status', ''),
                    organization_uri=membership.get('organization', ''),
                    membership_uri=membership.get('uri', '')
                ))
            
            logger.info(f"Successfully extracted {len(organization_members)} organization members")
            return organization_members
            
        except Exception as e:
            error_msg = f"Error extracting organization members: {str(e)}"
            logger.error(error_msg)
            raise CalendlyAPIException(error_msg)

    def export_user_information(self, filename: Optional[str] = None) -> str:
        """
        Export user information to JSON file
        
        Args:
            filename (str): Output filename
            
        Returns:
            str: Path to exported file
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"calendly_user_info_{timestamp}.json"
        
        try:
            user_info = self.get_user_information()
            
            # Convert dataclass to dictionary
            user_dict = {
                'user_information': {
                    'name': user_info.name,
                    'email': user_info.email,
                    'uri': user_info.uri,
                    'timezone': user_info.timezone,
                    'scheduling_url': user_info.scheduling_url,
                    'created_at': user_info.created_at,
                    'updated_at': user_info.updated_at,
                    'current_organization': user_info.current_organization,
                    'resource_type': user_info.resource_type
                },
                'metadata': {
                    'extracted_at': datetime.now().isoformat(),
                    'source': 'CCDataW',
                    'version': '1.0.0'
                }
            }
            
            with open(filename, 'w') as f:
                json.dump(user_dict, f, indent=2, ensure_ascii=False)
            
            logger.info(f"User information exported to: {filename}")
            return filename
            
        except Exception as e:
            error_msg = f"Error exporting user information: {str(e)}"
            logger.error(error_msg)
            raise CalendlyAPIException(error_msg)

    def export_organization_members(self, filename: Optional[str] = None) -> str:
        """
        Export organization members to JSON file
        
        Args:
            filename (str): Output filename
            
        Returns:
            str: Path to exported file
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"calendly_organization_members_{timestamp}.json"
        
        try:
            members = self.get_organization_members()
            
            members_list = []
            for member in members:
                members_list.append({
                    'name': member.name,
                    'email': member.email,
                    'user_uri': member.user_uri,
                    'role': member.role,
                    'status': member.status,
                    'organization_uri': member.organization_uri,
                    'membership_uri': member.membership_uri
                })
            
            export_data = {
                'organization_members': members_list,
                'metadata': {
                    'total_members': len(members_list),
                    'extracted_at': datetime.now().isoformat(),
                    'source': 'CCDataW',
                    'version': '1.0.0'
                }
            }
            
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Organization members exported to: {filename}")
            return filename
            
        except Exception as e:
            error_msg = f"Error exporting organization members: {str(e)}"
            logger.error(error_msg)
            raise CalendlyAPIException(error_msg)

    def comprehensive_extraction(self, export: bool = True) -> Dict[str, Any]:
        """
        Perform comprehensive data extraction including user, organization, and members
        
        Args:
            export (bool): Whether to export data to files
            
        Returns:
            Dict: Comprehensive extracted data
        """
        logger.info("Starting comprehensive data extraction")
        
        try:
            # Extract all data
            user_info = self.get_user_information()
            organization_info = self.get_organization_information(user_info)
            organization_members = self.get_organization_members(user_info.current_organization)
            
            # Export if requested
            exported_files = {}
            if export:
                exported_files['user_info'] = self.export_user_information()
                exported_files['organization_members'] = self.export_organization_members()
            
            comprehensive_data = {
                'user_information': user_info,
                'organization_information': organization_info,
                'organization_members': organization_members,
                'summary': {
                    'total_members': len(organization_members),
                    'extraction_timestamp': datetime.now().isoformat()
                },
                'exported_files': exported_files if export else {}
            }
            
            logger.info("Comprehensive data extraction completed successfully")
            return comprehensive_data
            
        except Exception as e:
            error_msg = f"Comprehensive extraction failed: {str(e)}"
            logger.error(error_msg)
            raise CalendlyAPIException(error_msg)


# Usage Example
if __name__ == "__main__":
    import os
    
    # Initialize with your access token
    access_token = os.getenv('CALENDLY_ACCESS_TOKEN', 'your_personal_access_token_here')
    
    if access_token == 'your_personal_access_token_here':
        print("Please set CALENDLY_ACCESS_TOKEN environment variable")
    else:
        data_extractor = CCDataW(access_token=access_token)
        
        try:
            # Comprehensive extraction
            result = data_extractor.comprehensive_extraction(export=True)
            
            # Individual method usage examples
            user_info = data_extractor.get_user_information()
            print(f"User: {user_info.name}")
            
            members = data_extractor.get_organization_members()
            print(f"Total members: {len(members)}")
            
        except CalendlyAPIException as e:
            print(f"Extraction failed: {e}")
