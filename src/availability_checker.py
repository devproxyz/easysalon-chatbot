"""
Availability Checker Module for EasySalon Chatbot
Handles appointment availability checking functionality.
"""

import json
import logging
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

from src import global_vars


@dataclass
class TimeSlot:
    """Represents an available time slot."""
    datetime: datetime
    duration: int  # in minutes
    service_id: str
    staff_id: Optional[str] = None
    price: Optional[float] = None
    available: bool = True

sample_api_availibility_check = """
import requests
 
url = "https://eoa.easysalon.vn/api/v1/branchs?q=mi&page=1&rowPerPage=1&orderBy=Id&orderType=desc"
 
payload={}
headers = {"X-Api-Key": global_vars.EASYSALON_API_KEY}
 
response = requests.request("GET", url, headers=headers, data=payload)
 
print(response.text)
"""
sample_availibility="""
{
  "meta": {
    "totalItem": 1,
    "totalPage": 1,
    "rowPerPage": 1,
    "currentPage": 1,
    "query": {}
  },
  "data": [
    {
      "address": "20 mai lão bạng",
      "email": "mimi@gmail.com",
      "location": null,
      "mobile": "0589956650",
      "name": "mi mi salon",
      "branchCode": "CN00000007",
      "googleMapLink": null,
      "latitude": null,
      "longitude": null,
      "openHour": null,
      "openHourFrom": null,
      "openHourTo": null,
      "id": 7,
      "created": "2023-04-03T09:59:10",
      "updated": "2024-10-14T11:37:02",
      "status": "ENABLE"
    }
  ]
}
"""


@dataclass
class AvailabilityQuery:
    """Represents a user's availability query."""
    date: Optional[str] = None
    time: Optional[str] = None
    service_type: Optional[str] = None
    staff_preference: Optional[str] = None
    salon_id: Optional[str] = None
    branch_id: Optional[str] = None


class AvailabilityChecker:
    """
    Handles appointment availability checking for EasySalon services.
    Integrates with EasySalon API to provide real-time availability data.
    """
    
    def __init__(self):
        """Initialize the availability checker."""
        self.logger = logging.getLogger(__name__)
        self.api_base_url = "https://eoa.easysalon.vn/api/v1"  # EasySalon API base URL
        self.api_timeout = 30
        self.max_retries = 3
        
    def check_availability(self, query: AvailabilityQuery) -> Dict[str, Any]:
        """
        Check availability based on user query using EasySalon API.
        
        Args:
            query: AvailabilityQuery object containing user preferences
            
        Returns:
            Dict containing availability information and formatted response
        """
        try:
            # Get salon branches from EasySalon API
            branches_data = self._fetch_salon_branches(query)
            
            if not branches_data:
                return self._create_error_response(
                    "I couldn't find any salon branches. Please try again later."
                )
            
            # Get service categories and services for better slot generation
            service_categories = self._fetch_service_categories()
            services_data = self._fetch_services(query)
            products_data = self._fetch_products(query)
            
            # Find available slots based on branch data and services
            availability_data = self._process_branch_availability(
                branches_data, query, services_data, products_data
            )
            
            if not availability_data:
                return self._create_no_availability_response(query)
            
            # Format response for chatbot
            formatted_response = self._format_availability_response(availability_data)
            
            return {
                "success": True,
                "branches": branches_data,
                "services": services_data,
                "products": products_data,
                "service_categories": service_categories,
                "slots": availability_data,
                "formatted_response": formatted_response,
                "query": query
            }
            
        except Exception as e:
            self.logger.error(f"Error checking availability: {str(e)}")
            return self._create_error_response(
                "I'm having trouble checking availability right now. Please try again in a moment."
            )
    
    def _validate_query(self, query: AvailabilityQuery) -> bool:
        """
        Validate availability query parameters.
        
        Args:
            query: AvailabilityQuery to validate
            
        Returns:
            True if query is valid, False otherwise
        """
        # Basic validation - at least some criteria should be provided
        if not any([query.date, query.service_type, query.salon_id]):
            return False
        
        # Validate date format if provided
        if query.date:
            try:
                datetime.strptime(query.date, "%Y-%m-%d")
            except ValueError:
                return False
        
        return True
    
    def _fetch_availability_from_api(self, query: AvailabilityQuery) -> List[TimeSlot]:
        """
        Fetch availability data from EasySalon API.
        
        Args:
            query: AvailabilityQuery containing search criteria
            
        Returns:
            List of TimeSlot objects
        """
        try:
            # Prepare API request
            api_url = f"{self.api_base_url}/booking-availability"
            
            params = {
                "date": query.date or datetime.now().strftime("%Y-%m-%d"),
                "service_type": query.service_type,
                "salon_id": query.salon_id,
                "branch_id": query.branch_id,
                "staff_id": query.staff_preference
            }
            
            # Remove None values
            params = {k: v for k, v in params.items() if v is not None}
            
            # Make API request with retry logic
            for attempt in range(self.max_retries):
                try:
                    response = requests.get(
                        api_url,
                        params=params,
                        timeout=self.api_timeout,
                        headers={
                            "Authorization": f"Bearer {global_vars.EASYSALON_API_KEY}",
                            "Content-Type": "application/json"
                        }
                    )
                    
                    response.raise_for_status()
                    
                    # Parse response
                    data = response.json()
                    return self._parse_api_response(data)
                    
                except requests.exceptions.RequestException as e:
                    self.logger.warning(f"API request attempt {attempt + 1} failed: {str(e)}")
                    if attempt == self.max_retries - 1:
                        raise
                    
        except Exception as e:
            self.logger.error(f"Failed to fetch availability from API: {str(e)}")
            # Return mock data for development
            return self._get_mock_availability_data(query)
    
    def _parse_api_response(self, data: Dict[str, Any]) -> List[TimeSlot]:
        """
        Parse API response into TimeSlot objects.
        
        Args:
            data: Raw API response data
            
        Returns:
            List of TimeSlot objects
        """
        slots = []
        
        for slot_data in data.get("available_slots", []):
            try:
                slot = TimeSlot(
                    datetime=datetime.fromisoformat(slot_data["datetime"]),
                    duration=slot_data.get("duration", 60),
                    service_id=slot_data["service_id"],
                    staff_id=slot_data.get("staff_id"),
                    price=slot_data.get("price"),
                    available=slot_data.get("available", True)
                )
                slots.append(slot)
            except (KeyError, ValueError) as e:
                self.logger.warning(f"Invalid slot data: {slot_data}, error: {e}")
                continue
        
        return slots
    
    def _get_mock_availability_data(self, query: AvailabilityQuery) -> List[TimeSlot]:
        """
        Generate mock availability data for development/testing.
        
        Args:
            query: AvailabilityQuery for context
            
        Returns:
            List of mock TimeSlot objects
        """
        mock_slots = []
        base_date = datetime.now().replace(hour=9, minute=0, second=0, microsecond=0)
        
        # Generate some mock slots for the next few days
        for day_offset in range(3):
            current_date = base_date + timedelta(days=day_offset)
            
            # Generate slots for business hours (9 AM - 6 PM)
            for hour in range(9, 18):
                slot_time = current_date.replace(hour=hour)
                
                # Create some variety in availability
                available = not (hour == 12 or (hour == 15 and day_offset == 0))
                
                mock_slots.append(TimeSlot(
                    datetime=slot_time,
                    duration=60,
                    service_id=query.service_type or "haircut",
                    staff_id="staff_001",
                    price=50.0,
                    available=available
                ))
        
        return mock_slots
    
    def _format_availability_response(self, slots: List[TimeSlot]) -> str:
        """
        Format availability data into a user-friendly response.
        
        Args:
            slots: List of TimeSlot objects
            
        Returns:
            Formatted string response for the chatbot
        """
        if not slots:
            return "I couldn't find any available slots for your request. Would you like to try different dates or services?"
        
        # Filter available slots
        available_slots = [slot for slot in slots if slot.available]
        
        if not available_slots:
            return "Unfortunately, all slots are currently booked. Here are some alternative suggestions:\n\n" + \
                   self._generate_alternative_suggestions(slots)
        
        # Group slots by date
        slots_by_date = {}
        for slot in available_slots[:10]:  # Limit to 10 slots
            date_str = slot.datetime.strftime("%Y-%m-%d")
            if date_str not in slots_by_date:
                slots_by_date[date_str] = []
            slots_by_date[date_str].append(slot)
        
        # Format response
        response = "Here are the available appointment slots:\n\n"
        
        for date, day_slots in slots_by_date.items():
            date_obj = datetime.strptime(date, "%Y-%m-%d")
            formatted_date = date_obj.strftime("%A, %B %d, %Y")
            
            response += f"**{formatted_date}:**\n"
            
            for slot in day_slots:
                time_str = slot.datetime.strftime("%I:%M %p")
                price_str = f"{slot.price:,.0f} VND" if slot.price else "Price on request"
                
                response += f"• {time_str} - {slot.duration} minutes ({price_str})\n"
            
            response += "\n"
        
        response += "Would you like to book any of these slots? Just let me know which one you prefer!"
        
        return response
    
    def _generate_alternative_suggestions(self, slots: List[TimeSlot]) -> str:
        """
        Generate alternative suggestions when no slots are available.
        
        Args:
            slots: List of all slots (including unavailable ones)
            
        Returns:
            String with alternative suggestions
        """
        # Find the next available date
        future_date = datetime.now() + timedelta(days=1)
        
        suggestions = []
        suggestions.append(f"• Try booking for {future_date.strftime('%A, %B %d')}")
        suggestions.append("• Consider flexible timing (morning or evening slots)")
        suggestions.append("• Check availability at other nearby salon branches")
        suggestions.append("• Book for a different service that might have availability")
        
        return "\n".join(suggestions)
    
    def _create_error_response(self, message: str) -> Dict[str, Any]:
        """
        Create an error response dictionary.
        
        Args:
            message: Error message to display
            
        Returns:
            Error response dictionary
        """
        return {
            "success": False,
            "error": message,
            "formatted_response": f"❌ {message}"
        }
    
    def _create_no_availability_response(self, query: AvailabilityQuery) -> Dict[str, Any]:
        """
        Create a response when no availability is found.
        
        Args:
            query: Original query for context
            
        Returns:
            No availability response dictionary
        """
        message = "I couldn't find any available slots matching your criteria. " + \
                 "Would you like me to suggest alternative dates or services?"
        
        return {
            "success": True,
            "slots": [],
            "formatted_response": message,
            "query": query
        }
    
    def _fetch_salon_branches(self, query: AvailabilityQuery) -> List[Dict[str, Any]]:
        """
        Fetch salon branches from EasySalon API.
        
        Args:
            query: AvailabilityQuery containing search criteria
            
        Returns:
            List of branch data dictionaries
        """
        try:
            # Build search query - use salon name or location if provided
            search_query = ""
            if query.salon_id:
                search_query = query.salon_id
            elif hasattr(query, 'location') and query.location:
                search_query = query.location
            else:
                search_query = "salon"  # Default search term
            
            # Prepare API request based on sample
            api_url = f"{self.api_base_url}/branchs"
            
            params = {
                "page": 1,
                "rowPerPage": 9999,  # Get more results for better availability
                "orderBy": "Id",
                "orderType": "desc"
            }
            
            # Make API request with retry logic
            for attempt in range(self.max_retries):
                try:
                    headers = {"X-Api-Key": global_vars.EASYSALON_API_KEY}
                    response = requests.get(
                        api_url,
                        params=params,
                        timeout=self.api_timeout,
                        headers=headers # No authentication required based on API samples
                    )
                    
                    response.raise_for_status()
                    
                    # Parse response
                    data = response.json()
                    
                    if data.get("data"):
                        return data["data"]
                    else:
                        self.logger.warning("No branches found in API response")
                        return []
                        
                except requests.exceptions.RequestException as e:
                    self.logger.warning(f"API request attempt {attempt + 1} failed: {str(e)}")
                    if attempt == self.max_retries - 1:
                        raise
                    
        except Exception as e:
            self.logger.error(f"Failed to fetch salon branches from API: {str(e)}")
            # Return mock data for development
            return self._get_mock_branch_data()
    
    def _process_branch_availability(
        self, 
        branches_data: List[Dict[str, Any]], 
        query: AvailabilityQuery,
        services_data: List[Dict[str, Any]] = None,
        products_data: List[Dict[str, Any]] = None
    ) -> List[TimeSlot]:
        """
        Process branch data to generate availability slots.
        
        Args:
            branches_data: List of branch data from API
            query: AvailabilityQuery containing search criteria
            services_data: List of service data from API (optional)
            products_data: List of product data from API (optional)
            
        Returns:
            List of TimeSlot objects
        """
        try:
            slots = []
            
            for branch in branches_data:
                # Generate time slots for each branch
                branch_slots = self._generate_slots_for_branch(
                    branch, query, services_data, products_data
                )
                slots.extend(branch_slots)
            
            return slots
            
        except Exception as e:
            self.logger.error(f"Error processing branch availability: {str(e)}")
            return []
    
    def _generate_slots_for_branch(
        self, 
        branch: Dict[str, Any], 
        query: AvailabilityQuery,
        services_data: List[Dict[str, Any]] = None,
        products_data: List[Dict[str, Any]] = None
    ) -> List[TimeSlot]:
        """
        Generate time slots for a specific branch.
        
        Args:
            branch: Branch data dictionary
            query: AvailabilityQuery containing search criteria
            services_data: List of service data from API (optional)
            products_data: List of product data from API (optional)
            
        Returns:
            List of TimeSlot objects for the branch
        """
        slots = []
        
        try:
            # Get branch operating hours (use defaults if not provided)
            open_hour = branch.get("openHourFrom", "09:00")
            close_hour = branch.get("openHourTo", "18:00")
            
            # Parse branch info
            branch_id = str(branch.get("id", ""))
            branch_name = branch.get("name", "Salon")
            branch_code = branch.get("branchCode", "")
            
            # Get service pricing from API data
            service_price = self._get_service_price(query.service_type, services_data, products_data)
            service_duration = self._get_service_duration(query.service_type, services_data)
            
            # Generate slots for the requested date or next 3 days
            base_date = datetime.now()
            if query.date:
                try:
                    base_date = datetime.strptime(query.date, "%Y-%m-%d")
                except ValueError:
                    pass
            
            # Generate slots for multiple days
            for day_offset in range(3):
                current_date = base_date + timedelta(days=day_offset)
                
                # Generate hourly slots during business hours
                start_hour = int(open_hour.split(":")[0]) if open_hour else 9
                end_hour = int(close_hour.split(":")[0]) if close_hour else 18
                
                for hour in range(start_hour, end_hour):
                    slot_time = current_date.replace(hour=hour, minute=0, second=0, microsecond=0)
                    
                    # Create availability based on various factors
                    available = self._is_slot_available(slot_time, branch, query)
                    
                    slot = TimeSlot(
                        datetime=slot_time,
                        duration=service_duration,
                        service_id=query.service_type or "general",
                        staff_id=f"staff_{branch_id}",
                        price=service_price,
                        available=available
                    )
                    
                    slots.append(slot)
            
            return slots
            
        except Exception as e:
            self.logger.error(f"Error generating slots for branch {branch.get('name', 'Unknown')}: {str(e)}")
            return []
    
    def _is_slot_available(self, slot_time: datetime, branch: Dict[str, Any], query: AvailabilityQuery) -> bool:
        """
        Determine if a specific time slot is available.
        
        Args:
            slot_time: DateTime of the slot
            branch: Branch data dictionary
            query: AvailabilityQuery containing search criteria
            
        Returns:
            True if slot is available, False otherwise
        """
        # Basic availability logic (can be enhanced with real booking data)
        
        # Don't allow booking in the past
        if slot_time < datetime.now():
            return False
        
        # Simulate some unavailable slots (lunch time, popular hours)
        hour = slot_time.hour
        
        # Lunch break (12-13)
        if hour == 12:
            return False
        
        # Very early or very late hours have limited availability
        if hour < 9 or hour > 17:
            return False
        
        # Weekend might have different availability
        if slot_time.weekday() >= 5:  # Saturday = 5, Sunday = 6
            # Weekend slots are more limited
            return hour not in [10, 14, 16]
        
        # Weekday prime hours might be busy
        if hour in [11, 15, 17]:
            return slot_time.minute == 0  # Only on the hour
        
        return True
    
    def _get_mock_branch_data(self) -> List[Dict[str, Any]]:
        """
        Generate mock branch data for development/testing.
        
        Returns:
            List of mock branch data dictionaries
        """
        return [
            {
                "id": 1,
                "name": "EasySalon Downtown",
                "branchCode": "CN00000001",
                "address": "123 Main Street, Downtown",
                "email": "downtown@easysalon.vn",
                "mobile": "0123456789",
                "openHourFrom": "09:00",
                "openHourTo": "18:00",
                "status": "ENABLE"
            },
            {
                "id": 2,
                "name": "EasySalon Uptown",
                "branchCode": "CN00000002",
                "address": "456 Oak Avenue, Uptown",
                "email": "uptown@easysalon.vn",
                "mobile": "0987654321",
                "openHourFrom": "08:00",
                "openHourTo": "20:00",
                "status": "ENABLE"
            }
        ]
    
    def parse_availability_query(self, query_string: str) -> AvailabilityQuery:
        """
        Parse a natural language query string into an AvailabilityQuery object.
        
        Args:
            query_string: Natural language query from user
            
        Returns:
            AvailabilityQuery object
        """
        query = AvailabilityQuery()
        
        # Convert to lowercase for easier matching
        query_lower = query_string.lower()
        
        # Extract date information
        if "today" in query_lower:
            query.date = datetime.now().strftime("%Y-%m-%d")
        elif "tomorrow" in query_lower:
            query.date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        elif "next week" in query_lower:
            query.date = (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")
        
        # Extract service type
        if "haircut" in query_lower or "hair" in query_lower:
            query.service_type = "haircut"
        elif "facial" in query_lower or "skincare" in query_lower:
            query.service_type = "facial"
        elif "massage" in query_lower:
            query.service_type = "massage"
        elif "manicure" in query_lower or "nail" in query_lower:
            query.service_type = "manicure"
        elif "pedicure" in query_lower:
            query.service_type = "pedicure"
        
        # Extract time preferences
        if "morning" in query_lower:
            query.time = "morning"
        elif "afternoon" in query_lower:
            query.time = "afternoon"
        elif "evening" in query_lower:
            query.time = "evening"
        
        return query

    def _fetch_service_categories(self) -> List[Dict[str, Any]]:
        """
        Fetch service categories from EasySalon API.
        
        Returns:
            List of service category data dictionaries
        """
        try:
            # Prepare API request based on sample
            api_url = f"{self.api_base_url}/services/categories"
            
            params = {
                "q": "",
                "page": 1,
                "rowPerPage": 50,  # Get more categories
                "orderBy": "Id",
                "orderType": "desc"
            }
            
            # Make API request with retry logic
            for attempt in range(self.max_retries):
                try:
                    headers = {"X-Api-Key": global_vars.EASYSALON_API_KEY}
                    response = requests.get(
                        api_url,
                        params=params,
                        timeout=self.api_timeout,
                        headers=headers  # No authentication required based on API samples
                    )
                    
                    response.raise_for_status()
                    
                    # Parse response
                    data = response.json()
                    
                    if data.get("data"):
                        return data["data"]
                    else:
                        self.logger.warning("No service categories found in API response")
                        return []
                        
                except requests.exceptions.RequestException as e:
                    self.logger.warning(f"API request attempt {attempt + 1} failed: {str(e)}")
                    if attempt == self.max_retries - 1:
                        raise
                    
        except Exception as e:
            self.logger.error(f"Failed to fetch service categories from API: {str(e)}")
            # Return mock data for development
            return self._get_mock_service_categories()
    
    def _fetch_services(self, query: AvailabilityQuery) -> List[Dict[str, Any]]:
        """
        Fetch services from EasySalon API (using products endpoint as services).
        
        Args:
            query: AvailabilityQuery containing search criteria
            
        Returns:
            List of service data dictionaries
        """
        try:
            # Build search query
            search_query = ""
            if query.service_type:
                search_query = query.service_type
            
            # Prepare API request based on sample
            api_url = f"{self.api_base_url}/products"
            
            params = {
                "q": search_query,
                "branchId": query.branch_id or "",
                "productStatusId": "",
                "categoryId": "",  # Can be filtered by category if needed
                "page": 1,
                "rowPerPage": 20,  # Get more services
                "orderBy": "",
                "orderType": ""
            }
            
            # Make API request with retry logic
            for attempt in range(self.max_retries):
                try:
                    headers = {"X-Api-Key": global_vars.EASYSALON_API_KEY}
                    response = requests.get(
                        api_url,
                        params=params,
                        timeout=self.api_timeout,
                        headers=headers  # No authentication required based on API samples
                    )
                    
                    response.raise_for_status()
                    
                    # Parse response
                    data = response.json()
                    
                    if data.get("data"):
                        return data["data"]
                    else:
                        self.logger.warning("No services found in API response")
                        # Continue to next attempt or fall through to mock data
                        if attempt == self.max_retries - 1:
                            # Last attempt failed, fall through to mock data
                            break
                        
                except requests.exceptions.RequestException as e:
                    self.logger.warning(f"API request attempt {attempt + 1} failed: {str(e)}")
                    if attempt == self.max_retries - 1:
                        raise
                    
        except Exception as e:
            self.logger.error(f"Failed to fetch services from API: {str(e)}")
            
        # Return mock data for development when API fails or has no data
        return self._get_mock_services_data()
    
    def _fetch_products(self, query: AvailabilityQuery) -> List[Dict[str, Any]]:
        """
        Fetch products from EasySalon API.
        
        Args:
            query: AvailabilityQuery containing search criteria
            
        Returns:
            List of product data dictionaries
        """
        try:
            # Build search query
            search_query = ""
            if query.service_type:
                search_query = query.service_type
            
            # Prepare API request based on sample
            api_url = f"{self.api_base_url}/products"
            
            params = {
                "q": search_query,
                "branchId": query.branch_id or "",
                "productStatusId": "",
                "categoryId": "12",  # Sample category ID from API docs
                "page": 1,
                "rowPerPage": 20,  # Get more products
                "orderBy": "",
                "orderType": ""
            }
            
            # Make API request with retry logic
            for attempt in range(self.max_retries):
                try:
                    headers = {"X-Api-Key": global_vars.EASYSALON_API_KEY}
                    response = requests.get(
                        api_url,
                        params=params,
                        timeout=self.api_timeout,
                        headers=headers  # No authentication required based on API samples
                    )
                    
                    response.raise_for_status()
                    
                    # Parse response
                    data = response.json()
                    
                    if data.get("data"):
                        return data["data"]
                    else:
                        self.logger.warning("No products found in API response")
                        return []
                        
                except requests.exceptions.RequestException as e:
                    self.logger.warning(f"API request attempt {attempt + 1} failed: {str(e)}")
                    if attempt == self.max_retries - 1:
                        raise
                    
        except Exception as e:
            self.logger.error(f"Failed to fetch products from API: {str(e)}")
            # Return mock data for development
            return self._get_mock_products_data()
    
    def _get_branch_detail(self, branch_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific branch.
        
        Args:
            branch_id: ID of the branch to get details for
            
        Returns:
            Branch detail data dictionary
        """
        try:
            # Prepare API request based on sample
            api_url = f"{self.api_base_url}/branchs/{branch_id}"
            
            # Make API request with retry logic
            for attempt in range(self.max_retries):
                try:
                    headers = {"X-Api-Key": global_vars.EASYSALON_API_KEY}
                    response = requests.get(
                        api_url,
                        timeout=self.api_timeout,
                        headers=headers  # No authentication required based on API samples
                    )
                    
                    response.raise_for_status()
                    
                    # Parse response
                    data = response.json()
                    
                    if data.get("data"):
                        return data["data"]
                    else:
                        self.logger.warning(f"No branch detail found for ID: {branch_id}")
                        return {}
                        
                except requests.exceptions.RequestException as e:
                    self.logger.warning(f"API request attempt {attempt + 1} failed: {str(e)}")
                    if attempt == self.max_retries - 1:
                        raise
                    
        except Exception as e:
            self.logger.error(f"Failed to fetch branch detail from API: {str(e)}")
            # Return mock data for development
            return self._get_mock_branch_detail(branch_id)
    
    def _get_mock_service_categories(self) -> List[Dict[str, Any]]:
        """
        Generate mock service categories for development/testing.
        
        Returns:
            List of mock service category data dictionaries
        """
        return [
            {
                "id": 1,
                "name": "Hair Services",
                "description": "Professional hair care services",
                "slug": "hair-services"
            },
            {
                "id": 2,
                "name": "Nail Services",
                "description": "Manicure and pedicure services",
                "slug": "nail-services"
            },
            {
                "id": 3,
                "name": "Facial Services",
                "description": "Skincare and facial treatments",
                "slug": "facial-services"
            },
            {
                "id": 4,
                "name": "Massage Services",
                "description": "Relaxation and therapeutic massage",
                "slug": "massage-services"
            }
        ]
    
    def _get_mock_services_data(self) -> List[Dict[str, Any]]:
        """
        Generate mock services data for development/testing.
        
        Returns:
            List of mock service data dictionaries
        """
        return [
            {
                "id": 1,
                "name": "Hair Cut",
                "price": 200000,
                "originPrice": 150000,
                "productCategoryId": 1,
                "status": "ENABLE"
            },
            {
                "id": 2,
                "name": "Hair Coloring",
                "price": 500000,
                "originPrice": 400000,
                "productCategoryId": 1,
                "status": "ENABLE"
            },
            {
                "id": 3,
                "name": "Facial Treatment",
                "price": 300000,
                "originPrice": 250000,
                "productCategoryId": 3,
                "status": "ENABLE"
            },
            {
                "id": 4,
                "name": "Manicure",
                "price": 150000,
                "originPrice": 120000,
                "productCategoryId": 2,
                "status": "ENABLE"
            }
        ]
    
    def _get_mock_products_data(self) -> List[Dict[str, Any]]:
        """
        Generate mock products data for development/testing.
        
        Returns:
            List of mock product data dictionaries
        """
        return [
            {
                "id": 542,
                "name": "Dầu oliu",
                "price": 300000,
                "originPrice": 100000,
                "productCategoryId": 12,
                "status": "ENABLE"
            },
            {
                "id": 543,
                "name": "Serum dưỡng da",
                "price": 450000,
                "originPrice": 350000,
                "productCategoryId": 12,
                "status": "ENABLE"
            }
        ]
    
    def _get_mock_branch_detail(self, branch_id: str) -> Dict[str, Any]:
        """
        Generate mock branch detail for development/testing.
        
        Args:
            branch_id: ID of the branch
            
        Returns:
            Mock branch detail data dictionary
        """
        return {
            "id": int(branch_id) if branch_id.isdigit() else 1,
            "name": f"EasySalon Branch {branch_id}",
            "branchCode": f"CN{branch_id.zfill(8)}",
            "address": "123 Main Street, City Center",
            "email": f"branch{branch_id}@easysalon.vn",
            "mobile": "0123456789",
            "openHourFrom": "09:00",
            "openHourTo": "18:00",
            "status": "ENABLE"
        }
    
    def _get_service_price(self, service_type: str, services_data: List[Dict[str, Any]] = None, products_data: List[Dict[str, Any]] = None) -> float:
        """
        Get service price from API data.
        
        Args:
            service_type: Type of service to get price for
            services_data: List of service data from API
            products_data: List of product data from API
            
        Returns:
            Service price in VND
        """
        try:
            # First check services data
            if services_data:
                for service in services_data:
                    if service_type and service_type.lower() in service.get("name", "").lower():
                        return float(service.get("price", 0))
            
            # Then check products data
            if products_data:
                for product in products_data:
                    if service_type and service_type.lower() in product.get("name", "").lower():
                        return float(product.get("price", 0))
            
            # Default prices based on service type
            default_prices = {
                "haircut": 200000,
                "hair_coloring": 500000,
                "facial": 300000,
                "manicure": 150000,
                "pedicure": 180000,
                "massage": 250000
            }
            
            return default_prices.get(service_type, 200000)
            
        except Exception as e:
            self.logger.error(f"Error getting service price: {str(e)}")
            return 200000  # Default price
    
    def _get_service_duration(self, service_type: str, services_data: List[Dict[str, Any]] = None) -> int:
        """
        Get service duration from API data or defaults.
        
        Args:
            service_type: Type of service to get duration for
            services_data: List of service data from API
            
        Returns:
            Service duration in minutes
        """
        try:
            # Default durations based on service type
            default_durations = {
                "haircut": 60,
                "hair_coloring": 120,
                "facial": 90,
                "manicure": 45,
                "pedicure": 60,
                "massage": 60
            }
            
            return default_durations.get(service_type, 60)
            
        except Exception as e:
            self.logger.error(f"Error getting service duration: {str(e)}")
            return 60  # Default duration
    
    def create_booking_from_slot(self, slot: TimeSlot, customer_info: Dict[str, Any], branch_id: str) -> Dict[str, Any]:
        """
        Create a booking using the EasySalon booking API based on a selected slot.
        
        Args:
            slot: TimeSlot object representing the selected appointment time
            customer_info: Customer information dictionary
            branch_id: ID of the branch to book at
            
        Returns:
            Booking result dictionary
        """
        try:
            # Prepare booking data in EasySalon API format
            booking_payload = {
                "customerMobile": customer_info.get("phone", ""),
                "customerName": customer_info.get("name", ""),
                "totalCustomer": 1,
                "branchId": int(branch_id),
                "bookingDetails": [
                    {
                        "serviceStaffs": [
                            {
                                "serviceId": int(slot.service_id) if str(slot.service_id).isdigit() else 1,
                                "staffId": int(slot.staff_id.split("_")[-1]) if slot.staff_id and "_" in slot.staff_id else 0
                            }
                        ]
                    }
                ],
                "bookingDate": slot.datetime.strftime("%Y-%m-%d"),
                "bookingTime": slot.datetime.strftime("%H:%M")
            }
            
            # Make API request with retry logic
            for attempt in range(self.max_retries):
                try:
                    headers = {"X-Api-Key": global_vars.EASYSALON_API_KEY,
                               "Content-Type": "application/json"}
                    response = requests.post(
                        f"{self.api_base_url}/booking",
                        json=booking_payload,
                        timeout=self.api_timeout,
                        headers=headers
                    )
                    
                    response.raise_for_status()
                    
                    # Parse response
                    api_response = response.json()
                    
                    if api_response.get("data"):
                        booking_result = api_response["data"]
                        return {
                            "success": True,
                            "booking_id": str(booking_result.get("id", "")),
                            "booking_code": booking_result.get("bookingCode", ""),
                            "booking_status": booking_result.get("bookingStatus", ""),
                            "total": booking_result.get("total", 0),
                            "date": booking_result.get("date", ""),
                            "message": "Booking created successfully!",
                            "api_response": booking_result
                        }
                    else:
                        return {
                            "success": False,
                            "message": "Failed to create booking - no data in response"
                        }
                        
                except requests.exceptions.RequestException as e:
                    self.logger.warning(f"Booking API request attempt {attempt + 1} failed: {str(e)}")
                    if attempt == self.max_retries - 1:
                        raise
                    
        except Exception as e:
            self.logger.error(f"Error creating booking from slot: {str(e)}")
            return {
                "success": False,
                "message": f"Failed to create booking: {str(e)}"
            }
