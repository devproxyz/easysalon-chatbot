"""
Booking Retrieval Module for EasySalon Chatbot
Handles booking information retrieval and lookup functionality.
"""

import json
import logging
import requests
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

from src import global_vars


@dataclass
class BookingInfo:
    """Represents retrieved booking information."""
    booking_id: str
    confirmation_code: str
    service_name: str
    salon_name: str
    date: str
    time: str
    duration: int
    price: float
    staff_name: str
    customer_name: str
    customer_phone: str
    status: str
    special_requests: Optional[str] = None
    created_at: Optional[str] = None
    salon_address: Optional[str] = None
    salon_phone: Optional[str] = None


@dataclass
class BookingSearchResult:
    """Represents booking search result."""
    success: bool
    booking_info: Optional[BookingInfo] = None
    message: str = ""
    error_code: Optional[str] = None


class BookingRetriever:
    """
    Handles booking information retrieval for EasySalon services.
    Integrates with EasySalon API to fetch booking details.
    """
    
    def __init__(self):
        """Initialize the booking retriever."""
        self.logger = logging.getLogger(__name__)
        self.api_base_url = "https://www.beautysalon.vn"  # Mock API base URL
        self.api_timeout = 30
        self.max_retries = 3
        
    def retrieve_booking(self, booking_identifier: str) -> BookingSearchResult:
        """
        Retrieve booking information by ID or confirmation code.
        
        Args:
            booking_identifier: Booking ID or confirmation code
            
        Returns:
            BookingSearchResult object with booking information
        """
        try:
            # Validate booking identifier
            if not self._validate_booking_identifier(booking_identifier):
                return BookingSearchResult(
                    success=False,
                    message="Invalid booking ID or confirmation code format",
                    error_code="INVALID_IDENTIFIER"
                )
            
            # Determine identifier type
            identifier_type = self._determine_identifier_type(booking_identifier)
            
            # Make API call to retrieve booking
            api_response = self._make_retrieval_api_call(booking_identifier, identifier_type)
            
            if api_response.get("success", False):
                booking_data = api_response.get("booking_data", {})
                
                booking_info = BookingInfo(
                    booking_id=booking_data.get("booking_id"),
                    confirmation_code=booking_data.get("confirmation_code"),
                    service_name=booking_data.get("service_name"),
                    salon_name=booking_data.get("salon_name"),
                    date=booking_data.get("date"),
                    time=booking_data.get("time"),
                    duration=booking_data.get("duration"),
                    price=booking_data.get("price"),
                    staff_name=booking_data.get("staff_name"),
                    customer_name=booking_data.get("customer_name"),
                    customer_phone=booking_data.get("customer_phone"),
                    status=booking_data.get("status"),
                    special_requests=booking_data.get("special_requests"),
                    created_at=booking_data.get("created_at"),
                    salon_address=booking_data.get("salon_address"),
                    salon_phone=booking_data.get("salon_phone")
                )
                
                return BookingSearchResult(
                    success=True,
                    booking_info=booking_info,
                    message="Booking found successfully"
                )
            else:
                return BookingSearchResult(
                    success=False,
                    message=api_response.get("message", "Booking not found"),
                    error_code="BOOKING_NOT_FOUND"
                )
                
        except Exception as e:
            self.logger.error(f"Error retrieving booking: {e}")
            return BookingSearchResult(
                success=False,
                message="An error occurred while retrieving the booking. Please try again.",
                error_code="SYSTEM_ERROR"
            )
    
    def search_bookings_by_phone(self, phone: str) -> List[BookingInfo]:
        """
        Search for bookings by customer phone number.
        
        Args:
            phone: Customer phone number
            
        Returns:
            List of BookingInfo objects
        """
        try:
            # Validate phone number
            if not self._validate_phone_number(phone):
                return []
            
            # Make API call to search bookings
            api_response = self._make_search_api_call(phone)
            
            if api_response.get("success", False):
                bookings_data = api_response.get("bookings", [])
                bookings = []
                
                for booking_data in bookings_data:
                    booking_info = BookingInfo(
                        booking_id=booking_data.get("booking_id"),
                        confirmation_code=booking_data.get("confirmation_code"),
                        service_name=booking_data.get("service_name"),
                        salon_name=booking_data.get("salon_name"),
                        date=booking_data.get("date"),
                        time=booking_data.get("time"),
                        duration=booking_data.get("duration"),
                        price=booking_data.get("price"),
                        staff_name=booking_data.get("staff_name"),
                        customer_name=booking_data.get("customer_name"),
                        customer_phone=booking_data.get("customer_phone"),
                        status=booking_data.get("status"),
                        special_requests=booking_data.get("special_requests"),
                        created_at=booking_data.get("created_at"),
                        salon_address=booking_data.get("salon_address"),
                        salon_phone=booking_data.get("salon_phone")
                    )
                    bookings.append(booking_info)
                
                return bookings
            else:
                return []
                
        except Exception as e:
            self.logger.error(f"Error searching bookings by phone: {e}")
            return []
    
    def _validate_booking_identifier(self, identifier: str) -> bool:
        """Validate booking identifier format."""
        if not identifier or len(identifier) < 5:
            return False
        
        # Check for booking ID format (ESB followed by date and hex)
        booking_id_pattern = r'^ESB\d{8}[A-F0-9]{8}$'
        if re.match(booking_id_pattern, identifier):
            return True
        
        # Check for confirmation code format (EC followed by hex)
        confirmation_code_pattern = r'^EC[A-F0-9]{6}$'
        if re.match(confirmation_code_pattern, identifier):
            return True
        
        return False
    
    def _determine_identifier_type(self, identifier: str) -> str:
        """Determine if identifier is booking ID or confirmation code."""
        if identifier.startswith("ESB"):
            return "booking_id"
        elif identifier.startswith("EC"):
            return "confirmation_code"
        else:
            return "unknown"
    
    def _validate_phone_number(self, phone: str) -> bool:
        """Validate phone number format."""
        # Basic phone number validation
        phone_pattern = r'^\+?[\d\s\-\(\)]{10,15}$'
        return bool(re.match(phone_pattern, phone))
    
    def _make_retrieval_api_call(self, identifier: str, identifier_type: str) -> Dict[str, Any]:
        """
        Make API call to retrieve booking information (mock implementation).
        
        Args:
            identifier: Booking identifier
            identifier_type: Type of identifier (booking_id or confirmation_code)
            
        Returns:
            API response data
        """
        # Mock API response for development
        return {
            "success": True,
            "booking_data": {
                "booking_id": "ESB20241201ABC123DE",
                "confirmation_code": "EC123ABC",
                "service_name": "Haircut & Styling",
                "salon_name": "EasySalon Downtown",
                "date": "2024-12-15",
                "time": "14:00",
                "duration": 60,
                "price": 65.00,
                "staff_name": "Sarah Johnson",
                "customer_name": "John Smith",
                "customer_phone": "+1234567890",
                "status": "confirmed",
                "special_requests": "Please use organic products",
                "created_at": "2024-12-01T10:30:00Z",
                "salon_address": "123 Main St, Downtown",
                "salon_phone": "+1234567891"
            }
        }
    
    def _make_search_api_call(self, phone: str) -> Dict[str, Any]:
        """
        Make API call to search bookings by phone (mock implementation).
        
        Args:
            phone: Customer phone number
            
        Returns:
            API response data
        """
        # Mock API response for development
        return {
            "success": True,
            "bookings": [
                {
                    "booking_id": "ESB20241201ABC123DE",
                    "confirmation_code": "EC123ABC",
                    "service_name": "Haircut & Styling",
                    "salon_name": "EasySalon Downtown",
                    "date": "2024-12-15",
                    "time": "14:00",
                    "duration": 60,
                    "price": 65.00,
                    "staff_name": "Sarah Johnson",
                    "customer_name": "John Smith",
                    "customer_phone": phone,
                    "status": "confirmed",
                    "special_requests": "Please use organic products",
                    "created_at": "2024-12-01T10:30:00Z",
                    "salon_address": "123 Main St, Downtown",
                    "salon_phone": "+1234567891"
                }
            ]
        }
    
    def parse_booking_query(self, query: str) -> Dict[str, Any]:
        """
        Parse natural language query to extract booking identifier.
        
        Args:
            query: User's booking lookup request
            
        Returns:
            Parsed booking information
        """
        parsed_info = {
            "identifier": None,
            "phone": None,
            "query_type": None
        }
        
        query_upper = query.upper()
        
        # Look for booking ID pattern
        booking_id_match = re.search(r'ESB\d{8}[A-F0-9]{8}', query_upper)
        if booking_id_match:
            parsed_info["identifier"] = booking_id_match.group()
            parsed_info["query_type"] = "booking_id"
            return parsed_info
        
        # Look for confirmation code pattern
        confirmation_code_match = re.search(r'EC[A-F0-9]{6}', query_upper)
        if confirmation_code_match:
            parsed_info["identifier"] = confirmation_code_match.group()
            parsed_info["query_type"] = "confirmation_code"
            return parsed_info
        
        # Look for phone number
        phone_match = re.search(r'[\+]?[\d\s\-\(\)]{10,15}', query)
        if phone_match:
            parsed_info["phone"] = phone_match.group()
            parsed_info["query_type"] = "phone"
            return parsed_info
        
        return parsed_info
    
    def format_booking_info(self, booking_info: BookingInfo) -> str:
        """
        Format booking information for display.
        
        Args:
            booking_info: BookingInfo object
            
        Returns:
            Formatted booking information
        """
        status_emoji = {
            "confirmed": "✅",
            "cancelled": "❌",
            "completed": "✅",
            "pending": "⏳"
        }
        
        emoji = status_emoji.get(booking_info.status.lower(), "📋")
        
        formatted_info = f"""
{emoji} **Booking Information**

**Booking Details:**
- **Booking ID**: `{booking_info.booking_id}`
- **Confirmation Code**: `{booking_info.confirmation_code}`
- **Service**: {booking_info.service_name}
- **Date**: {booking_info.date}
- **Time**: {booking_info.time}
- **Duration**: {booking_info.duration} minutes
- **Price**: ${booking_info.price:.2f}
- **Status**: {booking_info.status.title()}

**Salon Information:**
- **Salon**: {booking_info.salon_name}
- **Staff**: {booking_info.staff_name}
{f"- **Address**: {booking_info.salon_address}" if booking_info.salon_address else ""}
{f"- **Phone**: {booking_info.salon_phone}" if booking_info.salon_phone else ""}

**Customer Information:**
- **Name**: {booking_info.customer_name}
- **Phone**: {booking_info.customer_phone}

{f"**Special Requests**: {booking_info.special_requests}" if booking_info.special_requests else ""}

{f"**Booking Created**: {booking_info.created_at}" if booking_info.created_at else ""}
"""
        
        return formatted_info.strip()
    
    def format_booking_list(self, bookings: List[BookingInfo]) -> str:
        """
        Format list of bookings for display.
        
        Args:
            bookings: List of BookingInfo objects
            
        Returns:
            Formatted booking list
        """
        if not bookings:
            return "No bookings found."
        
        formatted_list = f"**Found {len(bookings)} booking(s):**\n\n"
        
        for i, booking in enumerate(bookings, 1):
            status_emoji = {
                "confirmed": "✅",
                "cancelled": "❌",
                "completed": "✅",
                "pending": "⏳"
            }
            
            emoji = status_emoji.get(booking.status.lower(), "📋")
            
            formatted_list += f"""
{emoji} **Booking {i}**
- **ID**: `{booking.booking_id}`
- **Service**: {booking.service_name}
- **Date**: {booking.date} at {booking.time}
- **Salon**: {booking.salon_name}
- **Status**: {booking.status.title()}
- **Price**: ${booking.price:.2f}

"""
        
        return formatted_list.strip()


# Create global instance
booking_retriever = BookingRetriever()
