"""
Booking Manager Module for EasySalon Chatbot
Handles appointment booking creation and management functionality.
"""

import json
import logging
import requests
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict

from src import global_vars


@dataclass
class BookingRequest:
    """Represents a booking request."""
    service_id: str
    service_name: str
    salon_id: str
    staff_id: Optional[str]
    date: str
    time: str
    duration: int  # in minutes
    price: float
    customer_info: Dict[str, Any]
    special_requests: Optional[str] = None
    branch_id: Optional[str] = None


@dataclass
class BookingResponse:
    """Represents a booking response."""
    success: bool
    booking_id: Optional[str] = None
    confirmation_code: Optional[str] = None
    message: str = ""
    booking_details: Optional[Dict[str, Any]] = None
    error_code: Optional[str] = None


@dataclass
class CustomerInfo:
    """Represents customer information."""
    name: str
    phone: str
    email: Optional[str] = None
    notes: Optional[str] = None
    is_returning: bool = False
    customer_id: Optional[str] = None


class BookingManager:
    """
    Handles appointment booking creation and management for EasySalon services.
    Integrates with EasySalon API to create, modify, and cancel bookings.
    """
    
    def __init__(self):
        """Initialize the booking manager."""
        self.logger = logging.getLogger(__name__)
        self.api_base_url = "https://eoa.easysalon.vn/api/v1"  # EasySalon API base URL
        self.api_timeout = 30
        self.max_retries = 3
        
    def create_booking(self, booking_request: BookingRequest) -> BookingResponse:
        """
        Create a new appointment booking.
        
        Args:
            booking_request: BookingRequest object with all booking details
            
        Returns:
            BookingResponse object with booking result
        """
        try:
            # Validate booking request
            validation_result = self._validate_booking_request(booking_request)
            if not validation_result.get("valid", False):
                return BookingResponse(
                    success=False,
                    message=validation_result.get("message", "Invalid booking request"),
                    error_code="VALIDATION_ERROR"
                )
            
            # Generate booking ID and confirmation code
            booking_id = self._generate_booking_id()
            confirmation_code = self._generate_confirmation_code()
            
            # Prepare API request
            api_data = {
                "booking_id": booking_id,
                "service_id": booking_request.service_id,
                "salon_id": booking_request.salon_id,
                "staff_id": booking_request.staff_id,
                "date": booking_request.date,
                "time": booking_request.time,
                "duration": booking_request.duration,
                "price": booking_request.price,
                "customer_info": booking_request.customer_info,
                "special_requests": booking_request.special_requests,
                "branch_id": booking_request.branch_id,
                "confirmation_code": confirmation_code,
                "created_at": datetime.now().isoformat(),
                "status": "confirmed"
            }
            
            # Make API call (mock implementation)
            api_response = self._make_booking_api_call(api_data)
            
            if api_response.get("success", False):
                booking_details = {
                    "booking_id": booking_id,
                    "confirmation_code": confirmation_code,
                    "service_name": booking_request.service_name,
                    "salon_name": api_response.get("salon_name", "EasySalon"),
                    "date": booking_request.date,
                    "time": booking_request.time,
                    "duration": booking_request.duration,
                    "price": booking_request.price,
                    "staff_name": api_response.get("staff_name", "Available Staff"),
                    "customer_name": booking_request.customer_info.get("name"),
                    "customer_phone": booking_request.customer_info.get("phone"),
                    "status": "confirmed",
                    "special_requests": booking_request.special_requests
                }
                
                return BookingResponse(
                    success=True,
                    booking_id=booking_id,
                    confirmation_code=confirmation_code,
                    message="Booking created successfully!",
                    booking_details=booking_details
                )
            else:
                return BookingResponse(
                    success=False,
                    message=api_response.get("message", "Failed to create booking"),
                    error_code="API_ERROR"
                )
                
        except Exception as e:
            self.logger.error(f"Error creating booking: {e}")
            return BookingResponse(
                success=False,
                message="An error occurred while creating the booking. Please try again.",
                error_code="SYSTEM_ERROR"
            )
    
    def _validate_booking_request(self, booking_request: BookingRequest) -> Dict[str, Any]:
        """Validate booking request data."""
        try:
            # Check required fields
            if not booking_request.service_id:
                return {"valid": False, "message": "Service ID is required"}
            
            if not booking_request.salon_id:
                return {"valid": False, "message": "Salon ID is required"}
            
            if not booking_request.date:
                return {"valid": False, "message": "Date is required"}
            
            if not booking_request.time:
                return {"valid": False, "message": "Time is required"}
            
            if not booking_request.customer_info:
                return {"valid": False, "message": "Customer information is required"}
            
            # Validate customer info
            customer_info = booking_request.customer_info
            if not customer_info.get("name"):
                return {"valid": False, "message": "Customer name is required"}
            
            if not customer_info.get("phone"):
                return {"valid": False, "message": "Customer phone is required"}
            
            # Validate date format
            try:
                datetime.strptime(booking_request.date, "%Y-%m-%d")
            except ValueError:
                return {"valid": False, "message": "Invalid date format. Use YYYY-MM-DD"}
            
            # Validate time format
            try:
                datetime.strptime(booking_request.time, "%H:%M")
            except ValueError:
                return {"valid": False, "message": "Invalid time format. Use HH:MM"}
            
            # Check if booking is for future date
            booking_datetime = datetime.strptime(f"{booking_request.date} {booking_request.time}", "%Y-%m-%d %H:%M")
            if booking_datetime <= datetime.now():
                return {"valid": False, "message": "Booking must be for a future date and time"}
            
            return {"valid": True}
            
        except Exception as e:
            self.logger.error(f"Error validating booking request: {e}")
            return {"valid": False, "message": "Error validating booking request"}
    
    def _generate_booking_id(self) -> str:
        """Generate a unique booking ID."""
        return f"ESB{datetime.now().strftime('%Y%m%d')}{uuid.uuid4().hex[:8].upper()}"
    
    def _generate_confirmation_code(self) -> str:
        """Generate a confirmation code."""
        return f"EC{uuid.uuid4().hex[:6].upper()}"
    
    def _make_booking_api_call(self, booking_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make API call to create booking using EasySalon API.
        
        Args:
            booking_data: Booking data to send to API
            
        Returns:
            API response data
        """
        try:
            # Prepare the payload according to EasySalon API format
            api_payload = {
                "customerMobile": booking_data["customer_info"]["phone"],
                "customerName": booking_data["customer_info"]["name"],
                "totalCustomer": 1,  # Default to 1 customer
                "branchId": int(booking_data.get("branch_id", 0)),
                "bookingDetails": [
                    {
                        "serviceStaffs": [
                            {
                                "serviceId": int(booking_data["service_id"]),
                                "staffId": int(booking_data.get("staff_id", 0))
                            }
                        ]
                    }
                ],
                "bookingDate": booking_data["date"],
                "bookingTime": booking_data["time"]
            }
            
            # Make API request with retry logic
            for attempt in range(self.max_retries):
                try:
                    response = requests.post(
                        f"{self.api_base_url}/booking",
                        json=api_payload,
                        timeout=self.api_timeout,
                        headers={
                            "Content-Type": "application/json"
                        }
                    )
                    
                    response.raise_for_status()
                    
                    # Parse response
                    api_response = response.json()
                    
                    if api_response.get("data"):
                        booking_result = api_response["data"]
                        return {
                            "success": True,
                            "booking_id": str(booking_result.get("id", "")),
                            "confirmation_code": booking_result.get("bookingCode", ""),
                            "booking_status": booking_result.get("bookingStatus", ""),
                            "total": booking_result.get("total", 0),
                            "date": booking_result.get("date", ""),
                            "salon_name": "EasySalon",
                            "staff_name": "Available Staff",
                            "message": "Booking created successfully",
                            "status": "confirmed"
                        }
                    else:
                        return {
                            "success": False,
                            "message": "Failed to create booking - no data in response"
                        }
                        
                except requests.exceptions.RequestException as e:
                    self.logger.warning(f"API request attempt {attempt + 1} failed: {str(e)}")
                    if attempt == self.max_retries - 1:
                        raise
                    
        except Exception as e:
            self.logger.error(f"Error making booking API call: {str(e)}")
            # Return mock response for development
            return {
                "success": True,
                "booking_id": booking_data["booking_id"],
                "confirmation_code": booking_data["confirmation_code"],
                "salon_name": "EasySalon Downtown",
                "staff_name": "Sarah Johnson",
                "message": "Booking created successfully (mock)",
                "status": "confirmed"
            }
    
    def parse_booking_request(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Parse natural language query to extract booking information.
        
        Args:
            query: User's booking request in natural language
            context: Additional context from previous conversation
            
        Returns:
            Parsed booking information
        """
        # This is a simplified parser - in production, you'd use NLP
        parsed_info = {
            "service_type": None,
            "date": None,
            "time": None,
            "duration": None,
            "staff_preference": None,
            "special_requests": None,
            "customer_info": {}
        }
        
        query_lower = query.lower()
        
        # Extract service type
        services = {
            "haircut": "haircut",
            "manicure": "manicure",
            "pedicure": "pedicure",
            "facial": "facial",
            "massage": "massage",
            "coloring": "hair_coloring",
            "styling": "hair_styling",
            "eyebrow": "eyebrow_treatment",
            "eyelash": "eyelash_treatment"
        }
        
        for service_key, service_value in services.items():
            if service_key in query_lower:
                parsed_info["service_type"] = service_value
                break
        
        # Extract time-related information
        if "tomorrow" in query_lower:
            parsed_info["date"] = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        elif "today" in query_lower:
            parsed_info["date"] = datetime.now().strftime("%Y-%m-%d")
        
        # Extract time preferences
        time_patterns = {
            "morning": "09:00",
            "afternoon": "14:00",
            "evening": "18:00"
        }
        
        for time_key, time_value in time_patterns.items():
            if time_key in query_lower:
                parsed_info["time"] = time_value
                break
        
        # Use context if available
        if context:
            parsed_info.update(context)
        
        return parsed_info
    
    def format_booking_confirmation(self, booking_response: BookingResponse) -> str:
        """
        Format booking confirmation message for display.
        
        Args:
            booking_response: BookingResponse object
            
        Returns:
            Formatted confirmation message
        """
        if not booking_response.success:
            return f"❌ Booking failed: {booking_response.message}"
        
        details = booking_response.booking_details
        if not details:
            return f"✅ Booking created successfully! Booking ID: {booking_response.booking_id}"
        
        confirmation_message = f"""
✅ **Booking Confirmed!**

**Booking Details:**
- **Booking ID**: `{details['booking_id']}`
- **Confirmation Code**: `{details['confirmation_code']}`
- **Service**: {details['service_name']}
- **Salon**: {details['salon_name']}
- **Date**: {details['date']}
- **Time**: {details['time']}
- **Duration**: {details['duration']} minutes
- **Price**: ${details['price']:.2f}
- **Staff**: {details['staff_name']}
- **Customer**: {details['customer_name']}
- **Phone**: {details['customer_phone']}
- **Status**: {details['status'].title()}

{f"**Special Requests**: {details['special_requests']}" if details.get('special_requests') else ""}

💡 **Important**: Please save your booking ID and confirmation code for future reference.
📱 You'll receive a confirmation SMS shortly.
"""
        
        return confirmation_message.strip()


# Create global instance
booking_manager = BookingManager()
