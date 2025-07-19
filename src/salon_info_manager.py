"""
Salon Information Module for EasySalon Chatbot
Handles comprehensive salon information retrieval and display functionality.
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
class SalonDetails:
    """Represents detailed salon information."""
    salon_id: str
    name: str
    description: str
    address: str
    city: str
    state: str
    zip_code: str
    phone: str
    email: str
    website: str
    hours: Dict[str, str]
    services: List[str]
    amenities: List[str]
    staff_members: List[Dict[str, Any]]
    pricing: Dict[str, float]
    policies: Dict[str, str]
    social_media: Dict[str, str]
    photos: List[str]
    certifications: List[str]
    languages: List[str]
    parking_info: str
    accessibility: List[str]
    payment_methods: List[str]
    cancellation_policy: str
    rating: float
    review_count: int
    established_year: int
    owner_info: Dict[str, Any]


@dataclass
class StaffMember:
    """Represents staff member information."""
    staff_id: str
    name: str
    position: str
    specializations: List[str]
    experience_years: int
    certifications: List[str]
    languages: List[str]
    bio: str
    photo_url: str
    rating: float
    available_days: List[str]
    booking_preference: str


@dataclass
class SalonPolicy:
    """Represents salon policies."""
    policy_type: str
    title: str
    description: str
    effective_date: str
    details: List[str]


class SalonInfoManager:
    """
    Handles comprehensive salon information retrieval and management.
    Provides detailed information about salons, staff, policies, and services.
    """
    
    def __init__(self):
        """Initialize the salon information manager."""
        self.logger = logging.getLogger(__name__)
        self.api_base_url = "https://www.beautysalon.vn"  # Mock API base URL
        self.api_timeout = 30
        self.max_retries = 3
        self._salon_cache = {}
        self._staff_cache = {}
        
    def get_salon_details(self, salon_id: str) -> Optional[SalonDetails]:
        """
        Get comprehensive salon details.
        
        Args:
            salon_id: Salon ID
            
        Returns:
            SalonDetails object or None
        """
        try:
            # Check cache first
            if salon_id in self._salon_cache:
                return self._salon_cache[salon_id]
            
            # Make API call
            api_response = self._make_salon_details_api_call(salon_id)
            
            if api_response.get("success", False):
                salon_data = api_response.get("salon", {})
                
                salon_details = SalonDetails(
                    salon_id=salon_data.get("salon_id"),
                    name=salon_data.get("name"),
                    description=salon_data.get("description"),
                    address=salon_data.get("address"),
                    city=salon_data.get("city"),
                    state=salon_data.get("state"),
                    zip_code=salon_data.get("zip_code"),
                    phone=salon_data.get("phone"),
                    email=salon_data.get("email"),
                    website=salon_data.get("website"),
                    hours=salon_data.get("hours", {}),
                    services=salon_data.get("services", []),
                    amenities=salon_data.get("amenities", []),
                    staff_members=salon_data.get("staff_members", []),
                    pricing=salon_data.get("pricing", {}),
                    policies=salon_data.get("policies", {}),
                    social_media=salon_data.get("social_media", {}),
                    photos=salon_data.get("photos", []),
                    certifications=salon_data.get("certifications", []),
                    languages=salon_data.get("languages", []),
                    parking_info=salon_data.get("parking_info", ""),
                    accessibility=salon_data.get("accessibility", []),
                    payment_methods=salon_data.get("payment_methods", []),
                    cancellation_policy=salon_data.get("cancellation_policy", ""),
                    rating=salon_data.get("rating", 0.0),
                    review_count=salon_data.get("review_count", 0),
                    established_year=salon_data.get("established_year", 0),
                    owner_info=salon_data.get("owner_info", {})
                )
                
                # Cache the result
                self._salon_cache[salon_id] = salon_details
                return salon_details
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"Error getting salon details: {e}")
            return None
    
    def get_staff_information(self, salon_id: str) -> List[StaffMember]:
        """
        Get information about salon staff members.
        
        Args:
            salon_id: Salon ID
            
        Returns:
            List of StaffMember objects
        """
        try:
            # Check cache first
            cache_key = f"staff_{salon_id}"
            if cache_key in self._staff_cache:
                return self._staff_cache[cache_key]
            
            # Make API call
            api_response = self._make_staff_info_api_call(salon_id)
            
            if api_response.get("success", False):
                staff_data = api_response.get("staff", [])
                staff_members = []
                
                for staff_info in staff_data:
                    staff_member = StaffMember(
                        staff_id=staff_info.get("staff_id"),
                        name=staff_info.get("name"),
                        position=staff_info.get("position"),
                        specializations=staff_info.get("specializations", []),
                        experience_years=staff_info.get("experience_years", 0),
                        certifications=staff_info.get("certifications", []),
                        languages=staff_info.get("languages", []),
                        bio=staff_info.get("bio", ""),
                        photo_url=staff_info.get("photo_url", ""),
                        rating=staff_info.get("rating", 0.0),
                        available_days=staff_info.get("available_days", []),
                        booking_preference=staff_info.get("booking_preference", "")
                    )
                    staff_members.append(staff_member)
                
                # Cache the result
                self._staff_cache[cache_key] = staff_members
                return staff_members
            else:
                return []
                
        except Exception as e:
            self.logger.error(f"Error getting staff information: {e}")
            return []
    
    def get_salon_policies(self, salon_id: str) -> List[SalonPolicy]:
        """
        Get salon policies and terms.
        
        Args:
            salon_id: Salon ID
            
        Returns:
            List of SalonPolicy objects
        """
        try:
            # Make API call
            api_response = self._make_policies_api_call(salon_id)
            
            if api_response.get("success", False):
                policies_data = api_response.get("policies", [])
                policies = []
                
                for policy_info in policies_data:
                    policy = SalonPolicy(
                        policy_type=policy_info.get("policy_type"),
                        title=policy_info.get("title"),
                        description=policy_info.get("description"),
                        effective_date=policy_info.get("effective_date"),
                        details=policy_info.get("details", [])
                    )
                    policies.append(policy)
                
                return policies
            else:
                return []
                
        except Exception as e:
            self.logger.error(f"Error getting salon policies: {e}")
            return []
    
    def get_salon_hours(self, salon_id: str) -> Dict[str, str]:
        """
        Get salon operating hours.
        
        Args:
            salon_id: Salon ID
            
        Returns:
            Dictionary of hours by day
        """
        try:
            salon_details = self.get_salon_details(salon_id)
            return salon_details.hours if salon_details else {}
        except Exception as e:
            self.logger.error(f"Error getting salon hours: {e}")
            return {}
    
    def get_salon_contact_info(self, salon_id: str) -> Dict[str, str]:
        """
        Get salon contact information.
        
        Args:
            salon_id: Salon ID
            
        Returns:
            Dictionary of contact information
        """
        try:
            salon_details = self.get_salon_details(salon_id)
            if salon_details:
                return {
                    "phone": salon_details.phone,
                    "email": salon_details.email,
                    "website": salon_details.website,
                    "address": f"{salon_details.address}, {salon_details.city}, {salon_details.state} {salon_details.zip_code}"
                }
            return {}
        except Exception as e:
            self.logger.error(f"Error getting salon contact info: {e}")
            return {}
    
    def get_salon_amenities(self, salon_id: str) -> List[str]:
        """
        Get salon amenities.
        
        Args:
            salon_id: Salon ID
            
        Returns:
            List of amenities
        """
        try:
            salon_details = self.get_salon_details(salon_id)
            return salon_details.amenities if salon_details else []
        except Exception as e:
            self.logger.error(f"Error getting salon amenities: {e}")
            return []
    
    def check_salon_availability(self, salon_id: str, date: str) -> Dict[str, Any]:
        """
        Check if salon is open on a specific date.
        
        Args:
            salon_id: Salon ID
            date: Date in YYYY-MM-DD format
            
        Returns:
            Dictionary with availability information
        """
        try:
            salon_details = self.get_salon_details(salon_id)
            if not salon_details:
                return {"available": False, "reason": "Salon not found"}
            
            # Parse date
            date_obj = datetime.strptime(date, "%Y-%m-%d")
            day_name = date_obj.strftime("%A")
            
            # Check if salon is open on this day
            hours = salon_details.hours.get(day_name, "Closed")
            
            if hours.lower() == "closed":
                return {"available": False, "reason": f"Salon is closed on {day_name}"}
            
            return {
                "available": True,
                "hours": hours,
                "day": day_name,
                "date": date
            }
            
        except Exception as e:
            self.logger.error(f"Error checking salon availability: {e}")
            return {"available": False, "reason": "Error checking availability"}
    
    def _make_salon_details_api_call(self, salon_id: str) -> Dict[str, Any]:
        """
        Make API call to get salon details (mock implementation).
        
        Args:
            salon_id: Salon ID
            
        Returns:
            API response data
        """
        # Mock API response for development
        return {
            "success": True,
            "salon": {
                "salon_id": salon_id,
                "name": "EasySalon Downtown",
                "description": "Premier beauty salon in the heart of downtown, offering comprehensive beauty services in a luxurious, modern environment.",
                "address": "123 Main Street",
                "city": "San Francisco",
                "state": "CA",
                "zip_code": "94102",
                "phone": "+1 (555) 123-4567",
                "email": "info@easysalon.com",
                "website": "https://www.easysalon.com",
                "hours": {
                    "Monday": "9:00 AM - 8:00 PM",
                    "Tuesday": "9:00 AM - 8:00 PM",
                    "Wednesday": "9:00 AM - 8:00 PM",
                    "Thursday": "9:00 AM - 8:00 PM",
                    "Friday": "9:00 AM - 9:00 PM",
                    "Saturday": "8:00 AM - 9:00 PM",
                    "Sunday": "10:00 AM - 6:00 PM"
                },
                "services": [
                    "Haircuts & Styling",
                    "Hair Coloring",
                    "Facial Treatments",
                    "Manicure & Pedicure",
                    "Massage Therapy",
                    "Eyebrow & Eyelash Services",
                    "Waxing Services",
                    "Makeup Application"
                ],
                "amenities": [
                    "Free WiFi",
                    "Complimentary Beverages",
                    "Private Consultation Rooms",
                    "Wheelchair Accessible",
                    "Parking Available",
                    "Air Conditioning",
                    "Relaxation Area",
                    "Changing Rooms"
                ],
                "staff_members": [
                    {
                        "staff_id": "STAFF001",
                        "name": "Sarah Johnson",
                        "position": "Senior Hair Stylist",
                        "specializations": ["Hair Cutting", "Hair Coloring", "Bridal Styling"],
                        "experience_years": 8,
                        "certifications": ["Licensed Cosmetologist", "Color Specialist"],
                        "rating": 4.9
                    },
                    {
                        "staff_id": "STAFF002",
                        "name": "Maria Rodriguez",
                        "position": "Esthetician",
                        "specializations": ["Facial Treatments", "Acne Treatment", "Anti-aging"],
                        "experience_years": 6,
                        "certifications": ["Licensed Esthetician", "Dermalogica Certified"],
                        "rating": 4.8
                    }
                ],
                "pricing": {
                    "Haircut": 45.0,
                    "Hair Color": 85.0,
                    "Facial": 75.0,
                    "Manicure": 35.0,
                    "Pedicure": 45.0,
                    "Massage": 65.0
                },
                "policies": {
                    "cancellation": "24-hour cancellation policy",
                    "payment": "Cash, Credit Cards, and Digital Payments accepted",
                    "late_arrival": "15-minute grace period for late arrivals"
                },
                "social_media": {
                    "facebook": "https://facebook.com/easysalon",
                    "instagram": "https://instagram.com/easysalon",
                    "twitter": "https://twitter.com/easysalon"
                },
                "photos": [
                    "/images/salon-interior-1.jpg",
                    "/images/salon-interior-2.jpg",
                    "/images/salon-exterior.jpg"
                ],
                "certifications": [
                    "State Board Licensed",
                    "Health Department Certified",
                    "Professional Beauty Association Member"
                ],
                "languages": ["English", "Spanish", "Mandarin"],
                "parking_info": "Free parking available in adjacent lot",
                "accessibility": [
                    "Wheelchair accessible entrance",
                    "Accessible restrooms",
                    "Adjustable treatment chairs"
                ],
                "payment_methods": [
                    "Cash", "Credit Cards", "Debit Cards", "Apple Pay", "Google Pay"
                ],
                "cancellation_policy": "Appointments can be cancelled up to 24 hours in advance without charge. Late cancellations may incur a fee.",
                "rating": 4.8,
                "review_count": 245,
                "established_year": 2015,
                "owner_info": {
                    "name": "Jennifer Chen",
                    "experience_years": 15,
                    "background": "Licensed cosmetologist with extensive experience in salon management"
                }
            }
        }
    
    def _make_staff_info_api_call(self, salon_id: str) -> Dict[str, Any]:
        """
        Make API call to get staff information (mock implementation).
        
        Args:
            salon_id: Salon ID
            
        Returns:
            API response data
        """
        # Mock API response for development
        return {
            "success": True,
            "staff": [
                {
                    "staff_id": "STAFF001",
                    "name": "Sarah Johnson",
                    "position": "Senior Hair Stylist",
                    "specializations": ["Hair Cutting", "Hair Coloring", "Bridal Styling"],
                    "experience_years": 8,
                    "certifications": ["Licensed Cosmetologist", "Color Specialist"],
                    "languages": ["English", "Spanish"],
                    "bio": "Sarah has been perfecting her craft for over 8 years, specializing in modern cuts and creative color techniques. She loves helping clients express their personality through their hair.",
                    "photo_url": "/images/staff-sarah.jpg",
                    "rating": 4.9,
                    "available_days": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
                    "booking_preference": "Online booking preferred"
                },
                {
                    "staff_id": "STAFF002",
                    "name": "Maria Rodriguez",
                    "position": "Esthetician",
                    "specializations": ["Facial Treatments", "Acne Treatment", "Anti-aging"],
                    "experience_years": 6,
                    "certifications": ["Licensed Esthetician", "Dermalogica Certified"],
                    "languages": ["English", "Spanish"],
                    "bio": "Maria is passionate about skincare and helping clients achieve their best skin. She stays current with the latest treatments and technologies.",
                    "photo_url": "/images/staff-maria.jpg",
                    "rating": 4.8,
                    "available_days": ["Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"],
                    "booking_preference": "Phone or online booking"
                },
                {
                    "staff_id": "STAFF003",
                    "name": "Lisa Wang",
                    "position": "Nail Technician",
                    "specializations": ["Manicure", "Pedicure", "Nail Art"],
                    "experience_years": 4,
                    "certifications": ["Licensed Nail Technician", "Gel Specialist"],
                    "languages": ["English", "Mandarin"],
                    "bio": "Lisa is skilled in all aspects of nail care and loves creating beautiful nail art designs. She takes pride in maintaining the highest hygiene standards.",
                    "photo_url": "/images/staff-lisa.jpg",
                    "rating": 4.7,
                    "available_days": ["Monday", "Wednesday", "Thursday", "Friday", "Saturday"],
                    "booking_preference": "Walk-ins welcome"
                }
            ]
        }
    
    def _make_policies_api_call(self, salon_id: str) -> Dict[str, Any]:
        """
        Make API call to get salon policies (mock implementation).
        
        Args:
            salon_id: Salon ID
            
        Returns:
            API response data
        """
        # Mock API response for development
        return {
            "success": True,
            "policies": [
                {
                    "policy_type": "cancellation",
                    "title": "Cancellation Policy",
                    "description": "Our cancellation policy ensures fair scheduling for all clients",
                    "effective_date": "2024-01-01",
                    "details": [
                        "Cancellations must be made at least 24 hours in advance",
                        "No-shows and late cancellations may incur a 50% service fee",
                        "Emergency cancellations will be considered on a case-by-case basis",
                        "Repeated no-shows may result in booking restrictions"
                    ]
                },
                {
                    "policy_type": "payment",
                    "title": "Payment Policy",
                    "description": "Payment methods and terms",
                    "effective_date": "2024-01-01",
                    "details": [
                        "Payment is due at the time of service",
                        "We accept cash, credit cards, and digital payments",
                        "Gratuity is not included in service prices",
                        "Package deals must be paid in full at time of booking"
                    ]
                },
                {
                    "policy_type": "health_safety",
                    "title": "Health and Safety Policy",
                    "description": "Health and safety guidelines for all clients",
                    "effective_date": "2024-01-01",
                    "details": [
                        "All tools and equipment are sanitized between clients",
                        "Clients with contagious conditions cannot be serviced",
                        "Patch tests are required for certain chemical services",
                        "Pregnant clients should consult with their doctor before certain treatments"
                    ]
                }
            ]
        }
    
    def format_salon_details(self, salon_details: SalonDetails) -> str:
        """
        Format comprehensive salon details for display.
        
        Args:
            salon_details: SalonDetails object
            
        Returns:
            Formatted salon information
        """
        formatted_info = f"""
🏪 **{salon_details.name}**
📍 {salon_details.address}, {salon_details.city}, {salon_details.state} {salon_details.zip_code}

📝 **About Us:**
{salon_details.description}

📞 **Contact Information:**
• Phone: {salon_details.phone}
• Email: {salon_details.email}
• Website: {salon_details.website}

⏰ **Hours:**
{self._format_hours(salon_details.hours)}

💇‍♀️ **Services:**
{self._format_list(salon_details.services)}

🏢 **Amenities:**
{self._format_list(salon_details.amenities)}

👥 **Staff:**
{self._format_staff_summary(salon_details.staff_members)}

💰 **Pricing:**
{self._format_pricing(salon_details.pricing)}

🎓 **Certifications:**
{self._format_list(salon_details.certifications)}

🗣️ **Languages Spoken:**
{self._format_list(salon_details.languages)}

🚗 **Parking:**
{salon_details.parking_info}

♿ **Accessibility:**
{self._format_list(salon_details.accessibility)}

💳 **Payment Methods:**
{self._format_list(salon_details.payment_methods)}

⭐ **Rating:** {salon_details.rating}/5.0 ({salon_details.review_count} reviews)
📅 **Established:** {salon_details.established_year}

📱 **Social Media:**
{self._format_social_media(salon_details.social_media)}
"""
        
        return formatted_info.strip()
    
    def format_staff_information(self, staff_members: List[StaffMember]) -> str:
        """
        Format staff information for display.
        
        Args:
            staff_members: List of StaffMember objects
            
        Returns:
            Formatted staff information
        """
        if not staff_members:
            return "No staff information available."
        
        formatted_info = f"👥 **Our Professional Team ({len(staff_members)} members):**\n\n"
        
        for i, staff in enumerate(staff_members, 1):
            formatted_info += f"""
**{i}. {staff.name}**
🏷️ Position: {staff.position}
⭐ Rating: {staff.rating}/5.0
📅 Experience: {staff.experience_years} years

🎯 **Specializations:**
{self._format_list(staff.specializations)}

🎓 **Certifications:**
{self._format_list(staff.certifications)}

🗣️ **Languages:**
{self._format_list(staff.languages)}

📅 **Available Days:**
{self._format_list(staff.available_days)}

📝 **About {staff.name.split()[0]}:**
{staff.bio}

📞 **Booking:** {staff.booking_preference}

"""
        
        return formatted_info.strip()
    
    def format_salon_policies(self, policies: List[SalonPolicy]) -> str:
        """
        Format salon policies for display.
        
        Args:
            policies: List of SalonPolicy objects
            
        Returns:
            Formatted policies information
        """
        if not policies:
            return "No policy information available."
        
        formatted_info = "📋 **Salon Policies:**\n\n"
        
        for i, policy in enumerate(policies, 1):
            formatted_info += f"""
**{i}. {policy.title}**
📝 {policy.description}
📅 Effective: {policy.effective_date}

**Details:**
{self._format_list(policy.details)}

"""
        
        return formatted_info.strip()
    
    def _format_hours(self, hours: Dict[str, str]) -> str:
        """Format hours dictionary for display."""
        if not hours:
            return "Hours not available"
        
        formatted_hours = ""
        for day, time in hours.items():
            formatted_hours += f"• {day}: {time}\n"
        return formatted_hours.strip()
    
    def _format_list(self, items: List[str]) -> str:
        """Format list of items for display."""
        if not items:
            return "None available"
        
        return "\n".join(f"• {item}" for item in items)
    
    def _format_staff_summary(self, staff_members: List[Dict[str, Any]]) -> str:
        """Format staff summary for display."""
        if not staff_members:
            return "Staff information not available"
        
        summary = ""
        for staff in staff_members:
            name = staff.get("name", "Unknown")
            position = staff.get("position", "Staff Member")
            rating = staff.get("rating", 0)
            summary += f"• {name} - {position} (⭐ {rating}/5.0)\n"
        
        return summary.strip()
    
    def _format_pricing(self, pricing: Dict[str, float]) -> str:
        """Format pricing information for display."""
        if not pricing:
            return "Pricing information not available"
        
        formatted_pricing = ""
        for service, price in pricing.items():
            formatted_pricing += f"• {service}: ${price:.2f}\n"
        
        return formatted_pricing.strip()
    
    def _format_social_media(self, social_media: Dict[str, str]) -> str:
        """Format social media links for display."""
        if not social_media:
            return "Social media links not available"
        
        formatted_social = ""
        for platform, url in social_media.items():
            formatted_social += f"• {platform.title()}: {url}\n"
        
        return formatted_social.strip()
    
    def parse_salon_info_query(self, query: str) -> Dict[str, Any]:
        """
        Parse natural language query to extract salon information request.
        
        Args:
            query: User's salon information query
            
        Returns:
            Parsed query information
        """
        parsed_info = {
            "info_type": "general",
            "specific_request": None,
            "salon_identifier": None
        }
        
        query_lower = query.lower()
        
        # Determine information type requested
        if any(keyword in query_lower for keyword in ["hours", "open", "close", "time"]):
            parsed_info["info_type"] = "hours"
        elif any(keyword in query_lower for keyword in ["contact", "phone", "email", "address"]):
            parsed_info["info_type"] = "contact"
        elif any(keyword in query_lower for keyword in ["staff", "employee", "stylist", "technician"]):
            parsed_info["info_type"] = "staff"
        elif any(keyword in query_lower for keyword in ["policy", "rule", "cancel", "payment"]):
            parsed_info["info_type"] = "policies"
        elif any(keyword in query_lower for keyword in ["amenity", "facility", "feature"]):
            parsed_info["info_type"] = "amenities"
        elif any(keyword in query_lower for keyword in ["price", "cost", "fee", "charge"]):
            parsed_info["info_type"] = "pricing"
        elif any(keyword in query_lower for keyword in ["service", "treatment", "offer"]):
            parsed_info["info_type"] = "services"
        
        # Extract salon identifier
        salon_match = re.search(r'salon\s+(\w+)', query_lower)
        if salon_match:
            parsed_info["salon_identifier"] = salon_match.group(1)
        
        return parsed_info


# Create global instance
salon_info_manager = SalonInfoManager()
