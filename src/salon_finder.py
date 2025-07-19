"""
Salon Finder Module for EasySalon Chatbot
Handles salon location search and information retrieval functionality.
"""

import json
import logging
import requests
import re
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

from src import global_vars


@dataclass
class SalonInfo:
    """Represents salon information."""
    salon_id: str
    name: str
    address: str
    city: str
    state: str
    zip_code: str
    phone: str
    email: Optional[str] = None
    website: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    rating: Optional[float] = None
    review_count: Optional[int] = None
    hours: Optional[Dict[str, str]] = None
    services: Optional[List[str]] = None
    amenities: Optional[List[str]] = None
    staff_count: Optional[int] = None
    image_url: Optional[str] = None
    description: Optional[str] = None
    distance: Optional[float] = None  # in miles


@dataclass
class LocationQuery:
    """Represents a location-based search query."""
    address: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zip_code: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    radius: Optional[int] = None  # in miles
    max_results: Optional[int] = None


class SalonFinder:
    """
    Handles salon location search and information retrieval for EasySalon services.
    Integrates with EasySalon API and location services to find nearby salons.
    """
    
    def __init__(self):
        """Initialize the salon finder."""
        self.logger = logging.getLogger(__name__)
        self.api_base_url = "https://www.beautysalon.vn"  # Mock API base URL
        self.api_timeout = 30
        self.max_retries = 3
        self._salons_cache = {}
        
    def find_nearby_salons(self, location_query: LocationQuery) -> List[SalonInfo]:
        """
        Find nearby salons based on location criteria.
        
        Args:
            location_query: LocationQuery object with search parameters
            
        Returns:
            List of SalonInfo objects
        """
        try:
            # Generate cache key
            cache_key = self._generate_cache_key(location_query)
            
            # Check cache first
            if cache_key in self._salons_cache:
                return self._salons_cache[cache_key]
            
            # Make API call
            api_response = self._make_salon_search_api_call(location_query)
            
            if api_response.get("success", False):
                salons_data = api_response.get("salons", [])
                salons = []
                
                for salon_data in salons_data:
                    salon_info = SalonInfo(
                        salon_id=salon_data.get("salon_id"),
                        name=salon_data.get("name"),
                        address=salon_data.get("address"),
                        city=salon_data.get("city"),
                        state=salon_data.get("state"),
                        zip_code=salon_data.get("zip_code"),
                        phone=salon_data.get("phone"),
                        email=salon_data.get("email"),
                        website=salon_data.get("website"),
                        latitude=salon_data.get("latitude"),
                        longitude=salon_data.get("longitude"),
                        rating=salon_data.get("rating"),
                        review_count=salon_data.get("review_count"),
                        hours=salon_data.get("hours"),
                        services=salon_data.get("services", []),
                        amenities=salon_data.get("amenities", []),
                        staff_count=salon_data.get("staff_count"),
                        image_url=salon_data.get("image_url"),
                        description=salon_data.get("description"),
                        distance=salon_data.get("distance")
                    )
                    salons.append(salon_info)
                
                # Cache the results
                self._salons_cache[cache_key] = salons
                return salons
            else:
                return []
                
        except Exception as e:
            self.logger.error(f"Error finding nearby salons: {e}")
            return []
    
    def search_salons_by_name(self, name: str, location: Optional[str] = None) -> List[SalonInfo]:
        """
        Search salons by name.
        
        Args:
            name: Salon name to search for
            location: Optional location to filter results
            
        Returns:
            List of SalonInfo objects
        """
        try:
            # Make API call
            api_response = self._make_salon_name_search_api_call(name, location)
            
            if api_response.get("success", False):
                salons_data = api_response.get("salons", [])
                salons = []
                
                for salon_data in salons_data:
                    salon_info = SalonInfo(
                        salon_id=salon_data.get("salon_id"),
                        name=salon_data.get("name"),
                        address=salon_data.get("address"),
                        city=salon_data.get("city"),
                        state=salon_data.get("state"),
                        zip_code=salon_data.get("zip_code"),
                        phone=salon_data.get("phone"),
                        email=salon_data.get("email"),
                        website=salon_data.get("website"),
                        latitude=salon_data.get("latitude"),
                        longitude=salon_data.get("longitude"),
                        rating=salon_data.get("rating"),
                        review_count=salon_data.get("review_count"),
                        hours=salon_data.get("hours"),
                        services=salon_data.get("services", []),
                        amenities=salon_data.get("amenities", []),
                        staff_count=salon_data.get("staff_count"),
                        image_url=salon_data.get("image_url"),
                        description=salon_data.get("description"),
                        distance=salon_data.get("distance")
                    )
                    salons.append(salon_info)
                
                return salons
            else:
                return []
                
        except Exception as e:
            self.logger.error(f"Error searching salons by name: {e}")
            return []
    
    def get_salon_by_id(self, salon_id: str) -> Optional[SalonInfo]:
        """
        Get salon information by ID.
        
        Args:
            salon_id: Salon ID
            
        Returns:
            SalonInfo object or None
        """
        try:
            # Make API call
            api_response = self._make_salon_details_api_call(salon_id)
            
            if api_response.get("success", False):
                salon_data = api_response.get("salon", {})
                
                salon_info = SalonInfo(
                    salon_id=salon_data.get("salon_id"),
                    name=salon_data.get("name"),
                    address=salon_data.get("address"),
                    city=salon_data.get("city"),
                    state=salon_data.get("state"),
                    zip_code=salon_data.get("zip_code"),
                    phone=salon_data.get("phone"),
                    email=salon_data.get("email"),
                    website=salon_data.get("website"),
                    latitude=salon_data.get("latitude"),
                    longitude=salon_data.get("longitude"),
                    rating=salon_data.get("rating"),
                    review_count=salon_data.get("review_count"),
                    hours=salon_data.get("hours"),
                    services=salon_data.get("services", []),
                    amenities=salon_data.get("amenities", []),
                    staff_count=salon_data.get("staff_count"),
                    image_url=salon_data.get("image_url"),
                    description=salon_data.get("description"),
                    distance=salon_data.get("distance")
                )
                
                return salon_info
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"Error getting salon by ID: {e}")
            return None
    
    def get_top_rated_salons(self, location: Optional[str] = None, limit: int = 10) -> List[SalonInfo]:
        """
        Get top-rated salons.
        
        Args:
            location: Optional location to filter results
            limit: Maximum number of results
            
        Returns:
            List of SalonInfo objects
        """
        try:
            # Make API call
            api_response = self._make_top_rated_salons_api_call(location, limit)
            
            if api_response.get("success", False):
                salons_data = api_response.get("salons", [])
                salons = []
                
                for salon_data in salons_data:
                    salon_info = SalonInfo(
                        salon_id=salon_data.get("salon_id"),
                        name=salon_data.get("name"),
                        address=salon_data.get("address"),
                        city=salon_data.get("city"),
                        state=salon_data.get("state"),
                        zip_code=salon_data.get("zip_code"),
                        phone=salon_data.get("phone"),
                        email=salon_data.get("email"),
                        website=salon_data.get("website"),
                        latitude=salon_data.get("latitude"),
                        longitude=salon_data.get("longitude"),
                        rating=salon_data.get("rating"),
                        review_count=salon_data.get("review_count"),
                        hours=salon_data.get("hours"),
                        services=salon_data.get("services", []),
                        amenities=salon_data.get("amenities", []),
                        staff_count=salon_data.get("staff_count"),
                        image_url=salon_data.get("image_url"),
                        description=salon_data.get("description"),
                        distance=salon_data.get("distance")
                    )
                    salons.append(salon_info)
                
                return salons
            else:
                return []
                
        except Exception as e:
            self.logger.error(f"Error getting top-rated salons: {e}")
            return []
    
    def _generate_cache_key(self, location_query: LocationQuery) -> str:
        """Generate cache key for location query."""
        key_parts = [
            location_query.address or "",
            location_query.city or "",
            location_query.state or "",
            location_query.zip_code or "",
            str(location_query.latitude or ""),
            str(location_query.longitude or ""),
            str(location_query.radius or ""),
            str(location_query.max_results or "")
        ]
        return "|".join(key_parts)
    
    def _make_salon_search_api_call(self, location_query: LocationQuery) -> Dict[str, Any]:
        """
        Make API call to search salons by location (mock implementation).
        
        Args:
            location_query: LocationQuery object
            
        Returns:
            API response data
        """
        # Mock API response for development
        return {
            "success": True,
            "salons": [
                {
                    "salon_id": "SAL001",
                    "name": "EasySalon Downtown",
                    "address": "123 Main St",
                    "city": "San Francisco",
                    "state": "CA",
                    "zip_code": "94102",
                    "phone": "+1 (555) 123-4567",
                    "email": "downtown@easysalon.com",
                    "website": "https://easysalon.com/downtown",
                    "latitude": 37.7749,
                    "longitude": -122.4194,
                    "rating": 4.8,
                    "review_count": 245,
                    "hours": {
                        "Monday": "9:00 AM - 8:00 PM",
                        "Tuesday": "9:00 AM - 8:00 PM",
                        "Wednesday": "9:00 AM - 8:00 PM",
                        "Thursday": "9:00 AM - 8:00 PM",
                        "Friday": "9:00 AM - 9:00 PM",
                        "Saturday": "8:00 AM - 9:00 PM",
                        "Sunday": "10:00 AM - 6:00 PM"
                    },
                    "services": ["Haircut", "Manicure", "Pedicure", "Facial", "Massage"],
                    "amenities": ["WiFi", "Parking", "Refreshments", "Wheelchair Accessible"],
                    "staff_count": 12,
                    "image_url": "/images/salon-downtown.jpg",
                    "description": "Premier beauty salon in downtown San Francisco",
                    "distance": 0.8
                },
                {
                    "salon_id": "SAL002",
                    "name": "Beauty Oasis Spa",
                    "address": "456 Oak Ave",
                    "city": "San Francisco",
                    "state": "CA",
                    "zip_code": "94103",
                    "phone": "+1 (555) 987-6543",
                    "email": "info@beautyoasis.com",
                    "website": "https://beautyoasis.com",
                    "latitude": 37.7849,
                    "longitude": -122.4094,
                    "rating": 4.6,
                    "review_count": 189,
                    "hours": {
                        "Monday": "10:00 AM - 7:00 PM",
                        "Tuesday": "10:00 AM - 7:00 PM",
                        "Wednesday": "10:00 AM - 7:00 PM",
                        "Thursday": "10:00 AM - 7:00 PM",
                        "Friday": "10:00 AM - 8:00 PM",
                        "Saturday": "9:00 AM - 8:00 PM",
                        "Sunday": "11:00 AM - 5:00 PM"
                    },
                    "services": ["Facial", "Massage", "Eyebrow Threading", "Waxing"],
                    "amenities": ["WiFi", "Quiet Environment", "Organic Products"],
                    "staff_count": 8,
                    "image_url": "/images/beauty-oasis.jpg",
                    "description": "Relaxing spa environment with organic treatments",
                    "distance": 1.2
                }
            ]
        }
    
    def _make_salon_name_search_api_call(self, name: str, location: Optional[str]) -> Dict[str, Any]:
        """
        Make API call to search salons by name (mock implementation).
        
        Args:
            name: Salon name
            location: Optional location filter
            
        Returns:
            API response data
        """
        # Mock API response for development
        return {
            "success": True,
            "salons": [
                {
                    "salon_id": "SAL001",
                    "name": "EasySalon Downtown",
                    "address": "123 Main St",
                    "city": "San Francisco",
                    "state": "CA",
                    "zip_code": "94102",
                    "phone": "+1 (555) 123-4567",
                    "email": "downtown@easysalon.com",
                    "website": "https://easysalon.com/downtown",
                    "latitude": 37.7749,
                    "longitude": -122.4194,
                    "rating": 4.8,
                    "review_count": 245,
                    "hours": {
                        "Monday": "9:00 AM - 8:00 PM",
                        "Tuesday": "9:00 AM - 8:00 PM",
                        "Wednesday": "9:00 AM - 8:00 PM",
                        "Thursday": "9:00 AM - 8:00 PM",
                        "Friday": "9:00 AM - 9:00 PM",
                        "Saturday": "8:00 AM - 9:00 PM",
                        "Sunday": "10:00 AM - 6:00 PM"
                    },
                    "services": ["Haircut", "Manicure", "Pedicure", "Facial", "Massage"],
                    "amenities": ["WiFi", "Parking", "Refreshments", "Wheelchair Accessible"],
                    "staff_count": 12,
                    "image_url": "/images/salon-downtown.jpg",
                    "description": "Premier beauty salon in downtown San Francisco",
                    "distance": None
                }
            ]
        }
    
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
                "address": "123 Main St",
                "city": "San Francisco",
                "state": "CA",
                "zip_code": "94102",
                "phone": "+1 (555) 123-4567",
                "email": "downtown@easysalon.com",
                "website": "https://easysalon.com/downtown",
                "latitude": 37.7749,
                "longitude": -122.4194,
                "rating": 4.8,
                "review_count": 245,
                "hours": {
                    "Monday": "9:00 AM - 8:00 PM",
                    "Tuesday": "9:00 AM - 8:00 PM",
                    "Wednesday": "9:00 AM - 8:00 PM",
                    "Thursday": "9:00 AM - 8:00 PM",
                    "Friday": "9:00 AM - 9:00 PM",
                    "Saturday": "8:00 AM - 9:00 PM",
                    "Sunday": "10:00 AM - 6:00 PM"
                },
                "services": ["Haircut", "Manicure", "Pedicure", "Facial", "Massage"],
                "amenities": ["WiFi", "Parking", "Refreshments", "Wheelchair Accessible"],
                "staff_count": 12,
                "image_url": "/images/salon-downtown.jpg",
                "description": "Premier beauty salon in downtown San Francisco",
                "distance": None
            }
        }
    
    def _make_top_rated_salons_api_call(self, location: Optional[str], limit: int) -> Dict[str, Any]:
        """
        Make API call to get top-rated salons (mock implementation).
        
        Args:
            location: Optional location filter
            limit: Maximum number of results
            
        Returns:
            API response data
        """
        # Mock API response for development
        return {
            "success": True,
            "salons": [
                {
                    "salon_id": "SAL001",
                    "name": "EasySalon Downtown",
                    "address": "123 Main St",
                    "city": "San Francisco",
                    "state": "CA",
                    "zip_code": "94102",
                    "phone": "+1 (555) 123-4567",
                    "email": "downtown@easysalon.com",
                    "website": "https://easysalon.com/downtown",
                    "latitude": 37.7749,
                    "longitude": -122.4194,
                    "rating": 4.8,
                    "review_count": 245,
                    "hours": {
                        "Monday": "9:00 AM - 8:00 PM",
                        "Tuesday": "9:00 AM - 8:00 PM",
                        "Wednesday": "9:00 AM - 8:00 PM",
                        "Thursday": "9:00 AM - 8:00 PM",
                        "Friday": "9:00 AM - 9:00 PM",
                        "Saturday": "8:00 AM - 9:00 PM",
                        "Sunday": "10:00 AM - 6:00 PM"
                    },
                    "services": ["Haircut", "Manicure", "Pedicure", "Facial", "Massage"],
                    "amenities": ["WiFi", "Parking", "Refreshments", "Wheelchair Accessible"],
                    "staff_count": 12,
                    "image_url": "/images/salon-downtown.jpg",
                    "description": "Premier beauty salon in downtown San Francisco",
                    "distance": None
                }
            ]
        }
    
    def format_salon_info(self, salon: SalonInfo) -> str:
        """
        Format salon information for display.
        
        Args:
            salon: SalonInfo object
            
        Returns:
            Formatted salon information
        """
        rating_stars = "⭐" * int(salon.rating or 0)
        
        formatted_info = f"""
🏪 **{salon.name}**
📍 {salon.address}, {salon.city}, {salon.state} {salon.zip_code}
📞 {salon.phone}
{f"🌐 {salon.website}" if salon.website else ""}
{f"📧 {salon.email}" if salon.email else ""}

{f"⭐ {salon.rating}/5.0 ({salon.review_count} reviews)" if salon.rating else ""}
{f"📏 {salon.distance:.1f} miles away" if salon.distance else ""}

**Services:** {', '.join(salon.services) if salon.services else 'N/A'}
**Amenities:** {', '.join(salon.amenities) if salon.amenities else 'N/A'}
{f"👥 Staff: {salon.staff_count} professionals" if salon.staff_count else ""}

{f"**Description:** {salon.description}" if salon.description else ""}

**Hours:**
{self._format_hours(salon.hours) if salon.hours else "Contact for hours"}

**Salon ID:** `{salon.salon_id}`
"""
        
        return formatted_info.strip()
    
    def format_salons_list(self, salons: List[SalonInfo]) -> str:
        """
        Format list of salons for display.
        
        Args:
            salons: List of SalonInfo objects
            
        Returns:
            Formatted salons list
        """
        if not salons:
            return "No salons found."
        
        formatted_list = f"**Found {len(salons)} salon(s):**\n\n"
        
        for i, salon in enumerate(salons, 1):
            rating_stars = "⭐" * int(salon.rating or 0)
            
            formatted_list += f"""
{i}. 🏪 **{salon.name}**
📍 {salon.address}, {salon.city}, {salon.state}
📞 {salon.phone}
{f"⭐ {salon.rating}/5.0 ({salon.review_count} reviews)" if salon.rating else ""}
{f"📏 {salon.distance:.1f} miles away" if salon.distance else ""}
**Services:** {', '.join(salon.services[:3]) if salon.services else 'N/A'}{'...' if salon.services and len(salon.services) > 3 else ''}

"""
        
        return formatted_list.strip()
    
    def _format_hours(self, hours: Dict[str, str]) -> str:
        """Format hours dictionary for display."""
        formatted_hours = ""
        for day, time in hours.items():
            formatted_hours += f"• {day}: {time}\n"
        return formatted_hours.strip()
    
    def parse_location_query(self, query: str) -> LocationQuery:
        """
        Parse natural language query to extract location information.
        
        Args:
            query: User's location search query
            
        Returns:
            LocationQuery object
        """
        location_query = LocationQuery()
        query_lower = query.lower()
        
        # Extract city names (common patterns)
        city_patterns = [
            r'in\s+([a-zA-Z\s]+)',
            r'near\s+([a-zA-Z\s]+)',
            r'around\s+([a-zA-Z\s]+)',
            r'close\s+to\s+([a-zA-Z\s]+)'
        ]
        
        for pattern in city_patterns:
            match = re.search(pattern, query_lower)
            if match:
                location_query.city = match.group(1).strip()
                break
        
        # Extract zip codes
        zip_match = re.search(r'\b\d{5}\b', query)
        if zip_match:
            location_query.zip_code = zip_match.group()
        
        # Extract radius
        radius_patterns = [
            r'within\s+(\d+)\s+miles?',
            r'(\d+)\s+miles?\s+radius',
            r'(\d+)\s+miles?\s+away'
        ]
        
        for pattern in radius_patterns:
            match = re.search(pattern, query_lower)
            if match:
                location_query.radius = int(match.group(1))
                break
        
        # Default radius if not specified
        if not location_query.radius:
            location_query.radius = 10
        
        # Extract max results
        if "top" in query_lower:
            match = re.search(r'top\s+(\d+)', query_lower)
            if match:
                location_query.max_results = int(match.group(1))
        
        return location_query


# Create global instance
salon_finder = SalonFinder()
