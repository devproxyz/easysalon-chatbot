"""
Service Browser Module for EasySalon Chatbot
Handles service and pricing information retrieval functionality.
"""

import json
import logging
import requests
import re
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

from src import global_vars


@dataclass
class ServiceInfo:
    """Represents service information."""
    service_id: str
    name: str
    description: str
    duration: int  # in minutes
    price: float
    category: str
    staff_requirements: Optional[str] = None
    equipment_needed: Optional[str] = None
    preparation_notes: Optional[str] = None
    aftercare_tips: Optional[str] = None
    image_url: Optional[str] = None
    popularity_score: Optional[int] = None


@dataclass
class ServiceCategory:
    """Represents service category."""
    category_id: str
    name: str
    description: str
    services: List[ServiceInfo]
    image_url: Optional[str] = None


@dataclass
class PricingInfo:
    """Represents pricing information."""
    base_price: float
    senior_discount: Optional[float] = None
    student_discount: Optional[float] = None
    package_deals: Optional[List[Dict[str, Any]]] = None
    seasonal_promotions: Optional[List[Dict[str, Any]]] = None


class ServiceBrowser:
    """
    Handles service and pricing information retrieval for EasySalon services.
    Integrates with EasySalon API to fetch service details and pricing.
    """
    
    def __init__(self):
        """Initialize the service browser."""
        self.logger = logging.getLogger(__name__)
        self.api_base_url = "https://www.beautysalon.vn"  # Mock API base URL
        self.api_timeout = 30
        self.max_retries = 3
        self._services_cache = {}
        self._categories_cache = {}
        
    def get_all_services(self) -> List[ServiceInfo]:
        """
        Get all available services.
        
        Returns:
            List of ServiceInfo objects
        """
        try:
            # Check cache first
            if "all_services" in self._services_cache:
                return self._services_cache["all_services"]
            
            # Make API call
            api_response = self._make_services_api_call()
            
            if api_response.get("success", False):
                services_data = api_response.get("services", [])
                services = []
                
                for service_data in services_data:
                    service_info = ServiceInfo(
                        service_id=service_data.get("service_id"),
                        name=service_data.get("name"),
                        description=service_data.get("description"),
                        duration=service_data.get("duration"),
                        price=service_data.get("price"),
                        category=service_data.get("category"),
                        staff_requirements=service_data.get("staff_requirements"),
                        equipment_needed=service_data.get("equipment_needed"),
                        preparation_notes=service_data.get("preparation_notes"),
                        aftercare_tips=service_data.get("aftercare_tips"),
                        image_url=service_data.get("image_url"),
                        popularity_score=service_data.get("popularity_score")
                    )
                    services.append(service_info)
                
                # Cache the results
                self._services_cache["all_services"] = services
                return services
            else:
                return []
                
        except Exception as e:
            self.logger.error(f"Error getting all services: {e}")
            return []
    
    def get_services_by_category(self, category: str) -> List[ServiceInfo]:
        """
        Get services by category.
        
        Args:
            category: Service category name
            
        Returns:
            List of ServiceInfo objects
        """
        try:
            all_services = self.get_all_services()
            return [service for service in all_services if service.category.lower() == category.lower()]
        except Exception as e:
            self.logger.error(f"Error getting services by category: {e}")
            return []
    
    def search_services(self, query: str) -> List[ServiceInfo]:
        """
        Search services by name, description, or category.
        
        Args:
            query: Search query
            
        Returns:
            List of ServiceInfo objects
        """
        try:
            all_services = self.get_all_services()
            query_lower = query.lower()
            
            matching_services = []
            for service in all_services:
                # Check if query matches name, description, or category
                if (query_lower in service.name.lower() or 
                    query_lower in service.description.lower() or 
                    query_lower in service.category.lower()):
                    matching_services.append(service)
            
            # Sort by popularity score if available
            matching_services.sort(key=lambda x: x.popularity_score or 0, reverse=True)
            
            return matching_services
            
        except Exception as e:
            self.logger.error(f"Error searching services: {e}")
            return []
    
    def get_service_categories(self) -> List[ServiceCategory]:
        """
        Get all service categories.
        
        Returns:
            List of ServiceCategory objects
        """
        try:
            # Check cache first
            if "categories" in self._categories_cache:
                return self._categories_cache["categories"]
            
            # Make API call
            api_response = self._make_categories_api_call()
            
            if api_response.get("success", False):
                categories_data = api_response.get("categories", [])
                categories = []
                
                for category_data in categories_data:
                    # Get services for this category
                    category_services = self.get_services_by_category(category_data.get("name"))
                    
                    category = ServiceCategory(
                        category_id=category_data.get("category_id"),
                        name=category_data.get("name"),
                        description=category_data.get("description"),
                        services=category_services,
                        image_url=category_data.get("image_url")
                    )
                    categories.append(category)
                
                # Cache the results
                self._categories_cache["categories"] = categories
                return categories
            else:
                return []
                
        except Exception as e:
            self.logger.error(f"Error getting service categories: {e}")
            return []
    
    def get_service_by_id(self, service_id: str) -> Optional[ServiceInfo]:
        """
        Get service by ID.
        
        Args:
            service_id: Service ID
            
        Returns:
            ServiceInfo object or None
        """
        try:
            all_services = self.get_all_services()
            for service in all_services:
                if service.service_id == service_id:
                    return service
            return None
        except Exception as e:
            self.logger.error(f"Error getting service by ID: {e}")
            return None
    
    def get_pricing_info(self, service_id: str) -> Optional[PricingInfo]:
        """
        Get detailed pricing information for a service.
        
        Args:
            service_id: Service ID
            
        Returns:
            PricingInfo object or None
        """
        try:
            # Make API call
            api_response = self._make_pricing_api_call(service_id)
            
            if api_response.get("success", False):
                pricing_data = api_response.get("pricing", {})
                
                pricing_info = PricingInfo(
                    base_price=pricing_data.get("base_price"),
                    senior_discount=pricing_data.get("senior_discount"),
                    student_discount=pricing_data.get("student_discount"),
                    package_deals=pricing_data.get("package_deals"),
                    seasonal_promotions=pricing_data.get("seasonal_promotions")
                )
                
                return pricing_info
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"Error getting pricing info: {e}")
            return None
    
    def get_popular_services(self, limit: int = 10) -> List[ServiceInfo]:
        """
        Get popular services.
        
        Args:
            limit: Maximum number of services to return
            
        Returns:
            List of ServiceInfo objects
        """
        try:
            all_services = self.get_all_services()
            # Sort by popularity score
            popular_services = sorted(all_services, key=lambda x: x.popularity_score or 0, reverse=True)
            return popular_services[:limit]
        except Exception as e:
            self.logger.error(f"Error getting popular services: {e}")
            return []
    
    def _make_services_api_call(self) -> Dict[str, Any]:
        """
        Make API call to get all services (mock implementation).
        
        Returns:
            API response data
        """
        # Mock API response for development
        return {
            "success": True,
            "services": [
                {
                    "service_id": "SVC001",
                    "name": "Classic Haircut",
                    "description": "Professional haircut with wash and style",
                    "duration": 60,
                    "price": 45.00,
                    "category": "Hair Services",
                    "staff_requirements": "Licensed cosmetologist",
                    "equipment_needed": "Scissors, clippers, styling tools",
                    "preparation_notes": "Clean hair preferred",
                    "aftercare_tips": "Use provided styling products",
                    "image_url": "/images/haircut.jpg",
                    "popularity_score": 95
                },
                {
                    "service_id": "SVC002",
                    "name": "Manicure",
                    "description": "Complete nail care with polish",
                    "duration": 45,
                    "price": 35.00,
                    "category": "Nail Services",
                    "staff_requirements": "Nail technician",
                    "equipment_needed": "Nail tools, polish, UV lamp",
                    "preparation_notes": "Remove old polish",
                    "aftercare_tips": "Avoid water for 1 hour",
                    "image_url": "/images/manicure.jpg",
                    "popularity_score": 88
                },
                {
                    "service_id": "SVC003",
                    "name": "Facial Treatment",
                    "description": "Deep cleansing and moisturizing facial",
                    "duration": 90,
                    "price": 75.00,
                    "category": "Skincare Services",
                    "staff_requirements": "Licensed esthetician",
                    "equipment_needed": "Facial steamer, products",
                    "preparation_notes": "No makeup preferred",
                    "aftercare_tips": "Avoid sun exposure",
                    "image_url": "/images/facial.jpg",
                    "popularity_score": 82
                },
                {
                    "service_id": "SVC004",
                    "name": "Hair Coloring",
                    "description": "Professional hair color service",
                    "duration": 120,
                    "price": 85.00,
                    "category": "Hair Services",
                    "staff_requirements": "Senior colorist",
                    "equipment_needed": "Color products, tools",
                    "preparation_notes": "Skin patch test required",
                    "aftercare_tips": "Use color-safe shampoo",
                    "image_url": "/images/coloring.jpg",
                    "popularity_score": 78
                },
                {
                    "service_id": "SVC005",
                    "name": "Massage Therapy",
                    "description": "Relaxing full-body massage",
                    "duration": 60,
                    "price": 65.00,
                    "category": "Wellness Services",
                    "staff_requirements": "Licensed massage therapist",
                    "equipment_needed": "Massage table, oils",
                    "preparation_notes": "Comfortable clothing",
                    "aftercare_tips": "Drink plenty of water",
                    "image_url": "/images/massage.jpg",
                    "popularity_score": 75
                }
            ]
        }
    
    def _make_categories_api_call(self) -> Dict[str, Any]:
        """
        Make API call to get service categories (mock implementation).
        
        Returns:
            API response data
        """
        # Mock API response for development
        return {
            "success": True,
            "categories": [
                {
                    "category_id": "CAT001",
                    "name": "Hair Services",
                    "description": "Professional hair care and styling services",
                    "image_url": "/images/hair-category.jpg"
                },
                {
                    "category_id": "CAT002",
                    "name": "Nail Services",
                    "description": "Complete nail care and nail art services",
                    "image_url": "/images/nail-category.jpg"
                },
                {
                    "category_id": "CAT003",
                    "name": "Skincare Services",
                    "description": "Facial treatments and skin care services",
                    "image_url": "/images/skincare-category.jpg"
                },
                {
                    "category_id": "CAT004",
                    "name": "Wellness Services",
                    "description": "Massage and wellness treatments",
                    "image_url": "/images/wellness-category.jpg"
                }
            ]
        }
    
    def _make_pricing_api_call(self, service_id: str) -> Dict[str, Any]:
        """
        Make API call to get pricing information (mock implementation).
        
        Args:
            service_id: Service ID
            
        Returns:
            API response data
        """
        # Mock API response for development
        return {
            "success": True,
            "pricing": {
                "base_price": 45.00,
                "senior_discount": 0.15,
                "student_discount": 0.10,
                "package_deals": [
                    {
                        "name": "Hair Care Package",
                        "description": "Cut + Color + Style",
                        "price": 120.00,
                        "savings": 15.00
                    }
                ],
                "seasonal_promotions": [
                    {
                        "name": "Winter Special",
                        "description": "20% off all services",
                        "discount": 0.20,
                        "valid_until": "2024-12-31"
                    }
                ]
            }
        }
    
    def format_service_info(self, service: ServiceInfo) -> str:
        """
        Format service information for display.
        
        Args:
            service: ServiceInfo object
            
        Returns:
            Formatted service information
        """
        formatted_info = f"""
💇‍♀️ **{service.name}**
📂 *{service.category}*

**Description:** {service.description}
**Duration:** {service.duration} minutes
**Price:** ${service.price:.2f}

{f"**Staff Requirements:** {service.staff_requirements}" if service.staff_requirements else ""}
{f"**Preparation Notes:** {service.preparation_notes}" if service.preparation_notes else ""}
{f"**Aftercare Tips:** {service.aftercare_tips}" if service.aftercare_tips else ""}

**Service ID:** `{service.service_id}`
"""
        
        return formatted_info.strip()
    
    def format_services_list(self, services: List[ServiceInfo]) -> str:
        """
        Format list of services for display.
        
        Args:
            services: List of ServiceInfo objects
            
        Returns:
            Formatted services list
        """
        if not services:
            return "No services found."
        
        formatted_list = f"**Found {len(services)} service(s):**\n\n"
        
        for service in services:
            formatted_list += f"""
💇‍♀️ **{service.name}**
📂 *{service.category}*
💰 ${service.price:.2f} • ⏰ {service.duration} min
📝 {service.description}

"""
        
        return formatted_list.strip()
    
    def format_categories_list(self, categories: List[ServiceCategory]) -> str:
        """
        Format list of service categories for display.
        
        Args:
            categories: List of ServiceCategory objects
            
        Returns:
            Formatted categories list
        """
        if not categories:
            return "No service categories found."
        
        formatted_list = "**Service Categories:**\n\n"
        
        for category in categories:
            service_count = len(category.services)
            formatted_list += f"""
📂 **{category.name}**
📝 {category.description}
🔢 {service_count} service(s) available

"""
        
        return formatted_list.strip()
    
    def parse_service_query(self, query: str) -> Dict[str, Any]:
        """
        Parse natural language query to extract service search parameters.
        
        Args:
            query: User's service search query
            
        Returns:
            Parsed search parameters
        """
        parsed_info = {
            "service_type": None,
            "category": None,
            "price_range": None,
            "duration_preference": None
        }
        
        query_lower = query.lower()
        
        # Service type keywords
        service_keywords = {
            "haircut": "haircut",
            "manicure": "manicure",
            "pedicure": "pedicure",
            "facial": "facial",
            "massage": "massage",
            "coloring": "hair coloring",
            "styling": "hair styling",
            "eyebrow": "eyebrow treatment",
            "eyelash": "eyelash treatment"
        }
        
        for keyword, service_type in service_keywords.items():
            if keyword in query_lower:
                parsed_info["service_type"] = service_type
                break
        
        # Category keywords
        category_keywords = {
            "hair": "Hair Services",
            "nail": "Nail Services",
            "skin": "Skincare Services",
            "wellness": "Wellness Services",
            "massage": "Wellness Services"
        }
        
        for keyword, category in category_keywords.items():
            if keyword in query_lower:
                parsed_info["category"] = category
                break
        
        # Price range keywords
        if "cheap" in query_lower or "budget" in query_lower:
            parsed_info["price_range"] = "low"
        elif "expensive" in query_lower or "premium" in query_lower:
            parsed_info["price_range"] = "high"
        
        # Duration keywords
        if "quick" in query_lower or "fast" in query_lower:
            parsed_info["duration_preference"] = "short"
        elif "long" in query_lower or "extensive" in query_lower:
            parsed_info["duration_preference"] = "long"
        
        return parsed_info


# Create global instance
service_browser = ServiceBrowser()
