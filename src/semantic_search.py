"""
Semantic Search Module for EasySalon Chatbot
Handles intelligent search and recommendations functionality.
"""

import json
import logging
import requests
import re
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import numpy as np

from src import global_vars
from src import qdrant_db as db


@dataclass
class SearchResult:
    """Represents a search result."""
    content: str
    score: float
    metadata: Dict[str, Any]
    category: str
    source: str


@dataclass
class SearchQuery:
    """Represents a search query."""
    query: str
    category: Optional[str] = None
    filters: Optional[Dict[str, Any]] = None
    limit: int = 10
    min_score: float = 0.7


@dataclass
class Recommendation:
    """Represents a recommendation."""
    title: str
    description: str
    category: str
    relevance_score: float
    metadata: Dict[str, Any]
    action_type: str  # 'service', 'booking', 'info', 'consultation'


class SemanticSearchEngine:
    """
    Handles semantic search and intelligent recommendations for EasySalon services.
    Uses vector similarity search to provide relevant results and recommendations.
    """
    
    def __init__(self):
        """Initialize the semantic search engine."""
        self.logger = logging.getLogger(__name__)
        self.vectorstores = {}
        self.search_history = []
        
    def initialize_vectorstores(self, vectorstores: Dict[str, Any]):
        """Initialize vector stores for semantic search."""
        self.vectorstores = vectorstores
        
    def semantic_search(self, search_query: SearchQuery) -> List[SearchResult]:
        """
        Perform semantic search across all available data sources.
        
        Args:
            search_query: SearchQuery object with search parameters
            
        Returns:
            List of SearchResult objects
        """
        try:
            all_results = []
            
            # Search across different data sources
            search_sources = [
                ("services", "service"),
                ("salon_info", "salon"),
                ("products", "product"),
                ("packages", "package"),
                ("staff", "staff"),
                ("branches", "branch"),
                ("treatments", "treatment"),
                ("beauty_tips", "tip")
            ]
            
            for source_name, category in search_sources:
                if source_name in self.vectorstores:
                    results = self._search_vectorstore(
                        self.vectorstores[source_name], 
                        search_query.query, 
                        search_query.limit // len(search_sources) + 1
                    )
                    
                    for result in results:
                        if result.get("score", 0) >= search_query.min_score:
                            search_result = SearchResult(
                                content=result.get("content", ""),
                                score=result.get("score", 0),
                                metadata=result.get("metadata", {}),
                                category=category,
                                source=source_name
                            )
                            all_results.append(search_result)
            
            # Sort by relevance score
            all_results.sort(key=lambda x: x.score, reverse=True)
            
            # Apply category filter if specified
            if search_query.category:
                all_results = [r for r in all_results if r.category == search_query.category]
            
            # Apply additional filters
            if search_query.filters:
                all_results = self._apply_filters(all_results, search_query.filters)
            
            # Store search history
            self.search_history.append({
                "query": search_query.query,
                "category": search_query.category,
                "timestamp": datetime.now().isoformat(),
                "results_count": len(all_results)
            })
            
            return all_results[:search_query.limit]
            
        except Exception as e:
            self.logger.error(f"Error in semantic search: {e}")
            return []
    
    def get_recommendations(self, context: Dict[str, Any]) -> List[Recommendation]:
        """
        Get intelligent recommendations based on context.
        
        Args:
            context: Context information including user preferences, search history, etc.
            
        Returns:
            List of Recommendation objects
        """
        try:
            recommendations = []
            
            # Get context-based recommendations
            if "user_query" in context:
                query_recommendations = self._get_query_based_recommendations(context["user_query"])
                recommendations.extend(query_recommendations)
            
            # Get history-based recommendations
            if self.search_history:
                history_recommendations = self._get_history_based_recommendations()
                recommendations.extend(history_recommendations)
            
            # Get popular service recommendations
            popular_recommendations = self._get_popular_recommendations()
            recommendations.extend(popular_recommendations)
            
            # Get seasonal recommendations
            seasonal_recommendations = self._get_seasonal_recommendations()
            recommendations.extend(seasonal_recommendations)
            
            # Remove duplicates and sort by relevance
            unique_recommendations = self._deduplicate_recommendations(recommendations)
            unique_recommendations.sort(key=lambda x: x.relevance_score, reverse=True)
            
            return unique_recommendations[:10]
            
        except Exception as e:
            self.logger.error(f"Error getting recommendations: {e}")
            return []
    
    def _search_vectorstore(self, vectorstore: Any, query: str, limit: int) -> List[Dict[str, Any]]:
        """Search a specific vectorstore."""
        try:
            # Use the existing db.query function
            results = db.query(vectorstore, query, limit)
            
            # Convert to expected format
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "content": result.get("content", str(result)),
                    "score": result.get("score", 0.8),  # Default score if not available
                    "metadata": result.get("metadata", {})
                })
            
            return formatted_results
            
        except Exception as e:
            self.logger.error(f"Error searching vectorstore: {e}")
            return []
    
    def _apply_filters(self, results: List[SearchResult], filters: Dict[str, Any]) -> List[SearchResult]:
        """Apply filters to search results."""
        filtered_results = results
        
        # Price range filter
        if "price_range" in filters:
            price_range = filters["price_range"]
            filtered_results = [r for r in filtered_results 
                             if self._matches_price_range(r.metadata, price_range)]
        
        # Duration filter
        if "duration" in filters:
            duration_range = filters["duration"]
            filtered_results = [r for r in filtered_results 
                             if self._matches_duration_range(r.metadata, duration_range)]
        
        # Location filter
        if "location" in filters:
            location = filters["location"]
            filtered_results = [r for r in filtered_results 
                             if self._matches_location(r.metadata, location)]
        
        return filtered_results
    
    def _matches_price_range(self, metadata: Dict[str, Any], price_range: str) -> bool:
        """Check if metadata matches price range."""
        try:
            price = metadata.get("price", 0)
            if price_range == "budget":
                return price < 50
            elif price_range == "moderate":
                return 50 <= price < 100
            elif price_range == "premium":
                return price >= 100
            return True
        except:
            return True
    
    def _matches_duration_range(self, metadata: Dict[str, Any], duration_range: str) -> bool:
        """Check if metadata matches duration range."""
        try:
            duration = metadata.get("duration", 0)
            if duration_range == "short":
                return duration < 45
            elif duration_range == "medium":
                return 45 <= duration < 90
            elif duration_range == "long":
                return duration >= 90
            return True
        except:
            return True
    
    def _matches_location(self, metadata: Dict[str, Any], location: str) -> bool:
        """Check if metadata matches location."""
        try:
            metadata_location = metadata.get("location", "").lower()
            return location.lower() in metadata_location
        except:
            return True
    
    def _get_query_based_recommendations(self, query: str) -> List[Recommendation]:
        """Get recommendations based on current query."""
        recommendations = []
        query_lower = query.lower()
        
        # Service-based recommendations
        if any(keyword in query_lower for keyword in ["haircut", "hair", "style"]):
            recommendations.append(Recommendation(
                title="Hair Styling Package",
                description="Complete hair transformation with cut, color, and style",
                category="service",
                relevance_score=0.9,
                metadata={"price": 120, "duration": 180},
                action_type="service"
            ))
        
        if any(keyword in query_lower for keyword in ["facial", "skin", "acne"]):
            recommendations.append(Recommendation(
                title="Deep Cleansing Facial",
                description="Professional facial treatment for healthy, glowing skin",
                category="service",
                relevance_score=0.85,
                metadata={"price": 80, "duration": 75},
                action_type="service"
            ))
        
        if any(keyword in query_lower for keyword in ["massage", "relax", "stress"]):
            recommendations.append(Recommendation(
                title="Relaxation Massage",
                description="Full-body massage therapy for stress relief and wellness",
                category="service",
                relevance_score=0.8,
                metadata={"price": 90, "duration": 60},
                action_type="service"
            ))
        
        # Booking-based recommendations
        if any(keyword in query_lower for keyword in ["book", "appointment", "schedule"]):
            recommendations.append(Recommendation(
                title="Quick Booking",
                description="Book your appointment in just a few clicks",
                category="booking",
                relevance_score=0.95,
                metadata={},
                action_type="booking"
            ))
        
        # Consultation-based recommendations
        if any(keyword in query_lower for keyword in ["advice", "help", "recommend", "consultation"]):
            recommendations.append(Recommendation(
                title="Beauty Consultation",
                description="Get personalized beauty advice from our experts",
                category="consultation",
                relevance_score=0.9,
                metadata={"price": 40, "duration": 30},
                action_type="consultation"
            ))
        
        return recommendations
    
    def _get_history_based_recommendations(self) -> List[Recommendation]:
        """Get recommendations based on search history."""
        recommendations = []
        
        if not self.search_history:
            return recommendations
        
        # Analyze recent searches
        recent_searches = self.search_history[-5:]  # Last 5 searches
        
        # Count query themes
        theme_counts = {}
        for search in recent_searches:
            query = search["query"].lower()
            if any(keyword in query for keyword in ["hair", "haircut", "style"]):
                theme_counts["hair"] = theme_counts.get("hair", 0) + 1
            if any(keyword in query for keyword in ["skin", "facial", "acne"]):
                theme_counts["skin"] = theme_counts.get("skin", 0) + 1
            if any(keyword in query for keyword in ["nail", "manicure", "pedicure"]):
                theme_counts["nail"] = theme_counts.get("nail", 0) + 1
        
        # Generate recommendations based on frequent themes
        for theme, count in theme_counts.items():
            if count >= 2:  # Theme appeared at least twice
                if theme == "hair":
                    recommendations.append(Recommendation(
                        title="Hair Care Package",
                        description="Based on your interest in hair services",
                        category="service",
                        relevance_score=0.7,
                        metadata={"price": 95, "duration": 90},
                        action_type="service"
                    ))
                elif theme == "skin":
                    recommendations.append(Recommendation(
                        title="Skin Care Treatment",
                        description="Specialized treatment for your skin concerns",
                        category="service",
                        relevance_score=0.7,
                        metadata={"price": 85, "duration": 75},
                        action_type="service"
                    ))
                elif theme == "nail":
                    recommendations.append(Recommendation(
                        title="Nail Care Package",
                        description="Complete nail care and beautification",
                        category="service",
                        relevance_score=0.7,
                        metadata={"price": 55, "duration": 60},
                        action_type="service"
                    ))
        
        return recommendations
    
    def _get_popular_recommendations(self) -> List[Recommendation]:
        """Get popular service recommendations."""
        return [
            Recommendation(
                title="Most Popular: Classic Haircut",
                description="Our most requested hair service",
                category="service",
                relevance_score=0.6,
                metadata={"price": 45, "duration": 60},
                action_type="service"
            ),
            Recommendation(
                title="Customer Favorite: Manicure",
                description="Perfect nail care and polish",
                category="service",
                relevance_score=0.6,
                metadata={"price": 35, "duration": 45},
                action_type="service"
            ),
            Recommendation(
                title="Trending: Facial Treatment",
                description="Rejuvenating facial for healthy skin",
                category="service",
                relevance_score=0.6,
                metadata={"price": 75, "duration": 90},
                action_type="service"
            )
        ]
    
    def _get_seasonal_recommendations(self) -> List[Recommendation]:
        """Get seasonal recommendations."""
        current_month = datetime.now().month
        
        # Winter recommendations (Dec, Jan, Feb)
        if current_month in [12, 1, 2]:
            return [
                Recommendation(
                    title="Winter Skin Care",
                    description="Hydrating treatments for dry winter skin",
                    category="service",
                    relevance_score=0.5,
                    metadata={"price": 80, "duration": 75},
                    action_type="service"
                )
            ]
        
        # Spring recommendations (Mar, Apr, May)
        elif current_month in [3, 4, 5]:
            return [
                Recommendation(
                    title="Spring Refresh",
                    description="Refresh your look for the new season",
                    category="service",
                    relevance_score=0.5,
                    metadata={"price": 70, "duration": 90},
                    action_type="service"
                )
            ]
        
        # Summer recommendations (Jun, Jul, Aug)
        elif current_month in [6, 7, 8]:
            return [
                Recommendation(
                    title="Summer Glow",
                    description="Treatments for a healthy summer glow",
                    category="service",
                    relevance_score=0.5,
                    metadata={"price": 85, "duration": 75},
                    action_type="service"
                )
            ]
        
        # Fall recommendations (Sep, Oct, Nov)
        else:
            return [
                Recommendation(
                    title="Fall Renewal",
                    description="Prepare your skin and hair for fall",
                    category="service",
                    relevance_score=0.5,
                    metadata={"price": 75, "duration": 80},
                    action_type="service"
                )
            ]
    
    def _deduplicate_recommendations(self, recommendations: List[Recommendation]) -> List[Recommendation]:
        """Remove duplicate recommendations."""
        seen = set()
        unique_recommendations = []
        
        for rec in recommendations:
            key = (rec.title, rec.category)
            if key not in seen:
                seen.add(key)
                unique_recommendations.append(rec)
        
        return unique_recommendations
    
    def parse_search_query(self, query: str) -> SearchQuery:
        """Parse natural language query into SearchQuery object."""
        search_query = SearchQuery(query=query)
        query_lower = query.lower()
        
        # Extract category
        if any(keyword in query_lower for keyword in ["service", "treatment", "hair", "facial", "nail"]):
            search_query.category = "service"
        elif any(keyword in query_lower for keyword in ["salon", "location", "address"]):
            search_query.category = "salon"
        elif any(keyword in query_lower for keyword in ["product", "buy", "purchase"]):
            search_query.category = "product"
        elif any(keyword in query_lower for keyword in ["package", "deal", "bundle"]):
            search_query.category = "package"
        
        # Extract filters
        filters = {}
        
        # Price filters
        if any(keyword in query_lower for keyword in ["cheap", "budget", "affordable"]):
            filters["price_range"] = "budget"
        elif any(keyword in query_lower for keyword in ["expensive", "premium", "luxury"]):
            filters["price_range"] = "premium"
        elif any(keyword in query_lower for keyword in ["moderate", "mid-range"]):
            filters["price_range"] = "moderate"
        
        # Duration filters
        if any(keyword in query_lower for keyword in ["quick", "fast", "short"]):
            filters["duration"] = "short"
        elif any(keyword in query_lower for keyword in ["long", "extended", "comprehensive"]):
            filters["duration"] = "long"
        
        # Location filters
        location_match = re.search(r'near\s+([a-zA-Z\s]+)', query_lower)
        if location_match:
            filters["location"] = location_match.group(1).strip()
        
        search_query.filters = filters if filters else None
        
        return search_query
    
    def format_search_results(self, results: List[SearchResult]) -> str:
        """Format search results for display."""
        if not results:
            return "No results found for your search."
        
        formatted_results = f"🔍 **Found {len(results)} result(s):**\n\n"
        
        for i, result in enumerate(results, 1):
            category_emoji = {
                "service": "💇‍♀️",
                "salon": "🏪",
                "product": "🛍️",
                "package": "📦",
                "staff": "👩‍💼",
                "branch": "🏢",
                "treatment": "💆‍♀️",
                "tip": "💡"
            }
            
            emoji = category_emoji.get(result.category, "📋")
            
            formatted_results += f"{emoji} **{i}. {result.category.title()}**\n"
            formatted_results += f"📝 {result.content[:200]}{'...' if len(result.content) > 200 else ''}\n"
            formatted_results += f"🎯 Relevance: {result.score:.0%}\n\n"
        
        return formatted_results.strip()
    
    def format_recommendations(self, recommendations: List[Recommendation]) -> str:
        """Format recommendations for display."""
        if not recommendations:
            return "No recommendations available at this time."
        
        formatted_recommendations = "🌟 **Recommended for you:**\n\n"
        
        for i, rec in enumerate(recommendations, 1):
            action_emoji = {
                "service": "💇‍♀️",
                "booking": "📅",
                "info": "ℹ️",
                "consultation": "💬"
            }
            
            emoji = action_emoji.get(rec.action_type, "⭐")
            
            formatted_recommendations += f"{emoji} **{rec.title}**\n"
            formatted_recommendations += f"📝 {rec.description}\n"
            
            if rec.metadata.get("price"):
                formatted_recommendations += f"💰 ${rec.metadata['price']:.2f}"
                if rec.metadata.get("duration"):
                    formatted_recommendations += f" • ⏰ {rec.metadata['duration']} min"
                formatted_recommendations += "\n"
            
            formatted_recommendations += f"🎯 Relevance: {rec.relevance_score:.0%}\n\n"
        
        return formatted_recommendations.strip()


# Create global instance
semantic_search_engine = SemanticSearchEngine()
