"""
Beauty Consultation Module for EasySalon Chatbot
Handles beauty advice and treatment recommendations functionality.
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
class BeautyConsultationRequest:
    """Represents a beauty consultation request."""
    concern: str
    skin_type: Optional[str] = None
    hair_type: Optional[str] = None
    age_range: Optional[str] = None
    previous_treatments: Optional[List[str]] = None
    allergies: Optional[List[str]] = None
    budget_range: Optional[str] = None
    lifestyle_factors: Optional[List[str]] = None
    goals: Optional[List[str]] = None


@dataclass
class BeautyRecommendation:
    """Represents a beauty recommendation."""
    treatment_type: str
    treatment_name: str
    description: str
    benefits: List[str]
    duration: int  # in minutes
    price_range: str
    frequency: str
    aftercare: List[str]
    contraindications: Optional[List[str]] = None
    suitable_for: Optional[List[str]] = None


@dataclass
class ConsultationResponse:
    """Represents a consultation response."""
    recommendations: List[BeautyRecommendation]
    general_advice: List[str]
    product_suggestions: List[str]
    lifestyle_tips: List[str]
    follow_up_questions: List[str]


class BeautyConsultant:
    """
    Handles beauty consultation and advice for EasySalon services.
    Provides personalized recommendations based on user concerns and preferences.
    """
    
    def __init__(self):
        """Initialize the beauty consultant."""
        self.logger = logging.getLogger(__name__)
        self.knowledge_base = self._load_beauty_knowledge_base()
        
    def provide_consultation(self, request: BeautyConsultationRequest) -> ConsultationResponse:
        """
        Provide beauty consultation based on user's concerns and preferences.
        
        Args:
            request: BeautyConsultationRequest object
            
        Returns:
            ConsultationResponse object with recommendations
        """
        try:
            # Analyze the concern and context
            concern_analysis = self._analyze_concern(request.concern)
            
            # Get relevant recommendations
            recommendations = self._get_recommendations(concern_analysis, request)
            
            # Generate general advice
            general_advice = self._generate_general_advice(concern_analysis, request)
            
            # Suggest products
            product_suggestions = self._suggest_products(concern_analysis, request)
            
            # Provide lifestyle tips
            lifestyle_tips = self._generate_lifestyle_tips(concern_analysis, request)
            
            # Generate follow-up questions
            follow_up_questions = self._generate_follow_up_questions(request)
            
            return ConsultationResponse(
                recommendations=recommendations,
                general_advice=general_advice,
                product_suggestions=product_suggestions,
                lifestyle_tips=lifestyle_tips,
                follow_up_questions=follow_up_questions
            )
            
        except Exception as e:
            self.logger.error(f"Error providing consultation: {e}")
            return ConsultationResponse(
                recommendations=[],
                general_advice=["I'm having trouble providing consultation right now. Please try again later."],
                product_suggestions=[],
                lifestyle_tips=[],
                follow_up_questions=[]
            )
    
    def _analyze_concern(self, concern: str) -> Dict[str, Any]:
        """Analyze the user's beauty concern."""
        concern_lower = concern.lower()
        
        analysis = {
            "primary_concern": None,
            "concern_category": None,
            "severity": "moderate",
            "urgency": "normal",
            "keywords": []
        }
        
        # Categorize concerns
        skin_concerns = ["acne", "pimples", "breakouts", "wrinkles", "aging", "dark spots", "pigmentation", "dryness", "oily skin", "sensitive skin", "rosacea", "blackheads", "whiteheads"]
        hair_concerns = ["hair loss", "thinning hair", "dandruff", "oily hair", "dry hair", "frizzy hair", "split ends", "hair color", "grey hair", "hair growth"]
        nail_concerns = ["brittle nails", "nail growth", "cuticles", "nail health", "nail fungus", "nail art", "nail strength"]
        body_concerns = ["cellulite", "stretch marks", "body acne", "dry skin", "keratosis pilaris", "ingrown hairs"]
        
        if any(keyword in concern_lower for keyword in skin_concerns):
            analysis["concern_category"] = "skin"
            analysis["primary_concern"] = next((kw for kw in skin_concerns if kw in concern_lower), "general skin concern")
        elif any(keyword in concern_lower for keyword in hair_concerns):
            analysis["concern_category"] = "hair"
            analysis["primary_concern"] = next((kw for kw in hair_concerns if kw in concern_lower), "general hair concern")
        elif any(keyword in concern_lower for keyword in nail_concerns):
            analysis["concern_category"] = "nails"
            analysis["primary_concern"] = next((kw for kw in nail_concerns if kw in concern_lower), "general nail concern")
        elif any(keyword in concern_lower for keyword in body_concerns):
            analysis["concern_category"] = "body"
            analysis["primary_concern"] = next((kw for kw in body_concerns if kw in concern_lower), "general body concern")
        else:
            analysis["concern_category"] = "general"
            analysis["primary_concern"] = "beauty consultation"
        
        # Determine severity
        if any(word in concern_lower for word in ["severe", "very", "extremely", "terrible", "awful"]):
            analysis["severity"] = "high"
        elif any(word in concern_lower for word in ["mild", "slight", "little", "minor"]):
            analysis["severity"] = "low"
        
        # Determine urgency
        if any(word in concern_lower for word in ["urgent", "asap", "immediately", "emergency"]):
            analysis["urgency"] = "high"
        elif any(word in concern_lower for word in ["when possible", "eventually", "sometime"]):
            analysis["urgency"] = "low"
        
        # Extract keywords
        analysis["keywords"] = [word for word in concern_lower.split() if len(word) > 3]
        
        return analysis
    
    def _get_recommendations(self, concern_analysis: Dict[str, Any], request: BeautyConsultationRequest) -> List[BeautyRecommendation]:
        """Get treatment recommendations based on concern analysis."""
        recommendations = []
        
        category = concern_analysis["concern_category"]
        primary_concern = concern_analysis["primary_concern"]
        
        if category == "skin":
            if "acne" in primary_concern:
                recommendations.extend(self._get_acne_recommendations(request))
            elif "aging" in primary_concern or "wrinkles" in primary_concern:
                recommendations.extend(self._get_anti_aging_recommendations(request))
            elif "dark spots" in primary_concern or "pigmentation" in primary_concern:
                recommendations.extend(self._get_pigmentation_recommendations(request))
            elif "dryness" in primary_concern:
                recommendations.extend(self._get_hydration_recommendations(request))
            else:
                recommendations.extend(self._get_general_skin_recommendations(request))
        
        elif category == "hair":
            if "hair loss" in primary_concern:
                recommendations.extend(self._get_hair_loss_recommendations(request))
            elif "dandruff" in primary_concern:
                recommendations.extend(self._get_dandruff_recommendations(request))
            elif "dry hair" in primary_concern:
                recommendations.extend(self._get_hair_hydration_recommendations(request))
            else:
                recommendations.extend(self._get_general_hair_recommendations(request))
        
        elif category == "nails":
            recommendations.extend(self._get_nail_recommendations(request))
        
        elif category == "body":
            recommendations.extend(self._get_body_recommendations(request))
        
        else:
            recommendations.extend(self._get_general_recommendations(request))
        
        return recommendations[:5]  # Limit to top 5 recommendations
    
    def _get_acne_recommendations(self, request: BeautyConsultationRequest) -> List[BeautyRecommendation]:
        """Get acne treatment recommendations."""
        return [
            BeautyRecommendation(
                treatment_type="Facial Treatment",
                treatment_name="Deep Cleansing Acne Facial",
                description="Professional deep cleansing treatment targeting acne-prone skin",
                benefits=["Reduces acne breakouts", "Unclogs pores", "Controls oil production", "Improves skin texture"],
                duration=75,
                price_range="$80-$120",
                frequency="Every 2-3 weeks",
                aftercare=["Avoid touching face", "Use gentle cleanser", "Apply sunscreen daily"],
                contraindications=["Active severe acne", "Recent chemical peels"],
                suitable_for=["Oily skin", "Combination skin", "Mild to moderate acne"]
            ),
            BeautyRecommendation(
                treatment_type="Chemical Peel",
                treatment_name="Salicylic Acid Peel",
                description="Gentle chemical peel to exfoliate and unclog pores",
                benefits=["Reduces acne", "Minimizes pore appearance", "Improves skin tone", "Prevents future breakouts"],
                duration=45,
                price_range="$100-$150",
                frequency="Every 4-6 weeks",
                aftercare=["Avoid sun exposure", "Use gentle skincare", "Moisturize regularly"],
                contraindications=["Pregnancy", "Recent sun exposure", "Sensitive skin"],
                suitable_for=["Non-sensitive skin", "Persistent acne", "Oily skin"]
            )
        ]
    
    def _get_anti_aging_recommendations(self, request: BeautyConsultationRequest) -> List[BeautyRecommendation]:
        """Get anti-aging treatment recommendations."""
        return [
            BeautyRecommendation(
                treatment_type="Facial Treatment",
                treatment_name="Anti-Aging Collagen Facial",
                description="Intensive treatment to boost collagen production and reduce fine lines",
                benefits=["Reduces fine lines", "Improves skin elasticity", "Hydrates skin", "Promotes collagen production"],
                duration=90,
                price_range="$120-$180",
                frequency="Every 3-4 weeks",
                aftercare=["Use retinol products", "Apply anti-aging serum", "Maintain good skincare routine"],
                suitable_for=["Mature skin", "Fine lines", "Loss of elasticity"]
            ),
            BeautyRecommendation(
                treatment_type="Microneedling",
                treatment_name="Collagen Induction Therapy",
                description="Minimally invasive treatment to stimulate natural collagen production",
                benefits=["Reduces wrinkles", "Improves skin texture", "Minimizes pores", "Enhances product absorption"],
                duration=60,
                price_range="$150-$250",
                frequency="Every 4-6 weeks",
                aftercare=["Avoid sun exposure", "Use gentle products", "Apply growth factors"],
                contraindications=["Active acne", "Rosacea", "Blood thinners"],
                suitable_for=["Aging skin", "Acne scars", "Enlarged pores"]
            )
        ]
    
    def _get_pigmentation_recommendations(self, request: BeautyConsultationRequest) -> List[BeautyRecommendation]:
        """Get pigmentation treatment recommendations."""
        return [
            BeautyRecommendation(
                treatment_type="Chemical Peel",
                treatment_name="Vitamin C Brightening Peel",
                description="Lightening treatment to reduce dark spots and even skin tone",
                benefits=["Reduces dark spots", "Brightens complexion", "Evens skin tone", "Prevents future pigmentation"],
                duration=50,
                price_range="$90-$140",
                frequency="Every 3-4 weeks",
                aftercare=["Use sunscreen religiously", "Apply vitamin C serum", "Avoid harsh scrubs"],
                suitable_for=["Hyperpigmentation", "Melasma", "Sun damage"]
            )
        ]
    
    def _get_hydration_recommendations(self, request: BeautyConsultationRequest) -> List[BeautyRecommendation]:
        """Get hydration treatment recommendations."""
        return [
            BeautyRecommendation(
                treatment_type="Facial Treatment",
                treatment_name="Intensive Hydrating Facial",
                description="Deep moisturizing treatment for dry and dehydrated skin",
                benefits=["Deeply hydrates skin", "Restores moisture barrier", "Reduces flakiness", "Improves skin texture"],
                duration=75,
                price_range="$70-$110",
                frequency="Every 2-3 weeks",
                aftercare=["Use hyaluronic acid serum", "Apply rich moisturizer", "Drink plenty of water"],
                suitable_for=["Dry skin", "Dehydrated skin", "Mature skin"]
            )
        ]
    
    def _get_hair_loss_recommendations(self, request: BeautyConsultationRequest) -> List[BeautyRecommendation]:
        """Get hair loss treatment recommendations."""
        return [
            BeautyRecommendation(
                treatment_type="Scalp Treatment",
                treatment_name="Scalp Revitalization Therapy",
                description="Stimulating treatment to promote hair growth and scalp health",
                benefits=["Stimulates hair growth", "Improves scalp circulation", "Strengthens hair follicles", "Reduces hair loss"],
                duration=60,
                price_range="$80-$120",
                frequency="Weekly for 8-12 weeks",
                aftercare=["Use hair growth shampoo", "Massage scalp regularly", "Maintain healthy diet"],
                suitable_for=["Thinning hair", "Hair loss", "Weak hair"]
            )
        ]
    
    def _get_dandruff_recommendations(self, request: BeautyConsultationRequest) -> List[BeautyRecommendation]:
        """Get dandruff treatment recommendations."""
        return [
            BeautyRecommendation(
                treatment_type="Scalp Treatment",
                treatment_name="Anti-Dandruff Scalp Treatment",
                description="Therapeutic treatment to eliminate dandruff and soothe scalp",
                benefits=["Eliminates dandruff", "Soothes irritated scalp", "Reduces itching", "Prevents recurrence"],
                duration=45,
                price_range="$60-$90",
                frequency="Every 1-2 weeks initially",
                aftercare=["Use anti-dandruff shampoo", "Avoid harsh products", "Maintain scalp hygiene"],
                suitable_for=["Dandruff", "Itchy scalp", "Scalp irritation"]
            )
        ]
    
    def _get_hair_hydration_recommendations(self, request: BeautyConsultationRequest) -> List[BeautyRecommendation]:
        """Get hair hydration treatment recommendations."""
        return [
            BeautyRecommendation(
                treatment_type="Hair Treatment",
                treatment_name="Deep Conditioning Hair Mask",
                description="Intensive moisturizing treatment for dry and damaged hair",
                benefits=["Deeply moisturizes hair", "Repairs damage", "Adds shine", "Improves manageability"],
                duration=45,
                price_range="$50-$80",
                frequency="Every 2-3 weeks",
                aftercare=["Use leave-in conditioner", "Avoid heat styling", "Protect from sun"],
                suitable_for=["Dry hair", "Damaged hair", "Chemically treated hair"]
            )
        ]
    
    def _get_general_skin_recommendations(self, request: BeautyConsultationRequest) -> List[BeautyRecommendation]:
        """Get general skin treatment recommendations."""
        return [
            BeautyRecommendation(
                treatment_type="Facial Treatment",
                treatment_name="Customized Facial",
                description="Personalized facial treatment based on your specific skin needs",
                benefits=["Improves skin health", "Addresses specific concerns", "Relaxing experience", "Professional assessment"],
                duration=75,
                price_range="$75-$125",
                frequency="Every 3-4 weeks",
                aftercare=["Follow recommended skincare routine", "Use sunscreen daily", "Stay hydrated"],
                suitable_for=["All skin types", "General skin health", "Maintenance"]
            )
        ]
    
    def _get_general_hair_recommendations(self, request: BeautyConsultationRequest) -> List[BeautyRecommendation]:
        """Get general hair treatment recommendations."""
        return [
            BeautyRecommendation(
                treatment_type="Hair Treatment",
                treatment_name="Hair Health Assessment & Treatment",
                description="Comprehensive hair analysis with customized treatment",
                benefits=["Improves hair health", "Addresses specific concerns", "Professional advice", "Customized approach"],
                duration=60,
                price_range="$60-$100",
                frequency="Every 4-6 weeks",
                aftercare=["Use recommended products", "Protect from heat", "Maintain healthy diet"],
                suitable_for=["All hair types", "General hair health", "Maintenance"]
            )
        ]
    
    def _get_nail_recommendations(self, request: BeautyConsultationRequest) -> List[BeautyRecommendation]:
        """Get nail treatment recommendations."""
        return [
            BeautyRecommendation(
                treatment_type="Nail Treatment",
                treatment_name="Strengthening Nail Treatment",
                description="Professional treatment to improve nail health and strength",
                benefits=["Strengthens nails", "Improves nail growth", "Prevents breakage", "Enhances appearance"],
                duration=30,
                price_range="$40-$60",
                frequency="Every 2-3 weeks",
                aftercare=["Use nail oil daily", "Wear gloves when cleaning", "Avoid harsh chemicals"],
                suitable_for=["Weak nails", "Brittle nails", "Slow nail growth"]
            )
        ]
    
    def _get_body_recommendations(self, request: BeautyConsultationRequest) -> List[BeautyRecommendation]:
        """Get body treatment recommendations."""
        return [
            BeautyRecommendation(
                treatment_type="Body Treatment",
                treatment_name="Body Exfoliation & Moisturizing",
                description="Full body treatment to improve skin texture and hydration",
                benefits=["Removes dead skin", "Improves texture", "Hydrates skin", "Promotes circulation"],
                duration=90,
                price_range="$80-$120",
                frequency="Every 4-6 weeks",
                aftercare=["Moisturize daily", "Use gentle body wash", "Exfoliate weekly"],
                suitable_for=["Dry skin", "Rough texture", "General body care"]
            )
        ]
    
    def _get_general_recommendations(self, request: BeautyConsultationRequest) -> List[BeautyRecommendation]:
        """Get general beauty recommendations."""
        return [
            BeautyRecommendation(
                treatment_type="Consultation",
                treatment_name="Beauty Consultation",
                description="Professional consultation to assess your beauty needs and goals",
                benefits=["Professional assessment", "Personalized advice", "Treatment planning", "Product recommendations"],
                duration=30,
                price_range="$30-$50",
                frequency="As needed",
                aftercare=["Follow professional advice", "Book recommended treatments", "Maintain routine"],
                suitable_for=["All clients", "New clients", "Planning treatments"]
            )
        ]
    
    def _generate_general_advice(self, concern_analysis: Dict[str, Any], request: BeautyConsultationRequest) -> List[str]:
        """Generate general beauty advice."""
        advice = []
        
        category = concern_analysis["concern_category"]
        
        if category == "skin":
            advice.extend([
                "Always use sunscreen with at least SPF 30 daily",
                "Maintain a consistent skincare routine",
                "Stay hydrated by drinking plenty of water",
                "Get adequate sleep for skin repair",
                "Avoid touching your face frequently"
            ])
        elif category == "hair":
            advice.extend([
                "Use heat protectant before styling",
                "Trim hair regularly to prevent split ends",
                "Use a silk or satin pillowcase",
                "Avoid over-washing your hair",
                "Eat a balanced diet rich in vitamins"
            ])
        elif category == "nails":
            advice.extend([
                "Keep nails and cuticles moisturized",
                "Avoid using nails as tools",
                "Wear gloves when cleaning",
                "File nails in one direction",
                "Take breaks from nail polish"
            ])
        else:
            advice.extend([
                "Maintain a healthy, balanced diet",
                "Stay hydrated throughout the day",
                "Get regular exercise",
                "Manage stress levels",
                "Get adequate sleep"
            ])
        
        return advice[:3]  # Limit to top 3 pieces of advice
    
    def _suggest_products(self, concern_analysis: Dict[str, Any], request: BeautyConsultationRequest) -> List[str]:
        """Suggest beauty products."""
        products = []
        
        category = concern_analysis["concern_category"]
        
        if category == "skin":
            products.extend([
                "Gentle cleanser for daily use",
                "Moisturizer suitable for your skin type",
                "Broad-spectrum sunscreen",
                "Vitamin C serum for antioxidant protection",
                "Retinol product for anti-aging (if appropriate)"
            ])
        elif category == "hair":
            products.extend([
                "Sulfate-free shampoo",
                "Deep conditioning mask",
                "Leave-in conditioner",
                "Heat protectant spray",
                "Hair oil for nourishment"
            ])
        elif category == "nails":
            products.extend([
                "Cuticle oil",
                "Nail strengthener",
                "Base coat for protection",
                "Hand cream",
                "Nail file (glass or crystal)"
            ])
        
        return products[:3]  # Limit to top 3 product suggestions
    
    def _generate_lifestyle_tips(self, concern_analysis: Dict[str, Any], request: BeautyConsultationRequest) -> List[str]:
        """Generate lifestyle tips."""
        tips = []
        
        category = concern_analysis["concern_category"]
        
        if category == "skin":
            tips.extend([
                "Eat antioxidant-rich foods like berries and leafy greens",
                "Limit sugar and processed foods",
                "Manage stress through meditation or yoga",
                "Change pillowcases regularly",
                "Avoid excessive alcohol consumption"
            ])
        elif category == "hair":
            tips.extend([
                "Eat protein-rich foods for hair strength",
                "Take biotin supplements (consult doctor first)",
                "Avoid tight hairstyles that pull on hair",
                "Protect hair from chlorine when swimming",
                "Massage scalp regularly to improve circulation"
            ])
        elif category == "nails":
            tips.extend([
                "Eat calcium and protein-rich foods",
                "Avoid biting nails or picking cuticles",
                "Keep hands moisturized",
                "Wear gloves when doing household chores",
                "Don't use acetone-based removers frequently"
            ])
        
        return tips[:3]  # Limit to top 3 lifestyle tips
    
    def _generate_follow_up_questions(self, request: BeautyConsultationRequest) -> List[str]:
        """Generate follow-up questions."""
        questions = []
        
        if not request.skin_type:
            questions.append("What is your skin type (oily, dry, combination, sensitive)?")
        
        if not request.previous_treatments:
            questions.append("Have you had any previous beauty treatments?")
        
        if not request.budget_range:
            questions.append("What is your budget range for treatments?")
        
        if not request.allergies:
            questions.append("Do you have any known allergies or sensitivities?")
        
        questions.append("Would you like to schedule a consultation appointment?")
        
        return questions[:3]  # Limit to top 3 questions
    
    def _load_beauty_knowledge_base(self) -> Dict[str, Any]:
        """Load beauty knowledge base."""
        # This would typically load from a database or file
        return {
            "skin_types": ["oily", "dry", "combination", "sensitive", "normal"],
            "hair_types": ["straight", "wavy", "curly", "coily"],
            "age_ranges": ["teen", "20s", "30s", "40s", "50s", "60+"],
            "common_concerns": ["acne", "aging", "dryness", "sensitivity", "hair loss"],
            "treatment_categories": ["facial", "body", "hair", "nail", "massage"]
        }
    
    def parse_consultation_request(self, query: str) -> BeautyConsultationRequest:
        """Parse natural language query into consultation request."""
        request = BeautyConsultationRequest(concern=query)
        
        query_lower = query.lower()
        
        # Extract skin type
        if "oily skin" in query_lower:
            request.skin_type = "oily"
        elif "dry skin" in query_lower:
            request.skin_type = "dry"
        elif "sensitive skin" in query_lower:
            request.skin_type = "sensitive"
        elif "combination skin" in query_lower:
            request.skin_type = "combination"
        
        # Extract age range
        if any(age in query_lower for age in ["teen", "teenager", "young"]):
            request.age_range = "teen"
        elif any(age in query_lower for age in ["20s", "twenties"]):
            request.age_range = "20s"
        elif any(age in query_lower for age in ["30s", "thirties"]):
            request.age_range = "30s"
        elif any(age in query_lower for age in ["40s", "forties"]):
            request.age_range = "40s"
        elif any(age in query_lower for age in ["50s", "fifties"]):
            request.age_range = "50s"
        elif any(age in query_lower for age in ["60", "sixties", "older"]):
            request.age_range = "60+"
        
        # Extract budget preferences
        if any(budget in query_lower for budget in ["budget", "cheap", "affordable"]):
            request.budget_range = "budget"
        elif any(budget in query_lower for budget in ["premium", "expensive", "luxury"]):
            request.budget_range = "premium"
        else:
            request.budget_range = "moderate"
        
        return request
    
    def format_consultation_response(self, response: ConsultationResponse) -> str:
        """Format consultation response for display."""
        formatted_response = "💄 **Beauty Consultation Results**\n\n"
        
        if response.recommendations:
            formatted_response += "🌟 **Recommended Treatments:**\n"
            for i, rec in enumerate(response.recommendations, 1):
                formatted_response += f"\n**{i}. {rec.treatment_name}**\n"
                formatted_response += f"📋 *{rec.treatment_type}*\n"
                formatted_response += f"📝 {rec.description}\n"
                formatted_response += f"💰 {rec.price_range} • ⏰ {rec.duration} min\n"
                formatted_response += f"🔄 {rec.frequency}\n"
                formatted_response += f"✨ **Benefits:** {', '.join(rec.benefits)}\n"
                if rec.aftercare:
                    formatted_response += f"🏠 **Aftercare:** {', '.join(rec.aftercare)}\n"
                formatted_response += "\n"
        
        if response.general_advice:
            formatted_response += "💡 **General Advice:**\n"
            for advice in response.general_advice:
                formatted_response += f"• {advice}\n"
            formatted_response += "\n"
        
        if response.product_suggestions:
            formatted_response += "🛍️ **Product Suggestions:**\n"
            for product in response.product_suggestions:
                formatted_response += f"• {product}\n"
            formatted_response += "\n"
        
        if response.lifestyle_tips:
            formatted_response += "🌱 **Lifestyle Tips:**\n"
            for tip in response.lifestyle_tips:
                formatted_response += f"• {tip}\n"
            formatted_response += "\n"
        
        if response.follow_up_questions:
            formatted_response += "❓ **Follow-up Questions:**\n"
            for question in response.follow_up_questions:
                formatted_response += f"• {question}\n"
        
        return formatted_response.strip()


# Create global instance
beauty_consultant = BeautyConsultant()
