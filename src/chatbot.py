import os
import json
import requests
import threading
import uuid
import traceback
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Union

from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import Tool
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import LLMChain

from src import global_vars, tts, file_helper
from src import qdrant_db as db
from src import availability_checker
from src import booking_manager
from src import booking_retriever
from src import service_browser
from src import salon_finder
from src import beauty_consultant
from src import semantic_search
from src import salon_info_manager
from src import booking_workflow
from src.api_services import APIService
from src import easysalon

class Chatbot:
    """
    Advanced EasySalon Chatbot using LangChain with ReAct Agent architecture.
    Specialized for beauty salon services, appointment booking, and beauty consultation.
    """
    
    def __init__(self):
        """Initialize the LangChain chatbot with all necessary components."""
        self.user_id = None
        self.is_debug = False
        self.user_responses = []
        self.salon = easysalon.Easysalon(api_key=global_vars.EASYSALON_API_KEY)
        
        # Initialize OpenAI client through LangChain
        self.llm = ChatOpenAI(
            openai_api_key=global_vars.AZURE_OPENAI_API_KEY,
            openai_api_base=global_vars.AZURE_OPENAI_ENDPOINT,
            model_name=global_vars.OPENAI_MODEL,
            temperature=0.7
            # max_tokens=5000
        )
        
        # Initialize database and embedding function
        self.db_client, self.embedding_fn = db.init_db(
            global_vars.AZURE_OPENAI_ENDPOINT,
            global_vars.AZURE_OPENAI_API_EMBEDDED_KEY,
            global_vars.AZURE_OPENAI_EMBEDDING_MODEL,
            global_vars.QDRANT_API_KEY,
            global_vars.QDRANT_HOST
        )
        
        # Initialize vectorstores for different data types
        self.vectorstores = {}
        self._init_vectorstores()
        
        # Initialize memory for conversation history
        self.memory = ConversationBufferWindowMemory(
            k=10,
            return_messages=True,
            memory_key="chat_history"
        )
        
        # Initialize tools and agent
        self.tools = self._create_tools()
        self.agent = self._create_agent()
        
        # Initialize prompt templates
        self.prompt_templates = self._create_prompt_templates()
        
        # Initialize chains
        self.chains = self._create_chains()
        
        # Initialize LangGraph booking workflow
        self.booking_workflow = booking_workflow.BookingWorkflow()
        
        # Track active booking sessions
        self.active_booking_sessions = {}

        # api service instance
        self.api_service = APIService()
        
    def _init_vectorstores(self):
        """Initialize vectorstores for different data types."""
        try:
            # Initialize data and get vectorstores
            run_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.vectorstores = db.init_data(self.db_client, self.embedding_fn, run_path)
            self.debug("Vectorstores initialized successfully")
        except Exception as e:
            self.debug(f"Error initializing vectorstores: {e}")
    
    def _book_appointment_with_session_check(self, query: str) -> str:
        """Create a new appointment booking with session check to prevent conflicts with LangGraph workflow."""
        user_session_id = self.user_id or "default_session"
        
        # Check if user has an active booking session
        if user_session_id in self.active_booking_sessions:
            self.debug("=== PREVENTING TOOL CONFLICT: Active booking session detected ===")
            return ("I see you're already in the process of booking an appointment through our guided workflow. "
                   "Please continue with the current booking process or say 'start over' to restart. "
                   "The guided booking will handle all your appointment details step by step.")
        
        # If no active session, proceed with the original booking tool
        return self._book_appointment(query)
            
    def _create_tools(self) -> List[Tool]:
        """Create tools for the ReAct agent."""
        tools = [
            Tool(
                name="check_availability",
                description="Check appointment availability for beauty salon services. Input should be a query about available time slots, dates, or services.",
                func=self._check_availability
            ),
            Tool(
                name="book_appointment", 
                description="Create a new appointment booking. Only use this if user is NOT in an active booking session. Input should be booking details including service, date, time, and customer information.",
                func=self._book_appointment_with_session_check
            ),
            Tool(
                name="retrieve_booking",
                description="Retrieve existing booking information by ID or confirmation code. Input should be booking ID or confirmation code.",
                func=self._retrieve_booking
            ),
            Tool(
                name="search_services",
                description="Search for beauty services, treatments, and pricing information. Input should be a search query about beauty services.",
                func=self._search_services
            ),
            Tool(
                name="search_salons",
                description="Get salon information and its branches from API",
                func=self._search_salons
            ),
            Tool(
                name="get_salon_info",
                description="Get detailed information about a specific salon including hours, contact info, and staff. Input should be salon name or ID.",
                func=self._get_salon_info
            ),
            Tool(
                name="beauty_consultation",
                description="Provide beauty advice and treatment recommendations. Input should be a beauty-related question or concern.",
                func=self._beauty_consultation
            ),
            Tool(
                name="semantic_search",
                description="Perform intelligent semantic search across all salon data with personalized recommendations. Input should be a search query.",
                func=self._semantic_search
            )
        ]
        return tools
    
    def _create_prompt_templates(self) -> Dict[str, ChatPromptTemplate]:
        """Create optimized prompt templates for different use cases."""
        
        # Main system prompt template
        system_prompt = """You are a friendly and knowledgeable AI Beauty Assistant specialized in beauty salon services.
        
        Your personality:
        - Enthusiastic and helpful about beauty and wellness
        - Knowledgeable about beauty services, treatments, and products
        - Professional yet approachable
        - Focused on helping customers find the right beauty solutions
        
        Your capabilities:
        - Check appointment availability for beauty services
        - Search for beauty salons and spas by location
        - Provide information about beauty services and pricing
        - Offer beauty consultation and advice
        - Help with booking appointments
        - Answer beauty-related questions
        
        Guidelines:
        - Always be professional and helpful
        - Be concise but informative
        - Focus on practical and actionable recommendations
        - Include relevant beauty tips when appropriate
        - Help customers make informed decisions about beauty services
        
        If a user wants to exit, they can type 'exit' or 'quit'.
        """
        
        # Beauty relation check template
        beauty_relation_prompt = ChatPromptTemplate.from_messages([
            ("system", """Determine if the user's question is related to beauty salon services.

            **LANGUAGE CONSISTENCY RULE**: Detect the language of the user's question and respond in the SAME language throughout.
            
            Consider related if it involves:
            - Hair services, haircuts, coloring, styling, treatments
            - Beauty services, facials, skincare, makeup
            - Nail services, manicures, pedicures
            - Spa services, massages, relaxation treatments
            - Appointment booking, availability, scheduling
            - Beauty consultation, advice, recommendations
            - Salon information, location, hours, pricing
            
            **OUTPUT FORMAT**: Respond with JSON only:
            {{
                "is_related": true/false,
                "response": "friendly message if not related (in user's language), empty string if related",
                "detected_language": "English" or "Tiếng Việt"
            }}
            
            **LANGUAGE EXAMPLES**:
            - English question → English response
            - Vietnamese question → Vietnamese response
            - Mixed language → Use predominant language"""),
            ("human", "{question}")
        ])
        
        # Appointment booking detection template
        appointment_request_prompt = ChatPromptTemplate.from_messages([
            ("system", """Determine if the user is requesting to book an appointment.

            **LANGUAGE CONSISTENCY RULE**: Detect the language of the user's question and maintain consistency.
            
            Look for keywords indicating they want to book or schedule:
            - English: "book", "schedule", "appointment", "reserve", "make an appointment"
            - Vietnamese: "đặt lịch", "hẹn", "đặt hẹn", "lên lịch", "thời gian"
            
            **OUTPUT FORMAT**: Respond with JSON only:
            {{
                "is_booking_request": true/false,
                "detected_language": "English" or "Tiếng Việt",
                "confidence": 0.0-1.0
            }}"""),
            ("human", "{question}")
        ])
        
        # Beauty service info extraction template
        service_info_prompt = ChatPromptTemplate.from_messages([
            ("system", """Extract beauty service information from the user's request with language consistency.

            **LANGUAGE CONSISTENCY RULE**: Detect the user's language and respond in the SAME language.
            
            **CURRENT CONTEXT**:
            - Date: {current_date}
            - Weekday: {current_weekday}
            
            **EXTRACTION RULES**:
            1. **service_type** (exact values only):
               - "haircut", "coloring", "styling", "treatment", "facial", "manicure", "pedicure", "massage"
               - Missing: null

            2. **preferred_date** (YYYY-MM-DD format):
               - Explicit dates: "July 15, 2025" → "2025-07-15"
               - Relative dates: "next Monday" → calculate from {current_date}
               - Partial dates: "tomorrow" → calculate from {current_date}
               - Missing: null

            3. **preferred_time** (format: "HH:MM" or "morning", "afternoon", "evening"):
               - "9am" → "09:00", "morning" → "morning"
               - Missing: null

            4. **language**: Auto-detect ("English" or "Tiếng Việt")
            
            **RESPONSE LOGIC**:
            - ALL fields present → enthusiastic confirmation in user's language
            - ANY field missing → specific question for missing info in user's language
            - NOT a service request → friendly redirect in user's language
            
            **OUTPUT FORMAT**: Respond with JSON only:
            {{
                "service_type": "service_name" | null,
                "preferred_date": "YYYY-MM-DD" | null,
                "preferred_time": "HH:MM" | "morning|afternoon|evening" | null,
                "language": "English|Tiếng Việt",
                "prompt": "contextual response message in detected language"
            }}"""),
            ("human", "{question}")
        ])
        
        # Beauty consultation template
        consultation_prompt = ChatPromptTemplate.from_messages([
            ("system", """Provide personalized beauty consultation and recommendations.

            **LANGUAGE CONSISTENCY**: Respond in {language} throughout the entire consultation.
            
            **CONSULTATION DETAILS**:
            - Topic: {topic}
            - User Info: {user_info}
            - Service Type: {service_type}
            
            **STRUCTURE REQUIREMENTS**:
            � **Beauty Consultation: [Topic with Emoji]**
            
            **Personal Assessment**
            - 🎯 **Your Profile**: [Based on user info]
            - 💡 **Recommendations**: [Specific suggestions]
            - ⭐ **Top Picks**: [3 best options]
            
            **Service Suggestions**
            - � **Recommended Services**: [Specific services]
            - ⏰ **Timing**: [Best time for service]
            - 💰 **Budget**: [Expected cost range]
            - � **Maintenance Tips**: [Aftercare advice]
            
            **Style Tips**
            - � **Trending Styles**: [Current trends]
            - 👤 **Face Shape/Skin Type**: [Specific advice]
            - 🎨 **Color Recommendations**: [Best colors]
            
            **CONTENT GUIDELINES**:
            - Provide professional, personalized advice
            - Include practical tips and maintenance
            - Consider skin tone, face shape, lifestyle
            - Suggest complementary services
            - Be encouraging and confidence-building
            
            **TONE**: Professional, encouraging, personalized
            **LANGUAGE**: Maintain {language} consistency throughout"""),
            ("human", "Provide beauty consultation for {topic} with user info: {user_info}")
        ])
        
        # General beauty question template
        beauty_question_prompt = ChatPromptTemplate.from_messages([
            ("system", """Answer beauty-related questions in a friendly, professional way.

            **LANGUAGE CONSISTENCY RULE**: Detect the user's question language and respond in the SAME language throughout.
            
            **RESPONSE GUIDELINES**:
            - Use the provided context to give accurate, helpful information
            - Consider conversation history for better personalized responses
            - Maintain consistent language throughout the response
            - Include practical tips and professional advice
            - Be encouraging and supportive
            - Provide actionable recommendations
            - Focus on beauty, wellness, and self-care
            
            **LANGUAGE DETECTION**:
            - English question → English response
            - Vietnamese question → Vietnamese response
            - Mixed language → Use predominant language
            
            **CONVERSATION HISTORY**: {conversation_context}
            
            **CURRENT CONTEXT**: {context}
            
            Remember: Your response language must match the user's question language. Use conversation history to provide more personalized and relevant beauty recommendations."""),
            ("human", "{question}")
        ])
        
        return {
            "system": system_prompt,
            "beauty_relation": beauty_relation_prompt,
            "appointment_request": appointment_request_prompt,
            "service_info": service_info_prompt,
            "consultation": consultation_prompt,
            "beauty_question": beauty_question_prompt
        }
    
    def _create_chains(self) -> Dict[str, Any]:
        """Create LangChain chains for different operations."""
        chains = {}
        
        try:
            # Beauty relation check chain
            chains["beauty_relation"] = self.prompt_templates["beauty_relation"] | self.llm | StrOutputParser()
            
            # Appointment booking detection chain
            chains["appointment_request"] = self.prompt_templates["appointment_request"] | self.llm | StrOutputParser()
            
            # Beauty service info extraction chain
            chains["service_info"] = self.prompt_templates["service_info"] | self.llm | StrOutputParser()
            
            # Beauty consultation chain
            chains["consultation"] = self.prompt_templates["consultation"] | self.llm | StrOutputParser()
            self.debug("Beauty consultation chain created successfully")
            
            # Beauty question answering chain
            chains["beauty_question"] = self.prompt_templates["beauty_question"] | self.llm | StrOutputParser()
            
            self.debug(f"All chains created successfully: {list(chains.keys())}")
            
        except Exception as e:
            self.debug(f"Error creating chains: {e}")
            import traceback
            self.debug(f"Full traceback: {traceback.format_exc()}")
        
        return chains
    
    def _create_agent(self) -> AgentExecutor:
        """Create ReAct agent with tools."""
        # Create ReAct agent prompt
        react_prompt = PromptTemplate.from_template("""
        You are a specialized EasySalon Beauty Assistant AI with deep knowledge of beauty services, hair care, skincare, and salon operations.

        **LANGUAGE CONSISTENCY RULE**: Always detect and respond in the same language as the user's question.
        - English question → English response
        - Vietnamese question → Vietnamese response
        - Mixed language → Use predominant language

        **CONTEXT**: You help customers with beauty salon services, appointment booking, beauty consultation, and salon information.

        **CONVERSATION HISTORY**: Consider previous interactions to provide personalized beauty recommendations.

        **TOOL USAGE STRATEGY**:
        - **IMPORTANT**: Check if user has an active booking session before using book_appointment tool
        - For appointments: Use check_availability first, then book_appointment ONLY if no active booking session
        - For services: Use search_services to show available services and pricing
        - For beauty advice: Use beauty_consultation for personalized recommendations
        - For salon info: Use get_salon_info for hours, location, policies
        - For bookings: Use retrieve_booking to find existing reservations
        - For recommendations: Use semantic_search for intelligent suggestions
        - Always cross-reference multiple tools for comprehensive answers
        - Consider conversation history for personalized recommendations
        - **BOOKING SESSION RULE**: If user is in a guided booking workflow, avoid using book_appointment tool

        **RESPONSE GUIDELINES**:
        - Maintain language consistency throughout the conversation
        - Include practical tips (pricing, timing, maintenance)
        - Prioritize customer satisfaction and beauty goals
        - Be professional, friendly, and encouraging
        - Focus on beauty, wellness, and self-care
        - Reference previous conversation when relevant for personalization
        - Provide actionable beauty advice and service recommendations

        **BEAUTY EXPERTISE AREAS**:
        - Hair services: cuts, coloring, styling, treatments
        - Skincare: facials, treatments, product recommendations
        - Nail services: manicures, pedicures, nail art
        - Beauty consultation: style advice, color matching
        - Spa services: massages, relaxation treatments

        **AVAILABLE TOOLS**:
        {tools}

        **TOOL NAMES**: {tool_names}

        **CONVERSATION HISTORY**:
        {chat_history}

        **QUESTION**: {input}

        **INSTRUCTIONS**:
        1. Think about what the user is asking for
        2. Determine which tools will help answer their question
        3. Use the tools in logical order
        4. Provide a comprehensive, helpful response
        5. Maintain the same language as the user's question

        Use the following format:
        
        Question: the input question you must answer
        Thought: you should always think about what to do
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question

        Begin!

        Question: {input}
        Thought: {agent_scratchpad}
        """)
        
        try:
            # Create ReAct agent
            agent = create_react_agent(
                llm=self.llm,
                tools=self.tools,
                prompt=react_prompt
            )
            
            # Create agent executor
            agent_executor = AgentExecutor(
                agent=agent,
                tools=self.tools,
                memory=self.memory,
                verbose=True,
                handle_parsing_errors=True,
                max_iterations=2,  # Reduced from 3 to prevent long loops
                early_stopping_method="generate"  # Stop early and generate a response
            )
            
            self.debug("ReAct agent created successfully")
            return agent_executor
            
        except Exception as e:
            self.debug(f"Error creating agent: {e}")
            import traceback
            self.debug(f"Full traceback: {traceback.format_exc()}")
            return None
    
    # Tool functions
    def _check_availability(self, query: str) -> str:
        """Check appointment availability for beauty salon services."""
        try:
            # Parse the query to extract relevant information
            availability_query = self._parse_availability_query(query)
            
            # Use the availability checker
            result = availability_checker.availability_checker.check_availability(availability_query)
            
            if result["success"]:
                return result["formatted_response"]
            else:
                return result.get("error", "Unable to check availability at this time.")
                
        except Exception as e:
            self.debug(f"Error checking availability: {e}")
            return "I'm having trouble checking availability right now. Please try again in a moment."
    
    def _search_salons(self, query: str) -> str:
        """Search for beauty salons and spas by location or services."""
        try:
            # Parse location query
            salon = self.salon.get_salon_info()
            branches = self.salon.get_branches()
            return json.dumps({
                "salon": salon,
                "branches": branches
            })

        except Exception as e:
            self.debug(f"Error searching salons: {e}")
            return "I'm having trouble searching for salons right now. Please try again in a moment."
    
    def _search_services(self, query: str) -> str:
        """Search for beauty services, treatments, and pricing information."""
        try:
            self.debug("SEARCH SERVICE ========\n")
            data = db.search(self.db_client,
                             self.embedding_fn,
                             global_vars.QDRANT_SALON_DATA_COLLECTION,
                             query,
                             limit= 10
                             )
            return json.dumps(data)
        except Exception as e:
            self.debug(f"Error searching services: {e}")
            return "I'm having trouble searching for services right now. Please try again in a moment."
    
    def _get_salon_info(self, salon_identifier: str) -> str:
        """Get detailed information about a specific salon."""
        try:
            # Parse the salon info query
            parsed_query = salon_info_manager.salon_info_manager.parse_salon_info_query(salon_identifier)
            
            # Default salon ID if not specified
            salon_id = parsed_query.get("salon_identifier", "SAL001")
            
            # Get information based on request type
            info_type = parsed_query.get("info_type", "general")
            
            if info_type == "general":
                # Get comprehensive salon details
                salon_details = salon_info_manager.salon_info_manager.get_salon_details(salon_id)
                print(salon_details)
                if salon_details:
                    return salon_info_manager.salon_info_manager.format_salon_details(salon_details)
                else:
                    return f"Salon information not found for '{salon_identifier}'"
            
            elif info_type == "hours":
                # Get salon hours
                hours = salon_info_manager.salon_info_manager.get_salon_hours(salon_id)
                if hours:
                    formatted_hours = "\n".join([f"• {day}: {time}" for day, time in hours.items()])
                    return f"⏰ **Salon Hours:**\n{formatted_hours}"
                else:
                    return "Salon hours information not available."
            
            elif info_type == "contact":
                # Get contact information
                contact_info = salon_info_manager.salon_info_manager.get_salon_contact_info(salon_id)
                if contact_info:
                    return f"""📞 **Contact Information:**
                            • Phone: {contact_info.get('phone', 'Not available')}
                            • Email: {contact_info.get('email', 'Not available')}
                            • Website: {contact_info.get('website', 'Not available')}
                            • Address: {contact_info.get('address', 'Not available')}"""
                else:
                    return "Contact information not available."
            
            elif info_type == "staff":
                # Get staff information
                staff_members = salon_info_manager.salon_info_manager.get_staff_information(salon_id)
                if staff_members:
                    return salon_info_manager.salon_info_manager.format_staff_information(staff_members)
                else:
                    return "Staff information not available."
            
            elif info_type == "policies":
                # Get salon policies
                policies = salon_info_manager.salon_info_manager.get_salon_policies(salon_id)
                if policies:
                    return salon_info_manager.salon_info_manager.format_salon_policies(policies)
                else:
                    return "Policy information not available."
            
            elif info_type == "amenities":
                # Get amenities
                amenities = salon_info_manager.salon_info_manager.get_salon_amenities(salon_id)
                if amenities:
                    formatted_amenities = "\n".join([f"• {amenity}" for amenity in amenities])
                    return f"🏢 **Salon Amenities:**\n{formatted_amenities}"
                else:
                    return "Amenities information not available."
            
            else:
                # Fall back to general information
                salon_details = salon_info_manager.salon_info_manager.get_salon_details(salon_id)
                if salon_details:
                    return salon_info_manager.salon_info_manager.format_salon_details(salon_details)
                else:
                    return f"Salon information not found for '{salon_identifier}'"
                
        except Exception as e:
            self.debug(f"Error getting salon information: {e}")
            return "I'm having trouble getting salon information right now. Please try again in a moment."
    
    def _beauty_consultation(self, query: str) -> str:
        """Provide beauty advice and treatment recommendations."""
        try:
            # Parse the consultation request
            consultation_request = beauty_consultant.beauty_consultant.parse_consultation_request(query)
            
            # Get consultation response
            consultation_response = beauty_consultant.beauty_consultant.provide_consultation(consultation_request)
            
            # Format and return the response
            return beauty_consultant.beauty_consultant.format_consultation_response(consultation_response)
            
        except Exception as e:
            self.debug(f"Error providing beauty consultation: {e}")
            return "I'm having trouble providing beauty consultation right now. Please try again in a moment."
    
    def _extract_beauty_info(self, user_input: str) -> str:
        """Extract beauty service information from user input."""
        try:
            current_date = datetime.now().strftime("%Y-%m-%d")
            current_weekday = datetime.now().strftime("%A")
            
            result = self.chains["service_info"].invoke({
                "question": user_input,
                "current_date": current_date,
                "current_weekday": current_weekday
            })
            
            return result
        except Exception as e:
            return f"Error extracting beauty service info: {e}"
    
    def _generate_beauty_consultation(self, beauty_info: str) -> str:
        """Generate beauty consultation based on service information."""
        try:
            # This would typically process the beauty info and generate a consultation
            return f"Generated beauty consultation based on: {beauty_info}"
        except Exception as e:
            return f"Error generating beauty consultation: {e}"
    
    # Utility methods
    def set_user_id(self, user_id: str):
        """Set user ID for session management."""
        self.user_id = user_id
    
    def set_debug(self, is_enable: bool):
        """Enable or disable debug mode."""
        self.is_debug = is_enable
    
    def debug(self, message: str):
        """Print debug message if debug mode is enabled."""
        # if self.is_debug:
        print(f"[DEBUG] {message}")
    
    def _safe_json_parse(self, response: str, default_response: Dict[str, Any]) -> Dict[str, Any]:
        """Safely parse JSON responses with fallback."""
        try:
            import json
            # Clean common JSON formatting issues
            cleaned = response.strip()
            if cleaned.startswith('```json'):
                cleaned = cleaned.replace('```json', '').replace('```', '')
            if cleaned.startswith('```'):
                cleaned = cleaned.replace('```', '')
            
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            self.debug(f"JSON parsing error: {e}, Response: {response}")
            return default_response
        except Exception as e:
            self.debug(f"Unexpected error parsing response: {e}")
            return default_response
    
    def _detect_language(self, text: str) -> str:
        """Detect language of the input text."""
        # Simple language detection based on Vietnamese characters
        vietnamese_chars = ['ă', 'â', 'đ', 'ê', 'ô', 'ơ', 'ư', 'à', 'á', 'ả', 'ã', 'ạ', 'ằ', 'ắ', 'ẳ', 'ẵ', 'ặ', 
                           'ầ', 'ấ', 'ẩ', 'ẫ', 'ậ', 'è', 'é', 'ẻ', 'ẽ', 'ẹ', 'ề', 'ế', 'ể', 'ễ', 'ệ', 'ì', 'í', 'ỉ', 'ĩ', 'ị',
                           'ò', 'ó', 'ỏ', 'õ', 'ọ', 'ồ', 'ố', 'ổ', 'ỗ', 'ộ', 'ờ', 'ớ', 'ở', 'ỡ', 'ợ', 'ù', 'ú', 'ủ', 'ũ', 'ụ',
                           'ừ', 'ứ', 'ử', 'ữ', 'ự', 'ỳ', 'ý', 'ỷ', 'ỹ', 'ỵ']
        
        vietnamese_words = ['tôi', 'bạn', 'và', 'là', 'có', 'không', 'của', 'được', 'cho', 'này', 'đi', 'du lịch', 'kế hoạch', 'lịch trình']
        
        text_lower = text.lower()
        
        # Check for Vietnamese characters
        has_vietnamese_chars = any(char in text for char in vietnamese_chars)
        
        # Check for Vietnamese words
        has_vietnamese_words = any(word in text_lower for word in vietnamese_words)
        
        if has_vietnamese_chars or has_vietnamese_words:
            return "Tiếng Việt"
        else:
            return "English"
    
    def _get_conversation_context(self) -> str:
        """Get relevant conversation context for better responses."""
        try:
            messages = self.memory.chat_memory.messages
            if not messages:
                return "No previous conversation context."
            
            # Extract last 3 exchanges for context
            recent_context = []
            for msg in messages[-6:]:  # Last 3 human+ai pairs
                if hasattr(msg, 'content'):
                    role = "User" if msg.type == "human" else "Assistant"
                    recent_context.append(f"{role}: {msg.content[:100]}...")
            
            return "CONVERSATION_CONTEXT:\n" + "\n".join(recent_context)
        except Exception as e:
            return f"Context retrieval error: {e}"
    
    def _add_to_memory(self, user_input: str, ai_response: str):
        """Add user input and AI response to conversation memory."""
        try:
            self.memory.chat_memory.add_user_message(user_input)
            self.memory.chat_memory.add_ai_message(ai_response)
            self.debug(f"Added to memory - User: {user_input[:50]}..., AI: {ai_response[:50]}...")
        except Exception as e:
            self.debug(f"Error adding to memory: {e}")
    
    def clear_memory(self):
        """Clear conversation memory."""
        try:
            self.memory.clear()
            self.debug("Conversation memory cleared")
        except Exception as e:
            self.debug(f"Error clearing memory: {e}")
    
    def get_memory_summary(self) -> str:
        """Get a summary of current conversation memory."""
        try:
            messages = self.memory.chat_memory.messages
            if not messages:
                return "No conversation history"
            
            total_messages = len(messages)
            user_messages = len([msg for msg in messages if msg.type == "human"])
            ai_messages = len([msg for msg in messages if msg.type == "ai"])
            
            return f"Memory Summary: {total_messages} total messages ({user_messages} user, {ai_messages} AI)"
        except Exception as e:
            return f"Error getting memory summary: {e}"
    
    def set_print_fn(self, func):
        """Set custom print function."""
        global_vars.print_fn = func
    
    # Main interaction methods
    def is_beauty_related(self, question: str) -> Dict[str, Any]:
        """Check if question is beauty/salon-related using LangChain."""
        try:
            result = self.chains["beauty_relation"].invoke({"question": question})
            # Parse JSON response safely
            default_response = {
                "is_related": True, 
                "response": "", 
                "detected_language": self._detect_language(question)
            }
            return self._safe_json_parse(result, default_response)
        except Exception as e:
            self.debug(f"Error in beauty relation check: {e}")
            return {"is_related": True, "response": "", "detected_language": self._detect_language(question)}
    
    def is_appointment_request(self, question: str) -> Dict[str, Any]:
        """Check if question is an appointment booking request and extract any existing booking information."""
        try:
            result = self.chains["appointment_request"].invoke({"question": question})
            default_response = {
                "is_booking_request": False,
                "detected_language": self._detect_language(question),
                "confidence": 0.5,
                "extracted_info": {},
                "friendly_prompt": ""
            }
            parsed = self._safe_json_parse(result, default_response)
            
            # If it's a booking request, extract any existing information and generate friendly prompt
            if parsed.get("is_booking_request", False):
                # Extract any existing booking information from the user input
                extracted_info = self._extract_booking_info_from_input(question)
                parsed["extracted_info"] = extracted_info
                
                # Generate a friendly prompt based on what's missing
                friendly_prompt = self._generate_booking_friendly_prompt(extracted_info, parsed.get("detected_language", "English"))
                parsed["friendly_prompt"] = friendly_prompt
            
            return parsed
            
        except Exception as e:
            self.debug(f"Error in booking request detection: {e}")
            return {
                "is_booking_request": False,
                "detected_language": self._detect_language(question),
                "confidence": 0.0,
                "extracted_info": {},
                "friendly_prompt": ""
            }
        
    def _extract_booking_info_from_input(self, question: str) -> Dict[str, Any]:
        """Extract any existing booking information from user input."""
        try:
            # Use LLM to extract booking information
            extraction_prompt = ChatPromptTemplate.from_messages([
                ("system", """Extract booking information from the user's message.

                **OUTPUT FORMAT**: JSON only:
                {{
                    "customer_name": "name if mentioned" | null,
                    "customer_mobile": "phone if mentioned" | null,
                    "customer_email": "email if mentioned" | null,
                    "service_type": "service mentioned" | null,
                    "preferred_date": "date if mentioned (YYYY-MM-DD)" | null,
                    "preferred_time": "time if mentioned (HH:MM)" | null,
                    "branch_name": "branch/location if mentioned" | null,
                    "special_requests": "any special requests" | null
                }}"""),
                ("human", "{question}")
            ])
            
            extraction_chain = extraction_prompt | self.llm | StrOutputParser()
            result = extraction_chain.invoke({"question": question})
            
            # Parse the JSON response
            extracted_info = self._safe_json_parse(result, {})
            
            # Clean up None values
            return {k: v for k, v in extracted_info.items() if v is not None}
            
        except Exception as e:
            self.debug(f"Error extracting booking info: {e}")
            return {}

    def _generate_booking_friendly_prompt(self, extracted_info: Dict[str, Any], language: str) -> str:
        """Generate a friendly prompt based on extracted information and missing fields."""
        try:
            # Define required fields
            required_fields = {
                "customer_name": "your name",
                "customer_mobile": "your phone number", 
                "service_type": "the service you'd like",
                "preferred_date": "your preferred date",
                "preferred_time": "your preferred time"
            }
            
            # Find missing fields
            missing_fields = []
            for field, description in required_fields.items():
                if field not in extracted_info or not extracted_info[field]:
                    missing_fields.append(description)
            
            # Generate prompt based on language and missing fields
            if language == "Tiếng Việt":
                if not missing_fields:
                    return "Tuyệt vời! Tôi có đầy đủ thông tin để đặt lịch cho bạn. Hãy để tôi kiểm tra lịch trống."
                else:
                    missing_text = ", ".join(missing_fields)
                    return f"Tôi rất vui được giúp bạn đặt lịch! Để hoàn tất việc đặt lịch, tôi cần thêm thông tin: {missing_text}. Bạn có thể cung cấp những thông tin này không?"
            else:
                if not missing_fields:
                    return "Great! I have all the information needed to book your appointment. Let me check availability for you."
                else:
                    missing_text = ", ".join(missing_fields)
                    return f"I'd be happy to help you book an appointment! To complete your booking, I need: {missing_text}. Could you please provide this information?"
                    
        except Exception as e:
            self.debug(f"Error generating friendly prompt: {e}")
            return "I'd be happy to help you book an appointment! What service are you interested in?"
    
    def collect_client_info(self, user_prompt: str) -> Tuple[Optional[Dict], str]:
        """Collect client information for beauty consultation using LangChain."""
        if user_prompt.lower() in ['exit', 'thoát']:
            global_vars.is_request_plan = False
            detected_lang = self._detect_language(user_prompt)
            goodbye_msg = "Goodbye! Hope to help you with your beauty needs soon!" if detected_lang == "English" else "Tạm biệt! Hy vọng sẽ giúp bạn chăm sóc sắc đẹp sớm nhé!"
            self._add_to_memory(user_prompt, goodbye_msg)
            return None, goodbye_msg
        
        try:
            current_date = datetime.now().strftime("%Y-%m-%d")
            current_weekday = datetime.now().strftime("%A")
            
            result = self.chains["consultation"].invoke({
                "question": user_prompt,
                "current_date": current_date,
                "current_weekday": current_weekday
            })
            
            default_response = {
                "preferred_date": None,
                "service_type": None,
                "duration": None,
                "language": self._detect_language(user_prompt),
                "prompt": "I need more information to help you with your beauty consultation." if self._detect_language(user_prompt) == "English" else "Tôi cần thêm thông tin để giúp bạn tư vấn làm đẹp."
            }
            
            parsed = self._safe_json_parse(result, default_response)
            response_message = parsed.get("prompt", "I need more information to help you with your beauty consultation.")
            
            # Add to memory regardless of completeness
            self._add_to_memory(user_prompt, response_message)
            
            # Check if all required info is present
            required_fields = ["preferred_date", "service_type", "duration"]
            missing_fields = [field for field in required_fields if field not in parsed or parsed[field] is None]
            
            if not missing_fields:
                global_vars.is_request_plan = True
                return parsed, response_message
            else:
                global_vars.is_request_plan = True
                return None, response_message
                
        except Exception as e:
            self.debug(f"Error collecting client info: {e}")
            detected_lang = self._detect_language(user_prompt)
            error_msg = "Sorry, I had trouble understanding your request. Could you please rephrase?" if detected_lang == "English" else "Xin lỗi, tôi gặp khó khăn trong việc hiểu yêu cầu của bạn. Bạn có thể nói lại được không?"
            self._add_to_memory(user_prompt, error_msg)
            return None, error_msg
    
    def generate_beauty_consultation(self, client_info: Dict) -> str:
        """Generate beauty consultation using LangChain."""
        try:
            # Get conversation context for personalized consultation
            conversation_context = self._get_conversation_context()
            
            # Get beauty service data using agent with conversation context
            if self.agent is not None:
                beauty_data = self.agent.invoke({
                    "input": f"Get comprehensive beauty service data for {client_info['service_type']} with duration {client_info['duration']}. Consider previous conversation: {conversation_context[:200]}..."
                })
            else:
                beauty_data = f"Beauty service data for {client_info['service_type']} - duration {client_info['duration']}"
            
            # Generate consultation using the consultation chain
            consultation_params = {
                "service_type": client_info["service_type"],
                "duration": client_info["duration"],
                "preferred_date": client_info["preferred_date"],
                "language": client_info.get("language", "English"),
                "beauty_data": beauty_data,
                "salon_availability": f"Availability data for {client_info['service_type']} on {client_info['preferred_date']}"
            }
            
            result = self.chains["consultation"].invoke(consultation_params)
            
            # Add the consultation to memory
            user_request = f"Generate beauty consultation for {client_info['service_type']} on {client_info['preferred_date']}"
            self._add_to_memory(user_request, result[:200] + "... [Complete consultation generated]")
            
            return result
        except Exception as e:
            self.debug(f"Error generating beauty consultation: {e}")
            error_msg = "Sorry, I encountered an error while generating your beauty consultation."
            # Add error to memory
            self._add_to_memory("Generate beauty consultation", error_msg)
            return error_msg
    
    def answer_beauty_question(self, question: str, context: str = "") -> str:
        """Answer beauty-related questions using LangChain."""
        try:
            # Check if beauty related first
            relation_check = self.is_beauty_related(question)
            
            if not relation_check["is_related"]:
                return relation_check["response"]
            
            # Get relevant context using agent if not provided
            if not context and self.agent is not None:
                context = self.agent.invoke({
                    "input": f"Search for information relevant to this beauty question: {question}"
                })
            elif not context:
                context = "No additional context available"
            
            # Get conversation context for personalized responses
            conversation_context = self._get_conversation_context()
            
            # Answer the question with conversation context
            result = self.chains["beauty_question"].invoke({
                "question": question,
                "context": context,
                "conversation_context": conversation_context
            })
            
            # Add to memory for future context
            self._add_to_memory(question, result)
            
            return result
        except Exception as e:
            self.debug(f"Error answering beauty question: {e}")
            error_msg = "Sorry, I encountered an error while answering your beauty question."
            # Still add error interaction to memory
            self._add_to_memory(question, error_msg)
            return error_msg
    
    def greeting(self):
        """Generate greeting message."""
        try:
            # Play TTS greeting
            # th = threading.Thread(target=tts.play_tts, args=("Hello, I'm your EasySalon Beauty Assistant AI. How can I help you today?",))
            # th.start()
            
            # Generate greeting text
            greeting_chain = ChatPromptTemplate.from_messages([
                ("system", """Generate a friendly, enthusiastic greeting as a beauty salon assistant.

                **LANGUAGE RULE**: Generate greeting in both English and Vietnamese to welcome all users.
                
                **REQUIREMENTS**:
                - Include a hint about typing 'exit' or 'thoát' to quit
                - Keep it brief and engaging
                - Show enthusiasm for beauty and salon services
                - Use appropriate emojis (💄 ✨ 💅 💆‍♀️ 💇‍♀️)
                - Mention your capabilities (beauty consultation, appointment booking, service info, beauty tips)
                
                **FORMAT**: Start with English, then Vietnamese, both clearly marked.""")
            ]) | self.llm | StrOutputParser()
            
            intro = greeting_chain.invoke({})
            if global_vars.print_fn:
                global_vars.print_fn(f"\n🤖 {intro}\n")
            else:
                print(f"\n🤖 {intro}\n")
            
        except Exception as e:
            self.debug(f"Error in greeting: {e}")
            default_greeting = "🤖 Hello! I'm your EasySalon Beauty Assistant! 💄✨ I'm here to help you with beauty consultation, appointment booking, and salon services. Ask me about treatments, book an appointment, or get personalized beauty advice. Type 'exit' anytime to quit. What beauty service can I help you with today? 💅💆‍♀️"
            if global_vars.print_fn:
                global_vars.print_fn(f"\n{default_greeting}\n")
            else:
                print(f"\n{default_greeting}\n")
    
    def goodbye(self):
        """Generate goodbye message."""
        try:
            # Play TTS goodbye
            # th = threading.Thread(target=tts.play_tts, args=("Goodbye! Hope to see you again soon for more beauty services!",))
            # th.start()
            
            # Generate goodbye text
            goodbye_chain = ChatPromptTemplate.from_messages([
                ("system", """Generate a friendly, slightly sad goodbye message as a beauty salon assistant.

                **LANGUAGE RULE**: Generate goodbye in both English and Vietnamese to accommodate all users.
                
                **REQUIREMENTS**:
                - Keep it warm and inviting for future visits
                - Express sadness about leaving but hope for return
                - Mention beauty salon and self-care context
                - Use appropriate emojis (💄 ✨ 💅 💆‍♀️ 💇‍♀️)
                - Be brief but heartfelt
                
                **FORMAT**: Provide both English and Vietnamese versions, clearly marked.""")
            ]) | self.llm | StrOutputParser()
            
            goodbye_msg = goodbye_chain.invoke({})
            if global_vars.print_fn:
                global_vars.print_fn(f"\n🤖 {goodbye_msg}")
            else:
                print(f"\n🤖 {goodbye_msg}")
            
        except Exception as e:
            self.debug(f"Error in goodbye: {e}")
            default_goodbye = "🤖 Goodbye! It was wonderful helping you with your beauty needs. Come back anytime for more pampering and self-care! 💄✨💅"
            if global_vars.print_fn:
                global_vars.print_fn(f"\n{default_goodbye}")
            else:
                print(f"\n🤖 {default_goodbye}")
    
    def handle_question(self, question: str) -> str:
        """Main method to handle user questions using LangChain agent."""
        try:
            self.debug(f"=== HANDLE_QUESTION CALLED ===")
            self.debug(f"Question: {question}")
            
            # Step 1: Capture user input
            user_input = question.strip()
    
            # Step 2: Intent Classification - Check if it's a booking request
            self.debug("Checking if it's a booking request...")
            booking_check = self.is_appointment_request(user_input)
            is_booking = booking_check.get("is_booking_request", False)
            friendly_prompt = booking_check.get("friendly_prompt", "")
            extracted_info = booking_check.get("extracted_info", {})
            self.debug(f"Is booking request: {is_booking}")
            
            # Check if user has an active booking session
            user_session_id = self.user_id or "default_session"
            
            if is_booking or user_session_id in self.active_booking_sessions:
                self.debug("=== HANDLING BOOKING REQUEST WITH LANGGRAPH ===")
                
                # Step 3: Query Qdrant Database & Information Gathering via LangGraph
                if user_session_id not in self.active_booking_sessions:
                    # Start new booking workflow
                    self.debug("Starting new booking workflow...")
                    
                    workflow_result = self.booking_workflow.start_workflow(user_input, extracted_info)
                    
                    # Store the booking session
                    self.active_booking_sessions[user_session_id] = {
                        "state": workflow_result["state"],
                        "started_at": datetime.now().isoformat()
                    }
                    
                    # Combine friendly prompt with workflow results
                    if friendly_prompt and workflow_result.get("response"):
                        # If workflow shows options/next steps, combine them
                        response = friendly_prompt + "\n\n" + workflow_result["response"]
                    else:
                        response = workflow_result["response"] or friendly_prompt
                    
                    # Check if booking is complete (API was actually called, not just workflow ended)
                    if self._is_booking_truly_complete(workflow_result):
                        # Remove completed session - only if API was actually called
                        print(f"=== BOOKING COMPLETED SUCCESSFULLY ===")
                        print(f"API call was triggered in workflow")
                        print(f"Response: {response}")
                        del self.active_booking_sessions[user_session_id]
                        self.debug("Booking completed - session removed")
                    else:
                        # Workflow ended but booking not complete - keep session active
                        self.debug(f"Workflow ended but booking not complete. Step: {workflow_result['state'].get('current_step', 'unknown')}")
                else:
                    # Continue existing booking workflow
                    self.debug("Continuing existing booking workflow...")
                    session_data = self.active_booking_sessions[user_session_id]
                    
                    workflow_result = self.booking_workflow.continue_workflow(
                        session_data["state"], 
                        user_input
                    )
                    
                    # Update session state
                    session_data["state"] = workflow_result["state"]
                    response = workflow_result["response"]
                    
                    # Check if booking is complete (API was actually called, not just workflow ended)
                    if self._is_booking_truly_complete(workflow_result):
                        # Remove completed session - only if API was actually called
                        print(f"=== BOOKING COMPLETED SUCCESSFULLY ===")
                        print(f"API call was triggered in workflow")
                        print(f"Response: {response}")
                        del self.active_booking_sessions[user_session_id]
                        self.debug("Booking completed - session removed")
                    else:
                        # Workflow ended but booking not complete - keep session active
                        self.debug(f"Workflow ended but booking not complete. Step: {workflow_result['state'].get('current_step', 'unknown')}")
                        
                
                # Add the interaction to memory
                self._add_to_memory(user_input, response)
                
                # Add suggestions to response
                response = self._format_response_with_suggestions(response, user_input)
                    
                return response
                
            else:
                self.debug("=== HANDLING REGULAR BEAUTY QUESTION ===")
                # Check if user has an active booking session
                user_session_id = self.user_id or "default_session"
                
                if user_session_id in self.active_booking_sessions:
                    self.debug("=== USER HAS ACTIVE BOOKING SESSION - Using limited agent ===")
                    # User has active booking session but asking non-booking question
                    # Use direct answer instead of agent to avoid conflicts
                    response = self.answer_beauty_question(user_input, "")
                else:
                    self.debug("=== NO ACTIVE BOOKING SESSION - Using full agent ===")
                    # Handle regular beauty questions with full agent capabilities
                    if self.agent is not None:
                        try:
                            agent_result = self.agent.invoke({"input": user_input})
                            response = agent_result.get("output", str(agent_result))
                        except Exception as agent_error:
                            self.debug(f"Agent error: {agent_error}")
                            # Fallback to direct answer if agent fails
                            response = self.answer_beauty_question(user_input, "")
                    else:
                        response = self.answer_beauty_question(user_input, "")
                
                # Add suggestions to response
                response = self._format_response_with_suggestions(response, user_input)
                return response
                
        except Exception as e:
            self.debug(f"Error handling question: {e}")
            self.debug(f"Full traceback: {traceback.format_exc()}")
            
            # Clean up any active booking session on error
            user_session_id = self.user_id or "default_session"
            if user_session_id in self.active_booking_sessions:
                del self.active_booking_sessions[user_session_id]
                
            error_msg = "Sorry, I encountered an error. Please try rephrasing your question."
            return error_msg
    
    def _book_appointment(self, query: str) -> str:
        """Create a new appointment booking."""
        print(f"=== START BOOK APPOINTMENT CALLED ===")
        try:
            # Parse the query to extract booking information
            booking_info = self._parse_booking_query(query)
            
            # Check if we have all required information
            required_fields = ["service_id", "date", "time", "customer_info"]
            missing_fields = []
            
            for field in required_fields:
                if field not in booking_info or not booking_info[field]:
                    missing_fields.append(field)
            
            if missing_fields:
                return f"I need more information to create your booking. Missing: {', '.join(missing_fields)}. Please provide the required details."
            
            # Create booking request
            booking_request = booking_manager.BookingRequest(
                service_id=booking_info["service_id"],
                service_name=booking_info.get("service_name", "Beauty Service"),
                salon_id=booking_info.get("salon_id", "SAL001"),
                staff_id=booking_info.get("staff_id"),
                date=booking_info["date"],
                time=booking_info["time"],
                duration=booking_info.get("duration", 60),
                price=booking_info.get("price", 0.0),
                customer_info=booking_info["customer_info"],
                special_requests=booking_info.get("special_requests")
            )
            
            # Create the booking
            booking_response = booking_manager.booking_manager.create_booking(booking_request)
            
            # Format and return the response
            return booking_manager.booking_manager.format_booking_confirmation(booking_response)
            
        except Exception as e:
            self.debug(f"Error creating booking: {e}")
            return "I'm having trouble creating your booking right now. Please try again in a moment."
    
    def _retrieve_booking(self, query: str) -> str:
        """Retrieve existing booking information by ID or confirmation code."""
        try:
            # Parse the query to extract booking identifier
            parsed_query = booking_retriever.booking_retriever.parse_booking_query(query)
            
            if parsed_query.get("identifier"):
                # Retrieve booking by ID or confirmation code
                booking_result = booking_retriever.booking_retriever.retrieve_booking(parsed_query["identifier"])
                
                if booking_result.success:
                    return booking_retriever.booking_retriever.format_booking_info(booking_result.booking_info)
                else:
                    return f"❌ {booking_result.message}"
            
            elif parsed_query.get("phone"):
                # Search bookings by phone number
                bookings = booking_retriever.booking_retriever.search_bookings_by_phone(parsed_query["phone"])
                
                if bookings:
                    return booking_retriever.booking_retriever.format_booking_list(bookings)
                else:
                    return "No bookings found for the provided phone number."
            
            else:
                return "Please provide a booking ID, confirmation code, or phone number to retrieve your booking information."
                
        except Exception as e:
            self.debug(f"Error retrieving booking: {e}")
            return "I'm having trouble retrieving your booking information right now. Please try again in a moment."
    
    def _parse_booking_query(self, query: str) -> Dict[str, Any]:
        """Parse booking query to extract relevant information."""
        try:
            # Use the booking manager's parsing function
            parsed_info = booking_manager.booking_manager.parse_booking_request(query)
            
            # Add default values if not provided
            if "salon_id" not in parsed_info:
                parsed_info["salon_id"] = "SAL001"  # Default salon
            
            if "price" not in parsed_info:
                parsed_info["price"] = 50.0  # Default price
                
            return parsed_info
            
        except Exception as e:
            self.debug(f"Error parsing booking query: {e}")
            return {}

    def _semantic_search(self, query: str) -> str:
        """Perform intelligent semantic search across all salon data with personalized recommendations."""
        try:
            # Initialize semantic search engine with vectorstores
            semantic_search.semantic_search_engine.initialize_vectorstores(self.vectorstores)
            
            # Parse the search query
            search_query = semantic_search.semantic_search_engine.parse_search_query(query)
            
            # Perform semantic search
            search_results = semantic_search.semantic_search_engine.semantic_search(search_query)
            
            # Get personalized recommendations
            context = {"user_query": query}
            recommendations = semantic_search.semantic_search_engine.get_recommendations(context)
            
            # Format response
            response = ""
            
            if search_results:
                response += semantic_search.semantic_search_engine.format_search_results(search_results)
                response += "\n\n"
            
            if recommendations:
                response += semantic_search.semantic_search_engine.format_recommendations(recommendations)
            
            if not response:
                response = "I couldn't find any relevant information for your search. Try asking about specific services, salons, or beauty treatments."
            
            return response
            
        except Exception as e:
            self.debug(f"Error in semantic search: {e}")
            return "I'm having trouble searching for information right now. Please try again in a moment."
    
    def get_suggestion(self, question: str) -> List[str]:
        """
        Generate contextual suggestions based on the user's question using Qdrant vector database.
        Returns empty list if no suggestions are available from Qdrant.
        """
        try:
            self.debug(f"=== GET_SUGGESTION CALLED ===")
            collection = "pretrain_question"
            questions_vectorstore = db.get_vectorstore(self.db_client, self.embedding_fn, collection)
            # Use Qdrant DB to get contextual suggestions
            suggestions_data = db.get_suggested_questions(
                    vectorstore=questions_vectorstore,
                    user_input=question
            )
            return suggestions_data if suggestions_data else []
        except Exception as e:
            self.debug(f"Error getting suggestions: {e}")
            # Return empty list in case of error
            return []

    
    def _format_response_with_suggestions(self, response: str, question: str) -> str:
        """Format response. Suggestions are now handled separately via WebSocket events."""
        # Note: Suggestions are now sent separately via the WebSocket 'suggestions' event
        # in app.py, so we don't include them in the main response text anymore
        return response
        
    def cleanup_inactive_sessions(self, max_age_minutes: int = 30) -> int:
        """
        Clean up inactive booking sessions that have been idle for too long.
        
        Args:
            max_age_minutes: Maximum age in minutes before a session is considered inactive
            
        Returns:
            Number of sessions cleaned up
        """
        try:
            current_time = datetime.now()
            sessions_to_remove = []
            
            for session_id, session_data in self.active_booking_sessions.items():
                # Parse the session start time
                started_at = datetime.fromisoformat(session_data["started_at"])
                
                # Calculate session age in minutes
                age_minutes = (current_time - started_at).total_seconds() / 60
                
                # Check if session is older than the maximum allowed age
                if age_minutes > max_age_minutes:
                    sessions_to_remove.append(session_id)
                    self.debug(f"Marking inactive session for cleanup: {session_id} (age: {age_minutes:.2f} minutes)")
            
            # Remove inactive sessions
            for session_id in sessions_to_remove:
                del self.active_booking_sessions[session_id]
                
            num_removed = len(sessions_to_remove)
            if num_removed > 0:
                self.debug(f"Cleaned up {num_removed} inactive booking sessions")
                
            return num_removed
            
        except Exception as e:
            self.debug(f"Error cleaning up inactive sessions: {e}")
            return 0
    
    def _is_booking_truly_complete(self, workflow_result: Dict[str, Any]) -> bool:
        """Check if booking is truly complete (API was called) vs just workflow ended."""
        if not workflow_result.get("is_complete", False):
            return False
        
        state = workflow_result.get("state", {})
        
        # Check if we have a confirmation message indicating API success
        conversation_history = state.get("conversation_history", [])
        for msg in reversed(conversation_history):
            if msg.get("role") == "assistant" and "✅ **Booking Confirmed!**" in msg.get("content", ""):
                return True
        
        # Check if current step indicates successful booking creation
        current_step = state.get("current_step", "")
        if current_step == "create_booking" and state.get("is_complete", False):
            return True
            
        return False

    def get_active_sessions_info(self) -> str:
        """Get information about active booking sessions for debugging."""
        if not self.active_booking_sessions:
            return "No active booking sessions"
        
        session_info = []
        for session_id, session_data in self.active_booking_sessions.items():
            state = session_data["state"]
            session_info.append(
                f"Session {session_id}: "
                f"Step={state.get('current_step', 'unknown')}, "
                f"Complete={state.get('is_complete', False)}, "
                f"Missing={len(state.get('missing_fields', []))} fields"
            )
        
        return "Active sessions: " + "; ".join(session_info)
    
    def clear_booking_session(self, session_id: str = None) -> str:
        """Clear a specific booking session or all sessions."""
        if session_id:
            if session_id in self.active_booking_sessions:
                del self.active_booking_sessions[session_id]
                return f"Cleared booking session: {session_id}"
            else:
                return f"No active session found: {session_id}"
        else:
            # Clear all sessions
            count = len(self.active_booking_sessions)
            self.active_booking_sessions.clear()
            return f"Cleared {count} booking session(s)"

# Global instance for backward compatibility
langchain_chatbot = Chatbot()

# Export functions for backward compatibility
def set_user_id(user_id: str):
    langchain_chatbot.set_user_id(user_id)

def set_debug(is_enable: bool):
    langchain_chatbot.set_debug(is_enable)

def set_print_fn(func):
    langchain_chatbot.set_print_fn(func)

def greeting():
    langchain_chatbot.greeting()

def goodbye():
    langchain_chatbot.goodbye()

def handle_question(question: str) -> str:
    return langchain_chatbot.handle_question(question)

def is_beauty_related(question: str) -> Dict[str, Any]:
    return langchain_chatbot.is_beauty_related(question)

def is_appointment_request(question: str) -> bool:
    return langchain_chatbot.is_appointment_request(question)

def collect_client_info(user_prompt: str) -> Tuple[Optional[Dict], str]:
    return langchain_chatbot.collect_client_info(user_prompt)

def generate_beauty_consultation(client_info: Dict) -> str:
    return langchain_chatbot.generate_beauty_consultation(client_info)

def answer_beauty_question(question: str, context: str = "") -> str:
    return langchain_chatbot.answer_beauty_question(question, context)

def clear_memory():
    """Clear conversation memory."""
    return langchain_chatbot.clear_memory()

def get_memory_summary() -> str:
    """Get a summary of current conversation memory."""
    return langchain_chatbot.get_memory_summary()

def get_conversation_context() -> str:
    """Get current conversation context."""
    return langchain_chatbot._get_conversation_context()

def get_suggestion(question: str) -> List[str]:
    """Get contextual suggestions based on user question."""
    return langchain_chatbot.get_suggestion(question)

def cleanup_inactive_sessions(max_age_minutes: int = 30) -> int:
    """Clean up inactive booking sessions."""
    return langchain_chatbot.cleanup_inactive_sessions(max_age_minutes)