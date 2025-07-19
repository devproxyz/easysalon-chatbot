"""
Booking Workflow Module for EasySalon Chatbot
Implements the appointment booking workflow using LangGraph.
"""

import json
import logging
from datetime import datetime
import dateparser
from typing import Dict, List, Any, Optional, Tuple, Union, Annotated, TypedDict
import uuid

import requests
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain.prompts import ChatPromptTemplate
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_openai import ChatOpenAI

from src import global_vars
from src import qdrant_db as db
from src import availability_checker
from src import booking_manager
from src.api_services import APIService
from src import easysalon

# Define the state type for our booking workflow
class BookingState(TypedDict):
    """State for the booking workflow."""
    
    # Required booking information
    customer_mobile: Optional[str]
    customer_name: Optional[str]
    total_customer: Optional[int]
    branch_id: Optional[str]
    service_id: Optional[str]
    booking_date: Optional[str]
    booking_time: Optional[str]
    
    # Additional contextual information
    branch_options: Optional[List[Dict[str, Any]]]
    service_options: Optional[List[Dict[str, Any]]]
    conversation_history: List[Dict[str, str]]
    
    # Workflow control
    current_step: str
    missing_fields: List[str]
    is_complete: bool
    error_message: Optional[str]
    
    # Memory and editing context
    session_id: Optional[str]  # For persistent memory
    edit_mode: bool  # Whether user is editing booking info
    last_user_intent: Optional[str]  # Track user's intent (edit, confirm, etc.)
    booking_summary_shown: bool  # Track if summary has been shown


# Helper functions for state management
def add_message(state: BookingState, role: str, content: str) -> None:
    """Add a message to the conversation history."""
    if "conversation_history" not in state:
        state["conversation_history"] = []
    
    state["conversation_history"].append({
        "role": role,
        "content": content,
        "timestamp": datetime.now().isoformat()
    })


def get_missing_fields(state: BookingState) -> List[str]:
    """Get a list of required fields that are still missing."""
    missing = []
    required_fields = [
        "customer_mobile", "customer_name", "branch_id", 
        "service_id", "booking_date", "booking_time"
    ]
    
    for field in required_fields:
        if not state.get(field):
            missing.append(field)
            
    return missing


def update_missing_fields(state: BookingState) -> None:
    """Update the list of missing fields."""
    state["missing_fields"] = get_missing_fields(state)
    state["is_complete"] = len(state["missing_fields"]) == 0


def initialize_booking_state(session_id: Optional[str] = None) -> BookingState:
    """Initialize a new booking state with default values."""
    return BookingState(
        customer_mobile=None,
        customer_name=None,
        total_customer=1,
        branch_id=None,
        service_id=None,
        booking_date=None,
        booking_time=None,
        branch_options=[],
        service_options=[],
        conversation_history=[],
        current_step="start",
        missing_fields=[],
        is_complete=False,
        error_message=None,
        session_id=session_id or str(uuid.uuid4()),
        edit_mode=False,
        last_user_intent=None,
        booking_summary_shown=False
    )


def detect_user_intent(user_message: str) -> str:
    """Detect user's intent from their message."""
    message_lower = user_message.lower().strip()
    
    # Edit commands
    if any(word in message_lower for word in ['edit', 'change', 'modify', 'update']):
        return 'edit'
    
    # Show summary/status
    if any(word in message_lower for word in ['summary', 'status', 'show', 'review', 'details']):
        return 'show_summary'
    
    # Start over
    if any(word in message_lower for word in ['start over', 'restart', 'begin again', 'reset']):
        return 'start_over'
    
    # Confirm booking - be more specific to avoid false positives
    # Only trigger on explicit confirmation words, not general "book" phrases
    if any(phrase in message_lower for phrase in ['confirm booking', 'confirm appointment', 'yes confirm', 'proceed with booking']):
        return 'confirm'
    elif message_lower in ['confirm', 'yes', 'ok', 'proceed', 'go ahead']:
        return 'confirm'
    
    # Cancel booking
    if any(word in message_lower for word in ['cancel', 'stop', 'quit', 'exit']):
        return 'cancel'
    
    # Default: provide information
    return 'provide_info'


def generate_booking_summary(state: BookingState) -> str:
    """Generate a comprehensive booking summary with edit instructions."""
    summary = "📋 **Current Booking Information:**\n\n"
    
    # Get display names for branch and service
    branch_name = "Not selected"
    service_name = "Not selected"
    
    if state.get("branch_id") and state.get("branch_options"):
        for branch in state["branch_options"]:
            if str(branch.get("id", "")) == str(state["branch_id"]):
                branch_name = branch.get("name", "Unknown branch")
                break
    
    if state.get("service_id") and state.get("service_options"):
        for service in state["service_options"]:
            if str(service.get("id", "")) == str(state["service_id"]):
                service_name = service.get("name", "Unknown service")
                break
    
    # Build summary with status indicators
    summary += f"👤 **Customer:** {state.get('customer_name', '❌ Not provided')}\n"
    summary += f"📱 **Phone:** {state.get('customer_mobile', '❌ Not provided')}\n"
    summary += f"🏪 **Branch:** {branch_name}\n"
    summary += f"💅 **Service:** {service_name}\n"
    summary += f"📅 **Date:** {state.get('booking_date', '❌ Not provided')}\n"
    summary += f"⏰ **Time:** {state.get('booking_time', '❌ Not provided')}\n"
    summary += f"👥 **People:** {state.get('total_customer', 1)}\n\n"
    
    # Show missing fields
    if state.get("missing_fields"):
        summary += "⚠️ **Missing Information:**\n"
        for field in state["missing_fields"]:
            field_display = field.replace("_", " ").title()
            summary += f"• {field_display}\n"
        summary += "\n"
    
    # Add edit instructions
    summary += "📝 **How to make changes:**\n"
    summary += "• Say 'edit name' to change your name\n"
    summary += "• Say 'edit phone' to change your phone number\n"
    summary += "• Say 'edit branch' to select a different branch\n"
    summary += "• Say 'edit service' to select a different service\n"
    summary += "• Say 'edit date' to change the date\n"
    summary += "• Say 'edit time' to change the time\n"
    summary += "• Say 'start over' to restart the booking\n"
    summary += "• Say 'confirm' when everything looks correct\n"
    
    return summary


def handle_edit_command(state: BookingState, user_message: str) -> Tuple[str, str]:
    """Handle user's edit command and return the field to edit and next step."""
    message_lower = user_message.lower()
    
    # Map edit commands to fields and next steps
    edit_mappings = {
        'name': ('customer_name', 'collect_customer_info'),
        'phone': ('customer_mobile', 'collect_customer_info'),
        'mobile': ('customer_mobile', 'collect_customer_info'),
        'branch': ('branch_id', 'query_branches'),
        'location': ('branch_id', 'query_branches'),
        'salon': ('branch_id', 'query_branches'),
        'service': ('service_id', 'query_services'),
        'treatment': ('service_id', 'query_services'),
        'date': ('booking_date', 'collect_customer_info'),
        'time': ('booking_time', 'collect_customer_info'),
        'people': ('total_customer', 'collect_customer_info'),
        'customer': ('total_customer', 'collect_customer_info')
    }
    
    for keyword, (field, next_step) in edit_mappings.items():
        if keyword in message_lower:
            return field, next_step
    
    # Default: show options
    return None, 'show_edit_options'


class BookingWorkflow:
    """
    LangGraph-based workflow for the appointment booking process.
    Manages the interactive collection of booking information and API calls.
    Includes persistent memory and user editing capabilities.
    """
    
    def __init__(self):
        """Initialize the booking workflow with memory support."""
        self.logger = logging.getLogger(__name__)
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            openai_api_key=global_vars.AZURE_OPENAI_API_KEY,
            openai_api_base=global_vars.AZURE_OPENAI_ENDPOINT,
            model_name=global_vars.OPENAI_MODEL,
            temperature=0.7
        )
        
        # Initialize memory saver for persistent state
        self.memory = MemorySaver()
        
        # Initialize API service for booking
        self.api_service = APIService()
        self.salon = easysalon.Easysalon(api_key=global_vars.EASYSALON_API_KEY)
        # Create the workflow graph
        self.workflow = self._create_workflow_graph()
        
    def _create_workflow_graph(self) -> StateGraph:
        """Create the LangGraph workflow for booking process with memory support."""
        # Define the graph
        workflow = StateGraph(BookingState)
        
        # Add nodes
        workflow.add_node("start", self._start_node)
        workflow.add_node("extract_info", self._extract_booking_info)
        workflow.add_node("handle_user_intent", self._handle_user_intent)
        workflow.add_node("show_summary", self._show_summary)
        workflow.add_node("handle_edit", self._handle_edit)
        workflow.add_node("query_branches", self._query_branches)
        workflow.add_node("query_services", self._query_services)
        workflow.add_node("collect_customer_info", self._collect_customer_info)
        workflow.add_node("confirm_details", self._confirm_details)
        workflow.add_node("create_booking", self._create_booking)
        workflow.add_node("handle_error", self._handle_error)
        
        # Define edges
        workflow.add_edge("start", "extract_info")
        workflow.add_conditional_edges(
            "extract_info",
            self._route_after_extraction,
            {
                "handle_user_intent": "handle_user_intent",
                "query_branches": "query_branches",
                "query_services": "query_services",
                "collect_customer_info": "collect_customer_info",
                "confirm_details": "confirm_details"
            }
        )
        
        # User intent handling
        workflow.add_conditional_edges(
            "handle_user_intent",
            self._route_user_intent,
            {
                "show_summary": "show_summary",
                "handle_edit": "handle_edit",
                "start_over": "start",
                "confirm_details": "confirm_details",
                "extract_info": "extract_info",
                "end": END
            }
        )
        
        workflow.add_edge("show_summary", END)
        workflow.add_conditional_edges(
            "handle_edit",
            self._route_after_edit,
            {
                "query_branches": "query_branches",
                "query_services": "query_services",
                "collect_customer_info": "collect_customer_info",
                "show_summary": "show_summary"
            }
        )
        
        workflow.add_conditional_edges(
            "query_branches",
            self._check_if_options_shown,
            {
                "end": END
            }
        )
        workflow.add_conditional_edges(
            "query_services", 
            self._check_if_options_shown,
            {
                "end": END
            }
        )
        workflow.add_conditional_edges(
            "collect_customer_info",
            self._check_if_customer_info_collected,
            {
                "end": END
            }
        )
        
        workflow.add_conditional_edges(
            "confirm_details",
            self._check_booking_completeness,
            {
                "create_booking": "create_booking",
                "end": END
            }
        )
        
        workflow.add_conditional_edges(
            "create_booking",
            self._check_booking_success,
            {
                "end": END,
                "handle_error": "handle_error"
            }
        )
        
        workflow.add_edge("handle_error", "extract_info")
        
        # Set the entry point
        workflow.set_entry_point("start")

        # Compile with memory support
        compiled_workflow = workflow.compile(checkpointer=self.memory)
        
        return compiled_workflow
        
        return compiled_workflow
    
    # Node implementations
    def _start_node(self, state: BookingState) -> BookingState:
        """Initialize the booking workflow."""
        # Set the current step
        state["current_step"] = "start"
        
        # Initialize session ID if not present
        if not state.get("session_id"):
            state["session_id"] = str(uuid.uuid4())
        
        # Add a welcome message
        add_message(state, "system", 
            "Starting the appointment booking process. I'll help you book an appointment at EasySalon. "
            "Please provide information like your name, phone number, preferred branch, service, date and time."
        )
        
        return state
    
    def _handle_user_intent(self, state: BookingState) -> BookingState:
        """Handle user's intent (edit, show summary, confirm, etc.)."""
        state["current_step"] = "handle_user_intent"
        
        # Get the latest user message
        latest_user_message = next((msg for msg in reversed(state["conversation_history"]) 
                                  if msg["role"] == "user"), None)
        
        if latest_user_message:
            user_message = latest_user_message["content"]
            intent = detect_user_intent(user_message)
            state["last_user_intent"] = intent
            
            self.logger.info(f"[USER_INTENT] Detected intent: {intent}")
            
            # Handle different intents
            if intent == "show_summary":
                state["current_step"] = "show_summary"
            elif intent == "edit":
                state["current_step"] = "handle_edit"
                state["edit_mode"] = True
            elif intent == "start_over":
                # Reset booking fields but keep session
                session_id = state.get("session_id")
                conversation_history = state.get("conversation_history", [])
                state.clear()
                state.update(initialize_booking_state(session_id))
                state["conversation_history"] = conversation_history
                add_message(state, "assistant", "Let's start over with your booking. What would you like to book?")
                state["current_step"] = "start"
            elif intent == "confirm":
                state["current_step"] = "confirm_details"
            elif intent == "cancel":
                add_message(state, "assistant", "Your booking has been cancelled. Feel free to start a new booking anytime!")
                state["current_step"] = "cancelled"
            
        return state
    
    def _show_summary(self, state: BookingState) -> BookingState:
        """Show booking summary with edit instructions."""
        state["current_step"] = "show_summary"
        
        summary = generate_booking_summary(state)
        add_message(state, "assistant", summary)
        state["booking_summary_shown"] = True
        
        return state
    
    def _handle_edit(self, state: BookingState) -> BookingState:
        """Handle user's request to edit booking information."""
        state["current_step"] = "handle_edit"
        
        # Get the latest user message
        latest_user_message = next((msg for msg in reversed(state["conversation_history"]) 
                                  if msg["role"] == "user"), None)
        
        if latest_user_message:
            user_message = latest_user_message["content"]
            field_to_edit, next_step = handle_edit_command(state, user_message)
            
            if field_to_edit:
                # Clear the field to be edited
                state[field_to_edit] = None
                
                # Also clear related options if editing branch or service
                if field_to_edit == "branch_id":
                    state["branch_options"] = []
                elif field_to_edit == "service_id":
                    state["service_options"] = []
                
                # Set next step
                state["current_step"] = next_step
                
                # Add informative message
                field_display = field_to_edit.replace("_", " ").title()
                add_message(state, "assistant", f"I'll help you update your {field_display}.")
                
                # Update missing fields
                update_missing_fields(state)
            else:
                # Show edit options
                edit_options = ("I can help you edit any of the following:\n"
                               "• Name - say 'edit name'\n"
                               "• Phone - say 'edit phone'\n"
                               "• Branch - say 'edit branch'\n"
                               "• Service - say 'edit service'\n"
                               "• Date - say 'edit date'\n"
                               "• Time - say 'edit time'\n\n"
                               "What would you like to change?")
                add_message(state, "assistant", edit_options)
                state["current_step"] = "show_summary"
        
        return state
    
    def _check_if_options_shown(self, state: BookingState) -> str:
        """Check if options have been shown to the user and determine next step."""
        try:
            # Get the latest assistant message
            latest_assistant_message = next((msg for msg in reversed(state["conversation_history"]) 
                                        if msg["role"] == "assistant"), None)
            
            if latest_assistant_message:
                content = latest_assistant_message["content"]
                
                # If we've shown branches or services, end workflow to wait for user input
                if ("Available Branches:" in content or 
                    "Available Services:" in content or
                    "Please let me know which" in content):
                    return "end"
            
            # If no options shown yet, end workflow (this should not happen in normal flow)
            return "end"
        
        except Exception as e:
            self.logger.error(f"Error in _check_if_options_shown: {str(e)}")
            return "end"
    
    def _extract_booking_info(self, state: BookingState) -> BookingState:
        """Extract booking information from user messages."""
        state["current_step"] = "extract_info"
        
        # Get the latest user message
        latest_user_message = next((msg for msg in reversed(state["conversation_history"]) 
                                  if msg["role"] == "user"), None)
        
        if not latest_user_message:
            add_message(state, "system", "No user message found.")
            return state
        
        user_message = latest_user_message["content"]
        
        # Check if user is selecting from branch options
        if state.get("branch_options") and not state.get("branch_id"):
            selection = self._parse_selection(user_message, state["branch_options"])
            if selection:
                state["branch_id"] = str(selection["id"])
                add_message(state, "system", f"Selected branch: {selection['name']}")
                update_missing_fields(state)
                return state
        
        # Check if user is selecting from service options
        if state.get("service_options") and not state.get("service_id"):
            selection = self._parse_selection(user_message, state["service_options"])
            if selection:
                state["service_id"] = str(selection["id"])
                add_message(state, "system", f"Selected service: {selection['name']}")
                update_missing_fields(state)
                return state
        
        # Continue with original LLM extraction for other fields
        self._extract_with_llm(state, user_message)
        
        return state
    
    def _parse_selection(self, user_message: str, options: List[Dict]) -> Optional[Dict]:
        """Parse user's selection from options."""
        user_msg = user_message.lower()
        
        # Check for number selection (1, 2, 3, etc.)
        for i, option in enumerate(options[:5], 1):
            patterns = [
                f" {i} ",      # " 1 "
                f" {i}.",      # " 1."
                f" {i},",      # " 1,"
                f"option {i}", # "option 1"
                f"choice {i}", # "choice 1"
                f"number {i}", # "number 1"
                f"#{i}",       # "#1"
                f"({i})",      # "(1)"
            ]
            
            # Also check if the message is just the number
            if user_msg.strip() == str(i):
                return option
                
            # Check all patterns
            for pattern in patterns:
                if pattern in user_msg:
                    return option
        
        # Check for name selection
        for option in options:
            if option["name"].lower() in user_msg:
                return option
        
        return None
    
    def _extract_with_llm(self, state: BookingState, user_message: str) -> None:
        """Extract booking information using LLM."""
        # Use LLM to extract booking information
        prompt = ChatPromptTemplate.from_template("""
        Extract booking information from this message: "{message}"
        
        Current known information:
        - Customer Name: {customer_name}
        - Customer Mobile: {customer_mobile}
        - Branch ID: {branch_id}
        - Service ID: {service_id}
        - Booking Date: {booking_date}
        - Booking Time: {booking_time}
        - Total Customers: {total_customer}
        
        Parse out any missing information and provide a JSON response with only the extracted fields.
        Only include fields that are mentioned in the user message.
        
        Return ONLY valid JSON without code blocks:
        {{
          "customer_name": "extracted name if mentioned",
          "customer_mobile": "extracted phone if mentioned",
          "booking_date": "extracted date if mentioned",
          "booking_time": "extracted time if mentioned",
          "next_step": "collect_customer_info"
        }}
        """)
        
        chain = prompt | self.llm | StrOutputParser()
        
        # Run the extraction chain
        result = chain.invoke({
            "message": user_message,
            "customer_name": state.get("customer_name", "Unknown"),
            "customer_mobile": state.get("customer_mobile", "Unknown"),
            "branch_id": state.get("branch_id", "Unknown"),
            "service_id": state.get("service_id", "Unknown"),
            "booking_date": state.get("booking_date", "Unknown"),
            "booking_time": state.get("booking_time", "Unknown"),
            "total_customer": state.get("total_customer", 1)
        })
        
        # Process the extraction result
        try:
            # Clean up JSON string if needed
            if "```json" in result:
                result = result.split("```json")[1].split("```")[0].strip()
            elif "```" in result:
                result = result.split("```")[1].split("```")[0].strip()
            
            # Remove any leading/trailing whitespace and ensure it's valid JSON
            result = result.strip()
            
            extraction_data = json.loads(result)
            
            # Update state with extracted information
            for key, value in extraction_data.items():
                if key != "next_step" and value and value != "Unknown":
                    state[key] = value
                    
            # Update missing fields
            update_missing_fields(state)
            
        except Exception as e:
            add_message(state, "system", f"Error processing extraction: {str(e)}")
            # Simple fallback: try to extract name and phone manually
            self._extract_manually(state, user_message)
        
        # Update missing fields
        update_missing_fields(state)
    
    def _query_branches(self, state: BookingState) -> BookingState:
        """Query available branches from Qdrant database."""
        state["current_step"] = "query_branches"
        
        # First check if user is selecting from existing branch options
        if state.get("branch_options") and not state.get("branch_id"):
            latest_user_message = next((msg for msg in reversed(state["conversation_history"]) 
                                      if msg["role"] == "user"), None)
            
            if latest_user_message:
                selection = self._parse_selection(latest_user_message["content"], state["branch_options"])
                if selection:
                    state["branch_id"] = str(selection["id"])
                    add_message(state, "system", f"Selected branch: {selection['name']}")
                    update_missing_fields(state)
                    return state
        
        try:
            # Use availability checker to fetch branches
            checker = availability_checker.AvailabilityChecker()
            query = availability_checker.AvailabilityQuery()
            
            # If we have a service type, include it in the query
            if state.get("service_id"):
                query.service_type = state["service_id"]
                
            branches = checker._fetch_salon_branches(query)
            
            if branches:
                # Store branch options
                state["branch_options"] = branches
                
                # Format branch information for response
                branch_info = "\n\n**Available Branches:**\n"
                for i, branch in enumerate(branches[:5], 1):  # Limit to 5 branches for display
                    branch_info += f"**{i}. {branch['name']}**\n"
                    branch_info += f"   • Address: {branch.get('address', 'N/A')}\n"
                    branch_info += f"   • Phone: {branch.get('mobile', 'N/A')}\n"
                    branch_info += f"   • ID: {branch.get('id', 'N/A')}\n"
                
                # Add response message
                add_message(state, "assistant", 
                    f"Here are some salon branches you can choose from: {branch_info}\n\n"
                    f"Please let me know which branch you'd prefer by name or number."
                )
            else:
                add_message(state, "assistant", 
                    "I couldn't find any salon branches. Could you please provide a location "
                    "or branch name you're interested in?"
                )
        except Exception as e:
            add_message(state, "system", f"Error querying branches: {str(e)}")
        
        return state
    
    def _query_services(self, state: BookingState) -> BookingState:
        """Query available services from Qdrant database."""
        state["current_step"] = "query_services"
        
        # First check if user is selecting from existing service options
        if state.get("service_options") and not state.get("service_id"):
            latest_user_message = next((msg for msg in reversed(state["conversation_history"]) 
                                      if msg["role"] == "user"), None)
            
            if latest_user_message:
                selection = self._parse_selection(latest_user_message["content"], state["service_options"])
                if selection:
                    state["service_id"] = str(selection["id"])
                    add_message(state, "system", f"Selected service: {selection['name']}")
                    update_missing_fields(state)
                    return state
        
        try:
            # Use availability checker to fetch services
            checker = availability_checker.AvailabilityChecker()
            query = availability_checker.AvailabilityQuery()
            
            # If we have a branch ID, include it in the query
            if state.get("branch_id"):
                query.branch_id = state["branch_id"]
                
            services = self.salon.get_services()
            
            if services:
                # Store service options
                state["service_options"] = services
                
                # Format service information for response
                service_info = "\n\n**Available Services:**\n"
                for i, service in enumerate(services[:5], 1):  # Limit to 5 services for display
                    service_info += f"**{i}. {service['name']}**\n"
                    service_info += f"   • Price: {service.get('price', 'N/A')} VND\n"
                    service_info += f"   • Duration: {service.get('time', 'N/A')} minutes\n"
                    service_info += f"   • ID: {service.get('id', 'N/A')}\n"
                
                # Add response message
                add_message(state, "assistant", 
                    f"Here are some services you can choose from: {service_info}\n\n"
                    f"Please let me know which service you'd like by name or number."
                )
            else:
                add_message(state, "assistant", 
                    "I couldn't find any services. Could you please tell me what kind of "
                    "beauty service you're interested in?"
                )
        except Exception as e:
            add_message(state, "system", f"Error querying services: {str(e)}")
        
        return state
    
    def _collect_customer_info(self, state: BookingState) -> BookingState:
        """Collect customer information."""
        state["current_step"] = "collect_customer_info"
        
        # Check which customer information is missing
        missing = []
        if not state.get("customer_name"):
            missing.append("name")
        if not state.get("customer_mobile"):
            missing.append("phone number")
            
        if missing:
            missing_str = " and ".join(missing)
            add_message(state, "assistant", 
                f"To book your appointment, I need your {missing_str}. "
                f"Could you please provide this information?"
            )
        else:
            # If we have all customer info, move to confirm details
            state["current_step"] = "confirm_details"
            
        return state
    
    def _confirm_details(self, state: BookingState) -> BookingState:
        """Confirm booking details with the user."""
        state["current_step"] = "confirm_details"
        
        # Update missing fields
        update_missing_fields(state)
        
        # Format booking details for confirmation
        confirmation_text = "**Please confirm your booking details:**\n\n"
        
        # Get branch name if we have branch_id
        branch_name = "Unknown branch"
        if state.get("branch_id") and state.get("branch_options"):
            for branch in state["branch_options"]:
                if str(branch.get("id", "")) == str(state["branch_id"]):
                    branch_name = branch.get("name", "Unknown branch")
                    break
        
        # Get service name if we have service_id
        service_name = "Unknown service"
        if state.get("service_id") and state.get("service_options"):
            for service in state["service_options"]:
                if str(service.get("id", "")) == str(state["service_id"]):
                    service_name = service.get("name", "Unknown service")
                    break
                    
        # Build confirmation message
        confirmation_text += f"• Name: {state.get('customer_name', 'Not provided')}\n"
        confirmation_text += f"• Phone: {state.get('customer_mobile', 'Not provided')}\n"
        confirmation_text += f"• Branch: {branch_name} (ID: {state.get('branch_id', 'Not selected')})\n"
        confirmation_text += f"• Service: {service_name} (ID: {state.get('service_id', 'Not selected')})\n"
        confirmation_text += f"• Date: {state.get('booking_date', 'Not provided')}\n"
        confirmation_text += f"• Time: {state.get('booking_time', 'Not provided')}\n"
        confirmation_text += f"• Number of people: {state.get('total_customer', 1)}\n\n"
        
        # Check if there are missing fields
        if state["missing_fields"]:
            missing_fields_text = ", ".join(state["missing_fields"])
            confirmation_text += f"⚠️ **The following information is still needed:** {missing_fields_text}\n\n"
            confirmation_text += "Please provide the missing information so I can complete your booking."
        else:
            confirmation_text += "If everything looks correct, please type 'confirm' to book your appointment.\n"
            confirmation_text += "If you want to make changes, please let me know what you'd like to change."
            
        add_message(state, "assistant", confirmation_text)
        
        return state
    
    def _create_booking(self, state: BookingState) -> BookingState:
        """Create the booking using the API."""
        state["current_step"] = "create_booking"
        
        try:
            # Prepare booking request data for the API service
            # Convert date and time to API format
            api_date = BookingWorkflow.convert_to_api_date(state["booking_date"])
            api_time = BookingWorkflow.convert_to_api_time(state["booking_time"])
            booking_request = {
                "customer_name": state["customer_name"],
                "customer_mobile": state["customer_mobile"],
                "total_customer": state.get("total_customer", 1),
                "branch_id": int(state["branch_id"]) if state["branch_id"] else 8850,  # Default branch
                "service_id": int(state["service_id"]) if state["service_id"] else 257170,  # Default service
                "booking_date": api_date,
                "booking_time": api_time
            }
            
            # Log the booking request payload for debugging
            self.logger.info(f"Booking API Request Data: {json.dumps(booking_request)}")
            
            # Use the API service to create the booking
            api_response = self.salon.book_appointment(booking_request)
            
            # Check if the response indicates success
            if api_response and api_response.get("bookingCode", False) == True:  # Success if no explicit failure
                # Log successful response
                self.logger.info(f"Booking API Response: {json.dumps(api_response)}")
                
                # Format confirmation message
                confirmation = "✅ **Booking Confirmed!**\n\n"
                
                # Extract information from API response
                if "id" in api_response:
                    confirmation += f"• Booking ID: {api_response['id']}\n"
                if "bookingCode" in api_response:
                    confirmation += f"• Confirmation Code: {api_response['bookingCode']}\n"
                
                # Add booking details
                confirmation += f"• Customer: {state['customer_name']}\n"
                confirmation += f"• Phone: {state['customer_mobile']}\n"
                
                # Get branch name
                branch_name = "Unknown branch"
                if state.get("branch_options"):
                    for branch in state["branch_options"]:
                        if str(branch.get("id", "")) == str(state["branch_id"]):
                            branch_name = branch.get("name", "Unknown branch")
                            break
                confirmation += f"• Branch: {branch_name}\n"
                
                # Get service name
                service_name = "Unknown service"
                if state.get("service_options"):
                    for service in state["service_options"]:
                        if str(service.get("id", "")) == str(state["service_id"]):
                            service_name = service.get("name", "Unknown service")
                            break
                confirmation += f"• Service: {service_name}\n"
                
                confirmation += f"• Date: {state['booking_date']}\n"
                confirmation += f"• Time: {state['booking_time']}\n\n"
                confirmation += "Thank you for booking with EasySalon! We look forward to seeing you."
                
                add_message(state, "assistant", confirmation)
                state["is_complete"] = True
            else:
                # Handle API error
                error_message = api_response.get("message", "Unknown error occurred") if api_response else "Failed to connect to booking service"
                state["error_message"] = error_message
                add_message(state, "assistant", 
                    f"❌ **Booking Failed:** {error_message}\n\n"
                    f"Let me help you fix the issue and try again."
                )
                state["current_step"] = "handle_error"
                
        except Exception as e:
            self.logger.error(f"Error creating booking via API service: {str(e)}")
            state["error_message"] = str(e)
            add_message(state, "assistant", 
                f"❌ **Booking Error:** I encountered an error while creating your booking.\n\n"
                f"Error details: {str(e)}\n\n"
                f"Let me help you try again."
            )
            state["current_step"] = "handle_error"
        
        return state
    
    def _handle_error(self, state: BookingState) -> BookingState:
        """Handle errors in the booking process."""
        state["current_step"] = "handle_error"
        
        error_msg = state.get("error_message", "Unknown error")
        
        # Analyze error message to determine the best recovery action
        if "service" in error_msg.lower():
            add_message(state, "assistant", 
                "It seems there was an issue with the service selection. "
                "Let me help you choose an available service."
            )
            state["current_step"] = "query_services"
        elif "branch" in error_msg.lower():
            add_message(state, "assistant", 
                "It seems there was an issue with the branch selection. "
                "Let me help you choose an available branch."
            )
            state["current_step"] = "query_branches"
        elif "date" in error_msg.lower() or "time" in error_msg.lower():
            add_message(state, "assistant", 
                "It seems there was an issue with the booking date or time. "
                "Please provide a different date or time for your appointment."
            )
            # Clear the problematic fields
            state["booking_date"] = None
            state["booking_time"] = None
            update_missing_fields(state)
            state["current_step"] = "collect_customer_info"
        else:
            add_message(state, "assistant", 
                "Let's try again with your booking. "
                "Please confirm your information again."
            )
            state["current_step"] = "confirm_details"
        
        # Clear error message after handling
        state["error_message"] = None
        
        return state
    
    # Conditional routing functions
    def _route_after_extraction(self, state: BookingState) -> str:
        """Determine the next step after information extraction."""
        # Check if user has a specific intent (edit, summary, etc.)
        latest_user_message = next((msg for msg in reversed(state["conversation_history"]) 
                                   if msg["role"] == "user"), None)
        
        if latest_user_message:
            user_message = latest_user_message["content"]
            intent = detect_user_intent(user_message)
            
            self.logger.info(f"[ROUTING] User intent: {intent}")
            
            # Handle user intents first
            if intent in ['edit', 'show_summary', 'start_over', 'confirm', 'cancel']:
                return "handle_user_intent"
        
        # Add debugging
        self.logger.info(f"[ROUTING] Current state: branch_id={state.get('branch_id')}, service_id={state.get('service_id')}")
        self.logger.info(f"[ROUTING] Branch options: {len(state.get('branch_options', []))}, Service options: {len(state.get('service_options', []))}")
        
        # Check if user is selecting from options
        if latest_user_message:
            content = latest_user_message["content"].lower()
            self.logger.info(f"[ROUTING] User message: {content}")
            
            # If user is trying to select a service option, don't route to query_services
            if (state.get("service_options") and not state.get("service_id") and 
                any(word in content for word in ["1", "2", "3", "4", "5", "first", "second", "third"])):
                self.logger.info("[ROUTING] → query_services (user selecting service, staying in query_services)")
                return "query_services"
            
            # If user is trying to select a branch option, don't route to query_branches
            if (state.get("branch_options") and not state.get("branch_id") and 
                any(word in content for word in ["1", "2", "3", "4", "5", "first", "second", "third"])):
                self.logger.info("[ROUTING] → query_branches (user selecting branch, staying in query_branches)")
                return "query_branches"
            
            # Check for specific requests only if not in selection mode
            if not ((state.get("branch_options") and not state.get("branch_id")) or 
                    (state.get("service_options") and not state.get("service_id"))):
                if "branch" in content or "salon" in content or "location" in content:
                    self.logger.info("[ROUTING] → query_branches (user asked for branches)")
                    return "query_branches"
                elif "service" in content or "treatment" in content:
                    self.logger.info("[ROUTING] → query_services (user asked for services)")
                    return "query_services"
                
        # Logic based on missing information
        if not state.get("branch_id"):
            self.logger.info("[ROUTING] → query_branches (no branch_id)")
            return "query_branches"
        elif not state.get("service_id"):
            self.logger.info("[ROUTING] → query_services (no service_id)")
            return "query_services"
        elif not state.get("customer_name") or not state.get("customer_mobile"):
            self.logger.info("[ROUTING] → collect_customer_info (missing customer info)")
            return "collect_customer_info"
        else:
            self.logger.info("[ROUTING] → confirm_details (all info collected)")
            return "confirm_details"
    
    def _route_user_intent(self, state: BookingState) -> str:
        """Route based on user's detected intent."""
        intent = state.get("last_user_intent", "provide_info")
        
        self.logger.info(f"[INTENT_ROUTING] Intent: {intent}")
        
        if intent == "show_summary":
            return "show_summary"
        elif intent == "edit":
            return "handle_edit"
        elif intent == "start_over":
            return "start_over"
        elif intent == "confirm":
            return "confirm_details"
        elif intent == "cancel":
            return "end"
        else:
            # Default: continue with extraction
            return "extract_info"
    
    def _route_after_edit(self, state: BookingState) -> str:
        """Route after handling edit command."""
        current_step = state.get("current_step", "")
        
        if current_step == "query_branches":
            return "query_branches"
        elif current_step == "query_services":
            return "query_services"
        elif current_step == "collect_customer_info":
            return "collect_customer_info"
        else:
            return "show_summary"
    
    def _check_if_customer_info_collected(self, state: BookingState) -> str:
        """Check if customer information has been collected and determine next step."""
        try:
            # Check if we have all required customer information
            has_name = bool(state.get("customer_name"))
            has_mobile = bool(state.get("customer_mobile"))
            
            self.logger.info(f"[CUSTOMER_INFO] Name: {has_name}, Mobile: {has_mobile}")
            
            # Get the latest assistant message to see if we're asking for customer info
            latest_assistant_message = next((msg for msg in reversed(state["conversation_history"]) 
                                        if msg["role"] == "assistant"), None)
            
            if latest_assistant_message:
                content = latest_assistant_message["content"]
                
                # If we've asked for customer info and are waiting for response
                if ("I need your" in content and ("name" in content or "phone" in content)):
                    self.logger.info("[CUSTOMER_INFO] → end (waiting for customer info)")
                    return "end"
            
            # Always end - either we have the info or we're waiting for it
            self.logger.info("[CUSTOMER_INFO] → end")
            return "end"
        
        except Exception as e:
            self.logger.error(f"Error in _check_if_customer_info_collected: {str(e)}")
            return "end"
    
    def _check_booking_completeness(self, state: BookingState) -> str:
        """Check if the booking information is complete."""
        # Update missing fields
        update_missing_fields(state)
        
        # Add debugging
        self.logger.info(f"[COMPLETENESS] Missing fields: {state.get('missing_fields', [])}")
        
        # Check for confirmation
        latest_user_message = next((msg for msg in reversed(state["conversation_history"]) 
                                   if msg["role"] == "user"), None)
        
        if latest_user_message:
            user_content = latest_user_message["content"].lower()
            self.logger.info(f"[COMPLETENESS] User message: {user_content}")
            
            if "confirm" in user_content:
                # If all required fields are provided, proceed to create booking
                if not state["missing_fields"]:
                    self.logger.info("[COMPLETENESS] → create_booking (user confirmed and all fields complete)")
                    return "create_booking"
                else:
                    self.logger.info("[COMPLETENESS] → end (user confirmed but missing fields)")
                    return "end"
            else:
                self.logger.info("[COMPLETENESS] → end (user didn't confirm)")
                return "end"
        else:
            self.logger.info("[COMPLETENESS] → end (no user message)")
            return "end"
    
    def _check_booking_success(self, state: BookingState) -> str:
        """Check if the booking was successful."""
        if state.get("is_complete", False):
            return "end"
        else:
            return "handle_error"
    
    # Public API
    def start_workflow(self, user_query: str, initial_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """Start the booking workflow with an initial user query."""
        # Initialize state
        initial_state = initialize_booking_state()
        # Pre-populate with extracted information
        if initial_info:
            for key, value in initial_info.items():
                if value:  # Only set non-empty values
                    initial_state[key] = value
        # Add user message            
        add_message(initial_state, "user", user_query)
        
        # Create configuration with thread_id for memory persistence
        config = {
            "configurable": {
                "thread_id": initial_state.get("session_id", str(uuid.uuid4()))
            }
        }
        
        # Run the workflow with config
        result = self.workflow.invoke(initial_state, config=config)
        
        # Get the final response
        last_message = next((msg["content"] for msg in reversed(result["conversation_history"]) 
                            if msg["role"] == "assistant"), None)
        
        return {
            "state": result,
            "response": last_message,
            "is_complete": result.get("is_complete", False)
        }
    
    def continue_workflow(self, state: Dict[str, Any], user_message: str) -> Dict[str, Any]:
        """Continue an existing workflow with a new user message."""
        # Convert dict to BookingState if needed
        if not isinstance(state, dict):
            current_state = state
        else:
            current_state = state
        
        # Add user message
        add_message(current_state, "user", user_message)
        
        # Always restart from extract_info to process the new user input
        current_state["current_step"] = "extract_info"
        
        # Create configuration with thread_id for memory persistence
        config = {
            "configurable": {
                "thread_id": current_state.get("session_id", str(uuid.uuid4()))
            }
        }
        
        # Run the workflow with config
        result = self.workflow.invoke(current_state, config=config)
        
        # Get the latest response
        last_message = next((msg["content"] for msg in reversed(result["conversation_history"]) 
                            if msg["role"] == "assistant"), None)
        
        return {
            "state": result,
            "response": last_message,
            "is_complete": result.get("is_complete", False)
        }
    
    @staticmethod
    def convert_to_api_date(date_str: str) -> Optional[str]:
        """Convert user-friendly date to yyyy-MM-dd format for API."""
        if not date_str:
            return None
        dt = dateparser.parse(date_str)
        if dt:
            return dt.strftime("%Y-%m-%d")
        return None

    @staticmethod
    def convert_to_api_time(time_str: str) -> Optional[str]:
        """Convert user-friendly time to HH:mm format for API."""
        if not time_str:
            return None
        dt = dateparser.parse(time_str)
        if dt:
            return dt.strftime("%H:%M")
        return None


# Create a global instance for easy import
booking_workflow = BookingWorkflow()
