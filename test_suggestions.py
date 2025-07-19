#!/usr/bin/env python3
"""
Test script to verify the get_suggestion function works correctly.
"""
import sys
import os
sys.path.append('.')

# Set up environment
os.environ['FLASK_ENV'] = 'development'

from src.chatbot import langchain_chatbot

def test_get_suggestion():
    print("Testing get_suggestion function...")
    print("="*60)
    
    # Test cases with different question types
    test_questions = [
        "I want to book an appointment",
        "What hair services do you offer?",
        "How much does a facial cost?",
        "I need skincare advice",
        "Where are you located?",
        "What's good for acne treatment?",
        "Can you recommend a hairstyle?",
        "Tôi muốn đặt lịch hẹn",  # Vietnamese
        "Hello, how are you?",  # General question
        "What beauty services do you provide?"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{i}. Question: '{question}'")
        print("-" * 50)
        
        try:
            suggestions = langchain_chatbot.get_suggestion(question)
            print(f"Suggestions ({len(suggestions)}):")
            for j, suggestion in enumerate(suggestions, 1):
                print(f"   {j}. {suggestion}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n" + "="*60)
    print("✅ get_suggestion function test completed!")

if __name__ == "__main__":
    test_get_suggestion()
