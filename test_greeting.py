#!/usr/bin/env python3
"""
Test script to verify the greeting function works correctly.
"""
import sys
import os
sys.path.append('.')

# Set up environment
os.environ['FLASK_ENV'] = 'development'

from src.chatbot import langchain_chatbot

def test_greeting():
    print("Testing EasySalon greeting function...")
    print("="*50)
    
    try:
        # Test greeting
        langchain_chatbot.greeting()
        print("\n" + "="*50)
        print("✅ Greeting function executed successfully!")
        
        # Test basic question answering
        print("\nTesting basic question answering...")
        response = langchain_chatbot.handle_question("What services do you offer?")
        print(f"Response: {response}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_greeting()
    sys.exit(0 if success else 1)
