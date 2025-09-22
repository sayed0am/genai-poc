#!/usr/bin/env python
# src/latest_ai_development/main.py
import sys
from crew import RagCrew

def run():
    """
    Run the crew.
    """
    inputs = {
    'query': "For a non-citizen employee with 12 years of service, how is their end-of-service gratuity calculated?"
    }
    result = RagCrew().crew().kickoff(inputs=inputs)


    print("\n--------------------------------------------------")
    print("✅ Crew execution finished. Here is the final answer:")
    print(result)


run()