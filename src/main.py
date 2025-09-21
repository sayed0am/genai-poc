#!/usr/bin/env python
# src/latest_ai_development/main.py
import sys
from crew import RagCrew

def run():
    """
    Run the crew.
    """
    inputs = {
    'query': "What is the maximum lump sum financial reward, in Dirhams, that a military retiree in the 'First' main grade can receive upon appointment?"
    }
    result = RagCrew().crew().kickoff(inputs=inputs)


    print("\n--------------------------------------------------")
    print("✅ Crew execution finished. Here is the final answer:")
    print(result)


run()