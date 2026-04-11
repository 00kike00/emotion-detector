import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent.parent
if str(root) not in sys.path:
    sys.path.append(str(root))

import time
from src.llm_wrapper.llm_wrapper import LLMWrapper

def test_interactive_session():
    print("--- Starting Interactive Test of the LLM Wrapper ---")
    print("Type 'quit' to exit or 'reset' to clear conversation history.\n")
    
    # 1. Initialize
    try:
        llm = LLMWrapper()
        print(f"Loaded model: {llm.model}")
    except Exception as e:
        print(f"Error initializing LLM: {e}")
        return

    # 2. Continuous Loop for messaging
    while True:
        user_input = input("\nUser: ")
        
        # Exit conditions
        if user_input.lower() in ['quit', 'exit', 'salir']:
            print("Exiting test session...")
            break
            
        if user_input.lower() == 'reset':
            llm.reset()
            print("Conversation history has been cleared.")
            continue

        # Simulate the emotional context (this would come from the manager in a real application)
        emotion = "happiness"  # Placeholder emotion
        context = f"The user feels: {emotion}."

        print("-" * 30)
        print(f"Assistant [{emotion}]: ", end="", flush=True)

        # 3. Measure time and execute stream
        start_time = time.time()
        first_token_time = None
        
        try:
            for token in llm.stream(user_input, emotion_context=context):
                if first_token_time is None:
                    first_token_time = time.time() - start_time
                print(token, end="", flush=True)
            
            total_time = time.time() - start_time

            # 4. Results for performance monitoring
            print("\n" + "-" * 30)
            print(f"Performance Metrics:")
            print(f"  - Time to first token: {first_token_time:.2f}s")
            print(f"  - Total response time: {total_time:.2f}s")
            
        except Exception as e:
            print(f"\nError during streaming: {e}")

    print("--- Quick Test Completed ---")

if __name__ == "__main__":
    test_interactive_session()