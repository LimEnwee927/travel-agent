from agent import TravelAgent, load_long_term_memory, DEFAULT_SESSION

agent = TravelAgent()
session_id = DEFAULT_SESSION
print("Travel Agent ready! Type 'quit' to exit, 'memory' to see saved preferences.\n")

while True:
    user = input("You: ")
    if user.lower() == "quit":
        break
    elif user.lower() == "memory":
        memories = load_long_term_memory(session_id)
        if memories:
            print("\n📝 Saved preferences:")
            for m in memories:
                print(f"  - {m['text']}")
        else:
            print("No memories saved yet.")
        print()
        continue

    result = agent.generate_trip(user, session_id=session_id)
    if result.get("plan"):
        print("\n📋 Plan:")
        for i, step in enumerate(result["plan"], 1):
            print(f"  {i}. {step}")
    print("\n" + result["reply"] + "\n")