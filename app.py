from agent import TravelAgent, load_long_term_memory

agent = TravelAgent()
print("Travel Agent ready! Type 'quit' to exit, 'memory' to see saved preferences.\n")

while True:
    user = input("You: ")
    if user.lower() == "quit":
        break
    elif user.lower() == "memory":
        memories = load_long_term_memory()
        if memories:
            print("\n📝 Saved preferences:")
            for m in memories:
                print(f"  - {m['text']}")
        else:
            print("No memories saved yet.")
        print()
        continue

    answer = agent.generate_trip(user)
    print("\n" + answer + "\n")