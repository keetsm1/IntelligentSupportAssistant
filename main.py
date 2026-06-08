from src.search import Search
from src.sentaiment_analysis import analyze_sentiment
from src.escalation import should_escalate, get_escalation_message


def main():
    print("Welcome to Student Support AI")
    print("Type 'quit' to exit.")

    conversation_history = []

    while True:
        query = input("\nYou: ").strip()

        if query.lower() == "quit":
            print("Goodbye!")
            break

        if query == "":
            print("Please enter a question or type 'quit' to exit.")
            continue

        sentiment = analyze_sentiment(query)
        label = sentiment["label"]
        confidence = sentiment["score"]

        search = Search(query)
        result = search.conduct_semantic_search()

        print(f"Sentiment: {label} ({confidence:.2f})")

        if should_escalate(sentiment):
            print(get_escalation_message())

        print(f"Answer: {result['answer']}")
        print(f"Similarity score: {result['score']:.4f}")

        conversation_history.append({
            "query": query,
            "sentiment": label,
            "confidence": confidence,
            "answer": result["answer"],
            "similarity_score": result["score"]
        })


if __name__ == "__main__":
    main()
