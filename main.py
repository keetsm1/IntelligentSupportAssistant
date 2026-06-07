from src.search import Search
from src.sentaiment_analysis import analyze_sentiment


def main():
    while True:
        print("enter Query:")
        query = input()

        search = Search(query)
        result = search.conduct_semantic_search()

        print(f"Answer: {result['answer']}")
        print(f"Score: {result['score']:.4f}")
        print(f"Sentiment: {analyze_sentiment(query)}")


if __name__ == "__main__":
    main()
