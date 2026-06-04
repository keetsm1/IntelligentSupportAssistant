from src.search import Search

def main():
    while (True):
        print("enter Query:")
        query = input()

        search = Search(query)

        print(search.conduct_semantic_search())


if __name__=="__main__":
    main()