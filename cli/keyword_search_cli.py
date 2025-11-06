#!/usr/bin/env python3

import argparse
from lib.keyword_search import search_command, calc_idf
from lib.index import InvertedIndex
from lib.files import load_file

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    subparsers.add_parser("build", help="Build movie search index")

    tf_parser = subparsers.add_parser("tf", help="Print term frequency for given term in given document ID")
    tf_parser.add_argument("doc_id", type=int)
    tf_parser.add_argument("term", type=str)

    idf_parser = subparsers.add_parser("idf")
    idf_parser.add_argument("term", type=str)

    tfidf_parser = subparsers.add_parser("tfidf")
    tfidf_parser.add_argument("doc_id", type=int)
    tfidf_parser.add_argument("term", type=str)

    args = parser.parse_args()

    index = InvertedIndex()

    match args.command:
        case "search":
            if index.load():
                return
            print(f"Searching for: {args.query}")
            results = search_command(index, args.query)
            for i, res in enumerate(results, 1):
                print(f"{i}. {res['title']}")

        case "tf":
            if index.load():
                return 
            tf = index.get_tf(args.doc_id, args.term)
            print(tf)

        case "idf":
            if index.load():
                return 
            idf_value = calc_idf(index, args.term)
            print(f"Inverse document frequency of '{args.term}': {idf_value:.2f}")

        case "tfidf":
            if index.load():
                return 
            tf = index.get_tf(args.doc_id, args.term)
            idf = calc_idf(index, args.term)
            tf_idf = tf * idf

            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}")

        case "build":
            index.build(load_file("movies.json"))
            index.save()
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()