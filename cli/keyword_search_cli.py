#!/usr/bin/env python3

import argparse
from lib.keyword_search import search_command, calc_idf
from lib.index import InvertedIndex, BM25_K1, BM25_B
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

    bm25_idf_parser = subparsers.add_parser("bm25idf", help="Get BM25 IDF score for a given term")
    bm25_idf_parser.add_argument("term", type=str, help="Term to gt BM25 IDF score for")

    bm25_tf_parser = subparsers.add_parser("bm25tf", help="Get BM25 TF score for a given document ID and term")

    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument("k1", type=float, nargs='?', default=BM25_K1, help="Tunable BM25 K1 parameter")
    bm25_tf_parser.add_argument("b", type=float, nargs='?', default=BM25_B, help="Tunable BM25 b parameter")

    bm25search_parser = subparsers.add_parser("bm25search", help="Search movies using full BM25 scoring")
    bm25search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    index = InvertedIndex()

    match args.command:
        case "search":
            if not index.load():
                return
            print(f"Searching for: {args.query}")
            results = search_command(index, args.query)
            for i, res in enumerate(results, 1):
                print(f"{i}. {res['title']}")

        case "bm25search":
            if not index.load():
                return 
            results = index.bm25_search(args.query, 5)
            for i, result in enumerate(results):
                id, score = result
                print(f"{i + 1}. {id} {index.docmap[id]['title']} - Score: {score:.2f}")

        case "tf":
            if not index.load():
                return 
            tf = index.get_tf(args.doc_id, args.term)
            print(tf)

        case "idf":
            if not index.load():
                return 
            idf_value = calc_idf(index, args.term)
            print(f"Inverse document frequency of '{args.term}': {idf_value:.2f}")

        case "tfidf":
            if not index.load():
                return 
            tf = index.get_tf(args.doc_id, args.term)
            idf = calc_idf(index, args.term)
            tf_idf = tf * idf

            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}")

        case "bm25idf":
            if not index.load():
                return 
            bm25idf = index.get_bm25_idf(args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")

        case "bm25tf":
            if not index.load():
                return 
            bm25tf = index.get_bm25_tf(args.doc_id, args.term, args.k1, args.b)
            print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}")

        case "build":
            index.build(load_file("movies.json"))
            index.save()
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()