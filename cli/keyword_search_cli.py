#!/usr/bin/env python3

import argparse
from lib.keyword_search import search_command
from lib.index import InvertedIndex
from lib.files import load_file

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    build_parser = subparsers.add_parser("build", help="Build movie search index")
    search_parser.add_argument("query", type=str, help="Search query")

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
        case "build":
            index.build(load_file("movies.json"))
            index.save()
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()