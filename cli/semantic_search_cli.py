#!/usr/bin/env python3

import argparse
from lib.semantic_search_helpers import embed_text, verify_embeddings, embed_query_text, chunk_text, semantic_chunk_text
from lib.semantic_search import verify_model, SemanticSearch

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="Verify model information")
    subparsers.add_parser("verify_embeddings", help="Embed text into vector array")

    embed_query_parser = subparsers.add_parser("embedquery", help="Embed query into vector")
    embed_query_parser.add_argument("query", type=str)

    embed_parser = subparsers.add_parser("embed_text", help="Embed text into vector")
    embed_parser.add_argument("text", type=str)

    search_parser = subparsers.add_parser("search")
    search_parser.add_argument("query", type=str)
    search_parser.add_argument("--limit", type=int, default=5)

    chunk_parser = subparsers.add_parser("chunk")
    chunk_parser.add_argument("text", type=str)
    chunk_parser.add_argument("--chunk-size", type=int, default=200)
    chunk_parser.add_argument("--overlap", type=int, default=0)

    chunk_parser = subparsers.add_parser("semantic_chunk")
    chunk_parser.add_argument("text", type=str)
    chunk_parser.add_argument("--max-chunk-size", type=int, default=4)
    chunk_parser.add_argument("--overlap", type=int, default=0)


    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()

        case "verify_embeddings":
            verify_embeddings()

        case "embedquery":
            embed_query_text(args.query)
        
        case "embed_text":
            embed_text(args.text)

        case "search":
            model = SemanticSearch()
            results = model.search(args.query, args.limit)
            for i in range(len(results)):
                if i == 0:
                    continue
                print(f"{i}. {results[i]['title']} (score: {results[i]['score']:.4f})")
                print(f"   {results[i]['description']}")
                print()

        case "chunk":
            chunks = chunk_text(args.text, args.chunk_size, args.overlap)
            print(f"Chunking {len(args.text)} characters")
            
            for i in range(len(chunks)):
                print(f"{i + 1}. {chunks[i]}")

        case "semantic_chunk":
            chunks = semantic_chunk_text(args.text, args.max_chunk_size, args.overlap)
            print(f"Semantically chunking {len(args.text)} characters")

            for i in range(len(chunks)):
                print(f"{i + 1}. {chunks[i]}")

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()