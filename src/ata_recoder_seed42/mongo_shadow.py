"""Score recent Mongo records into a separate seed-42 shadow field."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

from pymongo import MongoClient, UpdateOne
from tqdm import tqdm

from .inference import Seed42Predictor, resolve_device


def mongo_uri(uri_file: str | None) -> str:
    value = os.getenv("MONGODB_URI", "").strip()
    if value:
        return value
    if uri_file:
        value = Path(uri_file).read_text(encoding="utf-8").strip()
    if not value:
        raise RuntimeError("Set MONGODB_URI or provide --mongo-uri-file")
    if value.startswith(("mongodb://", "mongodb+srv://")):
        return value

    username = quote_plus(os.getenv("MONGODB_USER", "OR_DTFPTOWR_APP"))
    password = quote_plus(value)
    host = os.getenv("MONGODB_HOST", "techopsbt-dev-pl-0.gjbpj.mongodb.net")
    database = quote_plus(os.getenv("MONGODB_DATABASE", "DTFPTOWR"))
    return (
        f"mongodb+srv://{username}:{password}@{host}/{database}"
        "?retryWrites=true&w=majority&connectTimeoutMS=300000"
        "&maxIdleTimeMS=300000&serverSelectionTimeoutMS=300000"
        "&maxPoolSize=60&appName=ata-recorder-seed42"
    )


def prediction_document(predictions: list[Any], timestamp: datetime) -> dict[str, Any]:
    result: dict[str, Any] = {
        "autoRecode": False,
        "qaReview": True,
        "predDateTimeUTC": timestamp,
        "modelVersion": "dapt20-seed42-v1",
        "modelFamily": "DeBERTa-DAPT20-Hierarchical",
    }
    for rank in range(1, 6):
        prediction = predictions[rank - 1]
        result[f"ataPred{rank}"] = str(prediction.code).zfill(4)
        result[f"predProb{rank}"] = float(prediction.score)
    return result


def build_query(days: int, shadow_field: str, text_field: str) -> dict[str, Any]:
    return {
        "flightDate": {"$gte": datetime.now(timezone.utc) - timedelta(days=days)},
        "techopsDelays.code": {"$exists": True},
        text_field: {"$type": "string"},
        shadow_field: {"$exists": False},
    }


def score_records(
    collection: Any,
    predictor: Seed42Predictor,
    *,
    days: int,
    limit: int | None,
    batch_size: int,
    text_field: str,
    shadow_field: str,
    write: bool,
    output_jsonl: str | None,
) -> tuple[int, int]:
    query = build_query(days, shadow_field, text_field)
    cursor = collection.find(query, {"_id": 1, text_field: 1}).sort("flightDate", -1)
    if limit is not None:
        cursor = cursor.limit(limit)

    operations: list[UpdateOne] = []
    scored = 0
    written = 0
    output_handle = None
    try:
        if output_jsonl:
            output_path = Path(output_jsonl)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_handle = output_path.open("w", encoding="utf-8")
        for record in tqdm(cursor, desc="Scoring eligible Mongo records"):
            predictions = predictor.predict(record.get(text_field))
            scored += 1
            stamp = datetime.now(timezone.utc)
            if output_handle is not None:
                result_row = {
                    "record_id": str(record["_id"]),
                    "predictions": [
                        {"code": item.code, "score": item.score}
                        for item in predictions
                    ],
                    "model_version": "dapt20-seed42-v1",
                }
                output_handle.write(json.dumps(result_row, ensure_ascii=True) + "\n")
            if write:
                document = prediction_document(predictions, stamp)
                operations.append(UpdateOne({"_id": record["_id"]}, {"$set": {shadow_field: document}}))
                if len(operations) >= batch_size:
                    result = collection.bulk_write(operations, ordered=False)
                    written += result.modified_count
                    operations.clear()

        if operations:
            result = collection.bulk_write(operations, ordered=False)
            written += result.modified_count
    finally:
        if output_handle is not None:
            output_handle.close()
    return scored, written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, help="Seed42 artifact directory containing best_model and label_maps.json")
    parser.add_argument("--encoder-config-dir", required=True, help="DAPT encoder directory containing config.json")
    parser.add_argument("--mongo-uri-file")
    parser.add_argument("--database", default=os.getenv("MONGODB_DATABASE", "DTFPTOWR"))
    parser.add_argument("--collection", default=os.getenv("MONGODB_COLLECTION", "flights"))
    parser.add_argument("--text-field", default="reason")
    parser.add_argument("--shadow-field", default="ataRecoderMLSeed42")
    parser.add_argument("--days", type=int, default=10)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--output-jsonl", help="Optional privacy-minimized output with record IDs and top-five predictions")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--write", action="store_true", help="Write shadow results. Without this flag the run is read-only.")
    args = parser.parse_args()

    if args.days < 1 or args.batch_size < 1 or (args.limit is not None and args.limit < 1):
        parser.error("days, batch-size, and any supplied limit must be positive")

    device = resolve_device(args.device)
    predictor = Seed42Predictor(
        args.model_dir, args.encoder_config_dir, device=device
    )
    client = MongoClient(mongo_uri(args.mongo_uri_file), serverSelectionTimeoutMS=30000)
    try:
        client.admin.command("ping")
        collection = client[args.database][args.collection]
        scored, written = score_records(
            collection,
            predictor,
            days=args.days,
            limit=args.limit,
            batch_size=args.batch_size,
            text_field=args.text_field,
            shadow_field=args.shadow_field,
            write=args.write,
            output_jsonl=args.output_jsonl,
        )
        print(
            f"scored={scored} written={written} write_enabled={args.write} "
            f"field={args.shadow_field} device={device}",
            flush=True,
        )
    finally:
        client.close()


if __name__ == "__main__":
    main()