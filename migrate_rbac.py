"""Migrate existing documents and Qdrant points to Clerk/RBAC metadata.

Existing `private` documents are mapped to `faculty` visibility by default.
Review departments and ownership before enabling student access in production.
"""

from __future__ import annotations

import argparse
import os

from dotenv import load_dotenv
from qdrant_client.models import FieldCondition, Filter, MatchValue

from database import Base, SessionLocal, engine, ensure_rbac_schema
from models_db import DocumentRecord
from vector_db import COLLECTION_NAME, get_qdrant_storage


def main() -> int:
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-private-visibility", choices=("faculty", "admin"), default="faculty")
    args = parser.parse_args()

    Base.metadata.create_all(bind=engine)
    ensure_rbac_schema()
    db = SessionLocal()
    store = get_qdrant_storage()
    try:
        documents = db.query(DocumentRecord).all()
        for document in documents:
            if not document.doc_id:
                document.doc_id = document.source_id
            if document.visibility == "private":
                document.visibility = args.legacy_private_visibility
            if not document.owner_id:
                document.owner_id = document.uploaded_by
        db.commit()

        # Re-tag legacy points by source. New uploads already carry all fields.
        for document in documents:
            source_filter = Filter(must=[FieldCondition(key="source_id", match=MatchValue(value=document.source_id))])
            store.client.set_payload(
                collection_name=COLLECTION_NAME,
                payload={
                    "doc_id": document.doc_id or document.source_id,
                    "owner_id": document.owner_id,
                    "department": document.department,
                    "visibility": document.visibility,
                },
                points=source_filter,
                wait=True,
            )
        print(f"Migrated {len(documents)} document records and their Qdrant payloads")
    finally:
        db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
