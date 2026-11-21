"""Transactional PostgreSQL experiment catalog. Only stored research runs are imported."""

import argparse
import hashlib
import os
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import (
    JSON, CheckConstraint, DateTime, ForeignKey, Integer, String, UniqueConstraint,
    create_engine, select,
)
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.engine import make_url
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from .audit import audit_records
from .dataset import FEATURES, SOURCE
from .inference import load_trusted_model
from .saved_run import digest, verify_run


class Base(DeclarativeBase):
    pass


class DatasetVersion(Base):
    __tablename__ = "dataset_versions"

    checksum: Mapped[str] = mapped_column(String(64), primary_key=True)
    source: Mapped[str] = mapped_column(String)
    row_count: Mapped[int] = mapped_column(Integer)
    feature_count: Mapped[int] = mapped_column(Integer)
    audit: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class TrainingRun(Base):
    __tablename__ = "training_runs"

    fingerprint: Mapped[str] = mapped_column(String(64), primary_key=True)
    dataset_checksum: Mapped[str] = mapped_column(ForeignKey("dataset_versions.checksum"), index=True)
    algorithm: Mapped[str] = mapped_column(String)
    seed: Mapped[int] = mapped_column(Integer)
    configuration: Mapped[dict] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String)
    completed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class ModelResult(Base):
    __tablename__ = "model_results"
    __table_args__ = (
        UniqueConstraint("run_id", "evaluation", "algorithm", name="uq_model_result"),
        CheckConstraint("threshold >= 0 AND threshold <= 1", name="ck_result_threshold"),
        CheckConstraint("evaluation IN ('validation', 'holdout')", name="ck_result_evaluation"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("training_runs.fingerprint"), index=True)
    evaluation: Mapped[str] = mapped_column(String)
    algorithm: Mapped[str] = mapped_column(String)
    threshold: Mapped[float]
    metrics: Mapped[dict] = mapped_column(JSON)
    confusion_matrix: Mapped[list] = mapped_column(JSON)


class ModelFile(Base):
    __tablename__ = "model_artifacts"

    run_id: Mapped[str] = mapped_column(ForeignKey("training_runs.fingerprint"), primary_key=True)
    version: Mapped[str] = mapped_column(String)
    location: Mapped[str] = mapped_column(String)
    checksum: Mapped[str] = mapped_column(String(64))
    metadata_json: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


def engine_from_url(url):
    if not url or make_url(url).drivername != "postgresql+psycopg":
        raise ValueError("Set DATABASE_URL to a postgresql+psycopg URL")
    return create_engine(url, pool_pre_ping=True, connect_args={"connect_timeout": 3},
                         hide_parameters=True)


def fingerprint(directory):
    names = ("selection.json", "validation.json", "heldout.json", "model.joblib")
    return hashlib.sha256("".join(digest(directory / name) for name in names).encode()).hexdigest()


def register(engine, directory, data, manifest):
    selection, validation, holdout, records = verify_run(directory, data, manifest)
    path = directory / "model.joblib"
    model = load_trusted_model(path)
    if model.metadata != selection:
        raise ValueError("Trusted model metadata does not match the selection report")

    checksum = selection["dataset_sha256"]
    key = fingerprint(directory)
    completed = datetime.fromtimestamp((directory / "heldout.json").stat().st_mtime, timezone.utc)
    imported = datetime.now(timezone.utc)
    with Session(engine) as session, session.begin():
        session.execute(insert(DatasetVersion).values(
            checksum=checksum, source=SOURCE, row_count=len(records), feature_count=len(FEATURES),
            audit=audit_records(records, checksum), created_at=imported,
        ).on_conflict_do_nothing(index_elements=["checksum"]))
        registered = session.execute(insert(TrainingRun).values(
            fingerprint=key, dataset_checksum=checksum, algorithm=selection["algorithm"],
            seed=selection["seed"], configuration=selection, status="completed", completed_at=completed,
        ).on_conflict_do_nothing(index_elements=["fingerprint"]).returning(TrainingRun.fingerprint))
        if registered.scalar_one_or_none() is None:
            return key, False

        for name, result in validation["outer_comparison"].items():
            session.add(ModelResult(
                run_id=key, evaluation="validation", algorithm=name,
                threshold=validation["full_training_configurations"][name]["threshold"],
                metrics=result["summary"], confusion_matrix=result["confusion_matrix"],
            ))
        session.add(ModelResult(
            run_id=key, evaluation="holdout", algorithm=selection["algorithm"],
            threshold=selection["threshold"], metrics=holdout["metrics"],
            confusion_matrix=holdout["metrics"]["confusion_matrix"],
        ))
        session.add(ModelFile(
            run_id=key, version=selection["model_version"], location=path.as_posix(),
            checksum=digest(path), metadata_json=selection, created_at=imported,
        ))
    return key, True


def read_results(engine, key):
    with Session(engine) as session:
        rows = session.scalars(select(ModelResult).where(ModelResult.run_id == key)).all()
        return {(row.evaluation, row.algorithm): row.metrics for row in rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("artifacts/milestone-2"))
    parser.add_argument("--data", type=Path, default=Path("data/processed.cleveland.data"))
    parser.add_argument("--manifest", type=Path, default=Path("data/split.json"))
    options = parser.parse_args()
    engine = None
    try:
        engine = engine_from_url(os.getenv("DATABASE_URL"))
        key, created = register(engine, options.run, options.data, options.manifest)
    except SQLAlchemyError as exc:
        raise SystemExit(f"Database registration failed ({type(exc).__name__}). Check connectivity and migrations.") from None
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from None
    finally:
        if engine is not None:
            engine.dispose()
    print(f"{'Registered' if created else 'Already registered'} {key}")


if __name__ == "__main__":
    main()
