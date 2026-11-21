"""Create the experiment catalog."""

from alembic import op
import sqlalchemy as sa

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "dataset_versions",
        sa.Column("checksum", sa.String(64), primary_key=True),
        sa.Column("source", sa.String(), nullable=False),
        sa.Column("row_count", sa.Integer(), nullable=False),
        sa.Column("feature_count", sa.Integer(), nullable=False),
        sa.Column("audit", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_table(
        "training_runs",
        sa.Column("fingerprint", sa.String(64), primary_key=True),
        sa.Column("dataset_checksum", sa.String(64), sa.ForeignKey("dataset_versions.checksum"), nullable=False),
        sa.Column("algorithm", sa.String(), nullable=False),
        sa.Column("seed", sa.Integer(), nullable=False),
        sa.Column("configuration", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_training_runs_dataset_checksum", "training_runs", ["dataset_checksum"])
    op.create_table(
        "model_results",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("run_id", sa.String(64), sa.ForeignKey("training_runs.fingerprint"), nullable=False),
        sa.Column("evaluation", sa.String(), nullable=False),
        sa.Column("algorithm", sa.String(), nullable=False),
        sa.Column("threshold", sa.Float(), nullable=False),
        sa.Column("metrics", sa.JSON(), nullable=False),
        sa.Column("confusion_matrix", sa.JSON(), nullable=False),
        sa.UniqueConstraint("run_id", "evaluation", "algorithm", name="uq_model_result"),
        sa.CheckConstraint("threshold >= 0 AND threshold <= 1", name="ck_result_threshold"),
        sa.CheckConstraint("evaluation IN ('validation', 'holdout')", name="ck_result_evaluation"),
    )
    op.create_index("ix_model_results_run_id", "model_results", ["run_id"])
    op.create_table(
        "model_artifacts",
        sa.Column("run_id", sa.String(64), sa.ForeignKey("training_runs.fingerprint"), primary_key=True),
        sa.Column("version", sa.String(), nullable=False),
        sa.Column("location", sa.String(), nullable=False),
        sa.Column("checksum", sa.String(64), nullable=False),
        sa.Column("metadata_json", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )


def downgrade():
    for table in ("model_artifacts", "model_results", "training_runs", "dataset_versions"):
        op.drop_table(table)
