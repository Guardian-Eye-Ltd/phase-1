"""create_camera_table

Revision ID: 90eb052ae2a3
Revises: cd30ef5a177c
Create Date: 2026-08-09 21:20:50.191700

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '90eb052ae2a3'
down_revision: Union[str, Sequence[str], None] = 'cd30ef5a177c'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        'cameras',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('name', sa.String(length=100), nullable=False),
        sa.Column('camera_code', sa.String(length=50), nullable=False),
        sa.Column('location', sa.String(length=100), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('protocol', sa.String(length=20), server_default='RTSP', nullable=False),
        sa.Column('stream_url', sa.String(length=500), nullable=False),
        sa.Column('username', sa.String(length=100), nullable=True),
        sa.Column('password', sa.String(length=255), nullable=True),
        sa.Column('is_active', sa.Boolean(), server_default='true', nullable=False),
        sa.Column('status', sa.String(length=20), server_default='OFFLINE', nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('last_seen_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_cameras_id'), 'cameras', ['id'], unique=False)
    op.create_index(op.f('ix_cameras_name'), 'cameras', ['name'], unique=False)
    op.create_index(op.f('ix_cameras_camera_code'), 'cameras', ['camera_code'], unique=True)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f('ix_cameras_camera_code'), table_name='cameras')
    op.drop_index(op.f('ix_cameras_name'), table_name='cameras')
    op.drop_index(op.f('ix_cameras_id'), table_name='cameras')
    op.drop_table('cameras')
