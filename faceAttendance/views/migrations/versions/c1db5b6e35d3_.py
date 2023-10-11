"""empty message

Revision ID: c1db5b6e35d3
Revises: d6510429ef51
Create Date: 2023-10-12 02:30:15.372952

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'c1db5b6e35d3'
down_revision = 'd6510429ef51'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('user', schema=None) as batch_op:
        batch_op.alter_column('username',
               existing_type=sa.VARCHAR(length=200),
               nullable=False,
               existing_server_default=sa.text("'1'"))

    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('user', schema=None) as batch_op:
        batch_op.alter_column('username',
               existing_type=sa.VARCHAR(length=200),
               nullable=True,
               existing_server_default=sa.text("'1'"))

    # ### end Alembic commands ###
