"""empty message

Revision ID: 6845617bc15f
Revises: 64fe0918d5c4
Create Date: 2023-10-11 22:31:19.827418

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '6845617bc15f'
down_revision = '64fe0918d5c4'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table('_alembic_tmp_course')
    with op.batch_alter_table('course', schema=None) as batch_op:
        batch_op.add_column(sa.Column('professor_id', sa.Integer(), server_default='1', nullable=True))
        batch_op.create_foreign_key(batch_op.f('fk_course_professor_id_user'), 'user', ['professor_id'], ['id'])

    with op.batch_alter_table('user', schema=None) as batch_op:
        batch_op.create_unique_constraint(batch_op.f('uq_user_email'), ['email'])

    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('user', schema=None) as batch_op:
        batch_op.drop_constraint(batch_op.f('uq_user_email'), type_='unique')

    with op.batch_alter_table('course', schema=None) as batch_op:
        batch_op.drop_constraint(batch_op.f('fk_course_professor_id_user'), type_='foreignkey')
        batch_op.drop_column('professor_id')

    op.create_table('_alembic_tmp_course',
    sa.Column('id', sa.INTEGER(), nullable=False),
    sa.Column('course_name', sa.VARCHAR(length=200), nullable=False),
    sa.Column('image_path', sa.VARCHAR(length=255), nullable=False),
    sa.Column('professor_id', sa.INTEGER(), nullable=False),
    sa.ForeignKeyConstraint(['professor_id'], ['user.id'], name='fk_course_professor_id_user'),
    sa.PrimaryKeyConstraint('id')
    )
    # ### end Alembic commands ###
