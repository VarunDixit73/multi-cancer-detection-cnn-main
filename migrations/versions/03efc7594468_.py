"""empty message

Revision ID: 03efc7594468
Revises: 
Create Date: 2022-04-10 02:52:02.976664

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '03efc7594468'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('info',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('name', sa.String(length=50), nullable=True),
    sa.Column('category', sa.String(length=10), nullable=True),
    sa.Column('location', sa.String(), nullable=True),
    sa.Column('created_on', sa.DateTime(), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_info_created_on'), 'info', ['created_on'], unique=False)
    op.create_table('message_data',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('message', sa.String(), nullable=True),
    sa.Column('prediction', sa.Integer(), nullable=True),
    sa.Column('created_on', sa.DateTime(), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_message_data_created_on'), 'message_data', ['created_on'], unique=False)
    op.create_table('user',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('username', sa.String(length=64), nullable=True),
    sa.Column('email', sa.String(length=120), nullable=True),
    sa.Column('password_hash', sa.String(length=128), nullable=True),
    sa.Column('created_on', sa.DateTime(), nullable=True),
    sa.Column('about_me', sa.String(length=140), nullable=True),
    sa.Column('last_seen', sa.DateTime(), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_user_created_on'), 'user', ['created_on'], unique=False)
    op.create_index(op.f('ix_user_email'), 'user', ['email'], unique=True)
    op.create_index(op.f('ix_user_username'), 'user', ['username'], unique=True)
    op.create_table('my_upload',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('img', sa.String(length=255), nullable=True),
    sa.Column('imgtype', sa.String(length=4), nullable=True),
    sa.Column('user_id', sa.Integer(), nullable=True),
    sa.Column('created_on', sa.DateTime(), nullable=True),
    sa.ForeignKeyConstraint(['user_id'], ['user.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_my_upload_created_on'), 'my_upload', ['created_on'], unique=False)
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_index(op.f('ix_my_upload_created_on'), table_name='my_upload')
    op.drop_table('my_upload')
    op.drop_index(op.f('ix_user_username'), table_name='user')
    op.drop_index(op.f('ix_user_email'), table_name='user')
    op.drop_index(op.f('ix_user_created_on'), table_name='user')
    op.drop_table('user')
    op.drop_index(op.f('ix_message_data_created_on'), table_name='message_data')
    op.drop_table('message_data')
    op.drop_index(op.f('ix_info_created_on'), table_name='info')
    op.drop_table('info')
    # ### end Alembic commands ###
