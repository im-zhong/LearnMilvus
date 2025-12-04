# 2025/12/3
# zhangzhong
# https://milvus.io/docs/manage_databases.md

from pymilvus import MilvusClient

client = MilvusClient(
    uri="http://localhost:19530",
    # Milvus does NOT enable authentication by default.
    # So in your Docker Compose setup, there is no token, because Milvus is running without RBAC / auth.
    # token="root:Milvus"
    # If you want to enable RBAC, in docker compose, add: MILVUS_ENABLE_AUTH=true
    # then you could login with:
    # username: root
    # password: Milvus
    # token: "root:Milvus"
)

client.create_database(db_name="my_database_1")

client.create_database(
    db_name="my_database_2", properties={"database.replica.number": 3}
)


## View databases

# List all existing databases
client.list_databases()

# Output
# ['default', 'my_database_1', 'my_database_2']

# Check database details
client.describe_database(db_name="default")

# Output
# {"name": "default"}

## Alter, Drop, Use
