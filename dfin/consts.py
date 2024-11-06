# BIRD dataset
BIRD_DATASET_DIR = "dev"
BIRD_DEV_JSON_PATH = f"{BIRD_DATASET_DIR}/dev.json"
BIRD_DEV_DATABASES_PATH = f"{BIRD_DATASET_DIR}/dev_databases"
BIRD_DATABASE_DB_DESCRIPTION_DIR = "database_description"


# Preprocessing
DB_PREPROCESSING_DIR = "db_preprocessing"
PREPROCESSING_DEV_DB_EMBEDDINGS_PATH = f'{DB_PREPROCESSING_DIR}/column_description_embeddings'
PREPROCESSING_DEV_TABLE_DESCRIPTIONS = f'{DB_PREPROCESSING_DIR}/tables_description.json'
BIRD_DEV_SUBSET_JSON = f"{DB_PREPROCESSING_DIR}/dev_subset.json"


# Output
OUTPUT_PREDICT_JSON = "output/predict_dev.json"
LOGS_PATH = "output/logs.csv"
LINKED_TABLES_DIR = "output/linked_tables"
LINKED_COLUMNS_DIR = "output/linked_columns"
QUESTION_EMBEDDINGS_PATH = "output/question_embeddings.json"

MINIMAL = "minimal"
BALANCED = "balanced"
CONSERVATIVE = "conservative"

LINK_TABLE_MODES = [
    MINIMAL,
    BALANCED,
    CONSERVATIVE
]