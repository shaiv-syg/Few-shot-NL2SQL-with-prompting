import json
import os
import click
from typing import List, Dict

from consts import *
from utils.azure_openai import get_embedding
from link_columns import get_column_links_for_tables
from link_schema_tables import predict_linked_tables


def load_dataset(file_path: str) -> List[Dict[str, str]]:
    with open(file_path, 'r') as f:
        dataset = json.load(f)
    return dataset


def load_preprocessed_tables_descriptions(file_path: str) -> List[Dict[str, str]]:
    with open(file_path, 'r') as file:
        tables_descriptions = json.load(file)
    return tables_descriptions


def save_results_incrementally(file_path: str, data: dict):
    # Ensure the directory exists
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    # Create the file if it doesn’t exist
    if not os.path.exists(file_path):
        with open(file_path, 'w') as f:
            json.dump([], f)

    with open(file_path, 'r+') as f:
        existing_data = json.load(f)
        question_id = data["question_id"]

        # Check if the question_id already exists in the data
        existing_entry = next((entry for entry in existing_data if entry["question_id"] == question_id), None)

        if existing_entry:
            # Update existing entry
            index = existing_data.index(existing_entry)
            existing_data[index] = data
        else:
            # Add new entry
            existing_data.append(data)

        f.seek(0)
        json.dump(existing_data, f, indent=4)
        f.truncate()


def link_tables_and_persist(
        dataset: List[Dict[str, str]],
        preprocessed_tables_descriptions: List[Dict[str, str]],
        output_file: str,
        mode: str,
        from_question_id: int = 0
):
    for index, row in enumerate(dataset):
        question_id = int(row["question_id"])
        if question_id < from_question_id:
            continue

        print(f"Linking tables for question {index}/{len(dataset)}, question id: {question_id}")
        question = row["question"]
        db_id = row["db_id"]
        hint = row.get("evidence", None)

        linked_tables = predict_linked_tables(
            question=question,
            hint=hint,
            db_id=db_id,
            tables_descriptions=preprocessed_tables_descriptions,
            mode=mode)

        result = {
            "question_id": question_id,
            "linked_tables": linked_tables
        }

        # Save results incrementally
        save_results_incrementally(output_file, result)

    print("Linked tables results persisted to", output_file)


def link_columns_and_persist(dataset: List[Dict[str, str]], linked_tables_results: Dict[str, List[str]],
                             preprocessed_tables_descriptions: List[Dict[str, str]], output_file: str):
    table_columns_dict_results = {}
    for index, row in enumerate(dataset):
        print(f"Linking columns for question {index + 1}/{len(dataset)}")
        db_id = row["db_id"]
        question_embedding = ...  # Add code to generate question embedding

        table_links = linked_tables_results.get(db_id, [])
        table_columns_dict = get_column_links_for_tables(db_id, question_embedding, table_links,
                                                         preprocessed_tables_descriptions)
        table_columns_dict_results[index] = table_columns_dict

    with open(output_file, 'w') as f:
        json.dump(table_columns_dict_results, f)
    print("Table columns dictionary persisted to", output_file)


def compute_and_persist_questions_embeddings(
        dataset: List[Dict[str, str]],
        output_file: str,
        from_question_id: int = 0
):
    embeddings_results = []
    for index, row in enumerate(dataset):
        question_id = int(row["question_id"])
        if question_id < from_question_id:
            continue
        question = row["question"]
        hint = row.get("evidence", "")

        embedding_str = f"{question}"
        if hint:
            embedding_str += f"\thint: {hint}"
        embedding = get_embedding(embedding_str)

        result = {
            "question_id": question_id,
            "embedding": embedding
        }

        embeddings_results.append(result)
        print(f"Computed embedding for question {index}/{len(dataset)}, question id: {question_id}")

        # Save results incrementally
        save_results_incrementally(output_file, result)
    print("Embeddings results persisted to", output_file)


def compute_and_persist_column_links(
    dataset: List[Dict[str, str]],
    embeddings_file: str,
    linked_tables_file: str,
    output_file: str,
    from_question_id: int = 0,
    top_k: int = 15,
):
    # Load precomputed embeddings and linked tables
    with open(embeddings_file, 'r') as f:
        embeddings_data = json.load(f)
    embeddings_dict = {entry["question_id"]: entry["embedding"] for entry in embeddings_data}

    with open(linked_tables_file, 'r') as f:
        linked_tables_data = json.load(f)
    linked_tables_dict = {entry["question_id"]: entry["linked_tables"] for entry in linked_tables_data}

    for index, row in enumerate(dataset):
        question_id = int(row["question_id"])
        if question_id < from_question_id:
            continue

        db_id = row["db_id"]
        embedding = embeddings_dict[question_id]
        table_links = linked_tables_dict[question_id]

        column_links = get_column_links_for_tables(
            db_id=db_id,
            question_embedding=embedding,
            table_links=table_links,
            top_k=top_k
        )

        result = {
            "question_id": question_id,
            "column_links": column_links
        }

        # Save results incrementally
        save_results_incrementally(output_file, result)
        print(f"Computed column links for question {index}/{len(dataset)}, question id: {question_id}")

    print("Column links results persisted to", output_file)


def validate_mode(ctx, param, value):
    steps = ctx.params.get('steps')
    if steps in ['all', 'tables'] and value is None:
        raise click.BadParameter('Mode is required when steps is "all" or "tables"')
    return value


@click.command()
@click.option('--dataset-file-path', default=BIRD_DEV_JSON_PATH, show_default=True, help='Path to the dataset file')
@click.option('--steps', type=click.Choice(['tables', 'embeddings', 'columns', 'all']), required=True, help='Which part to run: tables, columns, or all')
@click.option('--mode', type=click.Choice(LINK_TABLE_MODES), callback=validate_mode, help='Mode for table linking')
@click.option('--from-question-id', default=0, show_default=True, help='Start processing from this question id')
@click.option('--top-k', default=15, show_default=True, type=int, help='Top k columns to consider for linking')
def main(dataset_file_path, steps, mode, from_question_id, top_k):
    # File paths
    preprocessed_tables_descriptions_file_path = PREPROCESSING_DEV_TABLE_DESCRIPTIONS

    # Load dataset and preprocessed tables descriptions
    dataset = load_dataset(dataset_file_path)
    preprocessed_tables_descriptions = load_preprocessed_tables_descriptions(preprocessed_tables_descriptions_file_path)

    # File paths to persist results with mode
    linked_tables_output_file = f"{LINKED_TABLES_DIR}/linked_tables_results_mode_{mode}.json"

    if steps in ['tables', 'all']:
        link_tables_and_persist(
            dataset,
            preprocessed_tables_descriptions,
            linked_tables_output_file,
            mode,
            from_question_id
        )

    if steps in ['embeddings', 'all']:
        compute_and_persist_questions_embeddings(
            dataset,
            QUESTION_EMBEDDINGS_PATH,
            from_question_id
        )

    if steps in ['columns', 'all']:
        column_links_output_file = f"{LINKED_COLUMNS_DIR}/column_links_results_{mode}_top_k_{top_k}.json"
        compute_and_persist_column_links(
            dataset,
            QUESTION_EMBEDDINGS_PATH,
            linked_tables_output_file,
            column_links_output_file,
            from_question_id,
            top_k
        )

    # Columns linking and persisting will be added here later


if __name__ == "__main__":
    main()
