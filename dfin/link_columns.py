
import json
import os
import re
import ast
from typing import List, Tuple, Dict

import pandas as pd
from langchain.utilities.sql_database import SQLDatabase
from sklearn.metrics.pairwise import cosine_similarity


DB_TABLES_RELATION = None


def load_db_tables_relation(file_path='dev/dev_tables.json'):
    global DB_TABLES_RELATION
    if not DB_TABLES_RELATION:
        try:
            with open(file_path, 'r') as f:
                DB_TABLES_RELATION = json.load(f)
        except FileNotFoundError:
            print("The specified file was not found.")
        except json.JSONDecodeError:
            print("The file is not a valid JSON file.")


def find_primary_and_foreign_keys(db_id, table_name):
    global DB_TABLES_RELATION
    load_db_tables_relation()
    if DB_TABLES_RELATION is None:
        return None

    # Step 2: Find the Database Entry
    db_entry = next((db for db in DB_TABLES_RELATION if db['db_id'] == db_id), None)
    if db_entry is None:
        return None, "The specified database ID was not found."

    # Step 3: Find the Table
    try:
        table_idx = db_entry['table_names_original'].index(table_name)
    except ValueError:
        return None, "The specified table was not found in the database."

    # Step 4: Identify Primary Keys
    pk_list = []
    for pk in db_entry['primary_keys']:
        if isinstance(pk, int):
            table_idx_pk, col_name = db_entry['column_names_original'][pk]
            if table_idx_pk == table_idx:
                pk_list.append(col_name)
        elif isinstance(pk, list):
            composite_pk = []
            for col_idx in pk:
                table_idx_pk, col_name = db_entry['column_names_original'][col_idx]
                if table_idx_pk == table_idx:
                    composite_pk.append(col_name)
            if composite_pk:
                pk_list.extend(composite_pk)

    # Step 5: Identify Foreign Keys
    fk_dict = {}
    for fk, ref in db_entry['foreign_keys']:
        fk_table_idx, fk_col_name = db_entry['column_names_original'][fk]
        ref_table_idx, ref_col_name = db_entry['column_names_original'][ref]
        if table_idx == fk_table_idx:
            ref_table_name = db_entry['table_names_original'][ref_table_idx]
            if ref_table_name not in fk_dict:
                fk_dict[ref_table_name] = set()
            fk_dict[ref_table_name].add(fk_col_name)
        elif table_idx == ref_table_idx:
            fk_table_name = db_entry['table_names_original'][fk_table_idx]
            if fk_table_name not in fk_dict:
                fk_dict[fk_table_name] = set()
            fk_dict[fk_table_name].add(ref_col_name)

    # Convert sets to lists for output
    for key in fk_dict:
        fk_dict[key] = list(fk_dict[key])

    return pk_list, fk_dict


def get_table_create_statement_with_sample(db_uri, table_name):
    db = SQLDatabase.from_uri("sqlite:///"+db_uri)
    db._sample_rows_in_table_info = 0
    return db.get_table_info_no_throw([table_name])


def find_create_table_boundaries(create_table: str):
    # Split the input SQL string into lines
    lines = create_table.strip().split('\n')

    # Flags to check if we are inside the CREATE TABLE statement
    inside_create_table = False
    parenthesis_counter = 0

    # Variables to store the start and end line numbers of the CREATE TABLE statement
    start_line = None
    end_line = None

    # Iterate through each line
    for i, line in enumerate(lines):
        # Check if the line contains the CREATE TABLE statement
        if line.strip().lower().startswith("create table"):
            inside_create_table = True
            start_line = i
            parenthesis_counter = line.count("(") - line.count(")")
        elif inside_create_table:
            # Update the parenthesis counter
            parenthesis_counter += line.count("(") - line.count(")")

            # If the parenthesis counter reaches zero, we have found the end of the CREATE TABLE statement
            if parenthesis_counter == 0:
                end_line = i
                break

    return start_line, end_line


def remove_duplicates(lst):
    result = []
    for item in lst:
        if item not in result:
            result.append(item)
    return result


def filter_columns_within_boundaries(create_table: str, start_line: int, end_line: int, columns_to_keep: List[str]
                                     ) -> Tuple[str, List[str]]:
    lines = create_table.strip().split('\n')
    filtered_lines = lines[start_line:end_line + 1]
    ordered_columns = []
    pattern = re.compile(r'"[^"]+"|\b\w+\b')

    for i in range(start_line + 1, end_line):
        line = lines[i]
        words = pattern.findall(line)
        for word in words:
            stripped_word = word.strip('"')
            if stripped_word in columns_to_keep:
                ordered_columns.append(stripped_word)
                break
        else:
            filtered_lines.remove(line)

    filtered_sql = '\n\t'.join(filtered_lines)
    ordered_columns = remove_duplicates(ordered_columns)
    return filtered_sql, ordered_columns


def get_filtered_sample(db: SQLDatabase, table_name, columns_to_keep) -> str:
    if table_name == 'order':
        table_name = "'order'"
    sample = db._execute(f"SELECT * FROM {table_name} LIMIT 3;")
    values = []
    for row in sample:
        # sample is a list of dicts where each key is the column name
        values.append("\t".join([str(row[col]) for col in columns_to_keep]))

    columns = '\t'.join(columns_to_keep)
    values = '\n'.join(values)

    sample_description = f"""
/*
3 rows from {table_name} table:
{columns}
{values}
*/    
"""
    return sample_description


def filter_create_table(create_table: str, columns_to_keep: List[str]) -> Tuple[str, List[str]]:

    # Find the boundaries of the CREATE TABLE statement
    start_line, end_line = find_create_table_boundaries(create_table)

    # If the CREATE TABLE statement is not found, return the original SQL
    if start_line is None or end_line is None:
        return create_table, columns_to_keep

    # Filter the columns within the boundaries
    return filter_columns_within_boundaries(create_table, start_line, end_line, columns_to_keep)


def get_filtered_table_context(db_uri: str, table_name: str, columns_to_keep: List[str]) -> str:
    db = SQLDatabase.from_uri(f"sqlite:///{db_uri}")
    db._sample_rows_in_table_info = 0
    create_table = db.get_table_info_no_throw([table_name])

    filtered_create_table, ordered_columns_to_keep = filter_create_table(create_table, columns_to_keep)
    sample = get_filtered_sample(db, table_name, ordered_columns_to_keep)

    return f"""{filtered_create_table}{sample}
"""


def table_descriptions_filtered_parser(database_dir, table_name, columns_to_keep):
    file_path = os.path.join(database_dir, f"{table_name}.csv")
    db_descriptions = f"Table: {table_name}\n"

    if not os.path.exists(file_path):
        return "The specified table CSV file does not exist."

    table_df = pd.read_csv(file_path, encoding='latin-1')
    for _, row in table_df.iterrows():
        column_name = str(row[0]).strip()
        if column_name in columns_to_keep:
            try:
                if pd.notna(row[2]):
                    col_description = re.sub(r'\s+', ' ', str(row[2]))  # noqa: E501
                    val_description = re.sub(r'\s+', ' ', str(row[4]))
                    if pd.notna(row[4]):
                        db_descriptions += f"Column {column_name}: column description -> {col_description}, value description -> {val_description}\n"  # noqa: E501
                    else:
                        db_descriptions += f"Column {column_name}: column description -> {col_description}\n"  # noqa: E501
            except Exception as e:
                print(e)
                db_descriptions += "No column description"
    db_descriptions += "\n"
    return db_descriptions


def get_top_k_columns(top_k: int, question_embedding: List[float], db_id: str, table_name: str) -> List[str]:
    # Load the CSV file
    df = pd.read_csv(f'db_preprocessing/column_description_embeddings/{db_id}.csv')

    # Filter the DataFrame
    df = df[df['table_name'] == table_name]

    # Convert the embedding strings back to lists of floats
    df['embedding'] = df['embedding'].apply(ast.literal_eval)

    # Calculate cosine similarity
    df['similarity'] = df.apply(lambda row: cosine_similarity([question_embedding], [row['embedding']])[0][0], axis=1)

    # Sort by similarity and get top K original column names
    top_k_columns = df.nlargest(top_k, 'similarity')['original_column_name'].tolist()

    return top_k_columns


def add_pk_fk_if_relation_exits_with_table_links(db_id, table_links, table_name, top_k_columns):
    pk_list, fk_dict = find_primary_and_foreign_keys(db_id, table_name)
    if pk_list:
        for pk in pk_list:
            if pk not in top_k_columns:
                top_k_columns.append(pk)
    if fk_dict and isinstance(fk_dict, dict):
        for foreign_table, foreign_keys in fk_dict.items():
            if foreign_table in table_links:
                for fk in foreign_keys:
                    if fk not in top_k_columns:
                        top_k_columns.append(fk)
    return [col.strip() for col in top_k_columns]


def get_focused_columns_for_table(db_id, question_embedding, table_name, table_links, top_k):
    top_k_columns = get_top_k_columns(
        top_k=top_k,
        question_embedding=question_embedding,
        db_id=db_id,
        table_name=table_name
    )
    # cleanup (some descriptions have leading whitespace
    top_k_columns = [col.strip() for col in top_k_columns]

    # As a safety we add all relevant primary keys and foreign keys
    return add_pk_fk_if_relation_exits_with_table_links(
        db_id,
        table_links,
        table_name,
        top_k_columns
    )


def get_column_links_for_tables(
    db_id: str,
    question_embedding,
    table_links: List[str],
    top_k: int = 0
) -> Dict[str, List[str]]:
    table_columns_dict = {}
    for table_name in table_links:
        columns = get_focused_columns_for_table(db_id, question_embedding, table_name, table_links, top_k)
        table_columns_dict[table_name] = columns
    return table_columns_dict


def link_schema_and_get_focused_context(
    db_uri: str,
    db_id: str,
    question_embedding,
    table_links: List[str],
    annotated_db_descriptions_path: str
) -> Tuple[str, str]:
    table_columns_dict = get_column_links_for_tables(db_id, question_embedding, table_links)
    return get_focused_schema_context_for_links(annotated_db_descriptions_path, db_uri, table_columns_dict)


def get_focused_schema_context_for_links(
        annotated_db_descriptions_path,
        db_uri,
        table_columns_dict
):
    schema_context = []
    descriptions_context = []
    for table_name, columns in table_columns_dict.items():
        schema = get_filtered_table_context(db_uri, table_name, columns)
        schema_context.append(schema)

        filtered_descriptions = table_descriptions_filtered_parser(annotated_db_descriptions_path, table_name, columns)
        descriptions_context.append(filtered_descriptions)
    return "\n".join(schema_context), "\n".join(descriptions_context)


