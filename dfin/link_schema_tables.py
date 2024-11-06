import json
import re
import pandas as pd
from typing import Set, Optional
from utils.azure_openai import get_completion_4
from consts import *


def extract_tables_result(text):
    """
    Extract table names from a given text based on the pattern.

    Args:
    - text (str): Input text

    Returns:
    - set: A set containing table names
    """
    # Extract string between curly braces
    matched_str = re.search(r"\{(.*?)\}", text)

    # If no match, return empty set
    if not matched_str:
        return set()

    # Split string by comma to get individual tables and strip whitespace
    tables = [table.strip(" '") for table in matched_str.group(1).split(",")]

    # Convert list to set and return
    return set(tables)


def get_table_links_minimal(question: str, hint: Optional[str], tables_descriptions: str) -> Set[str]:
    prompt = f"""
You are a sql query assistant schema linker. You are provided with a natural language question a possible hint and a list of table descriptions.
Your task is to pick the minimum relevant tables for the query. Each table you pick is costly so be cautious and weary for each table you consider.
the final output of tables should look like a pytho set -> {{'table_a', 'table_b'}}

Table Descriptions:
####

{tables_descriptions}

####

Question: {question}
{f"Hint: {hint}" if hint else ''}

tables for query, Let's think step by step."""

    result = get_completion_4(prompt)
    return extract_tables_result(result)


def get_table_links_conservative(question: str, hint: Optional[str], tables_descriptions: str) -> Set[str]:
    prompt = f"""
You are a sql query assistant schema linker with a strong focus on high recall. You are provided with a natural language question a possible hint and a list of table descriptions.
Your task is to identify all the tables that could possibly be relevant to the query. It is crucial that you do not miss any relevant tables, even if it means including a few that might not be strictly necessary. Each missed table is a significant issue, while including an extra table is a minor inconvenience.

Table Descriptions:
####

{tables_descriptions}

####

Question: {question}
{f"Hint: {hint}" if hint else ''}

Considering the importance of high recall and the relative cost of missing a table versus including an extra one, list the tables for the query. Aim for completeness and be err on the side of inclusion. Your output should be formatted as a python set -> {{'table_a', 'table_b'}}.

Tables for query, Let's think step by step."""

    result = get_completion_4(prompt)
    return extract_tables_result(result)



def predict_linked_tables(question: str,
                          hint: Optional[str],
                          db_id: str,
                          tables_descriptions,
                          mode: str = "original"):
    # Filter the table descriptions based on the db_id
    db_tables_descriptions = [desc for desc in tables_descriptions if desc['db_name'] == db_id]
    descriptions = "\n\n----\n\n".join(
        ["Table {}: {}".format(desc['table_name'], desc['description']) for desc in db_tables_descriptions])

    # Call the linking function
    if mode == MINIMAL:
        predicted_tables = get_table_links_minimal(question, hint, descriptions)
    elif mode == CONSERVATIVE:
        predicted_tables = get_table_links_conservative(question, hint, descriptions)
    else:
        raise Exception(f"Link table mode is missing use the following: {LINK_TABLE_MODES}")

    # Sort tables
    return sorted(list(predicted_tables))

#
# def process_questions_and_save_linked_tables_json(
#         subset_dataset_path: str = "dev/dev_subset.json",
#         tables_descriptions_path: str = "dev/tables_description.json",
#         linking_func: str = "original",
#         output_path: str = "linked_tables.csv"
# ):
#     # Read the subset dataset
#     with open(subset_dataset_path, 'r') as file:
#         subset_data = json.load(file)
#
#     # Read the tables descriptions JSON file
#     with open(tables_descriptions_path, 'r') as file:
#         tables_descriptions = json.load(file)
#
#     # Create a DataFrame to store the results
#     results_df = pd.DataFrame(columns=['question_id', 'db_id', 'gold_tables', 'predicted_tables', 'total_tables'])
#
#     # Process each question
#     for entry in subset_data:
#         question_id = entry["question_id"]
#         db_id = entry["db_id"]
#         question = entry["question"]
#         tables = entry.get("tables", [])  # Default to an empty list if "tables" is not present
#
#         # Filter the table descriptions based on the db_id
#         db_tables_descriptions = [desc for desc in tables_descriptions if desc['db_name'] == db_id]
#         descriptions = "\n".join(["Table {}: {}".format(desc['table_name'], desc['description']) for desc in db_tables_descriptions])
#         total_tables_in_db = len(db_tables_descriptions)
#
#         predicted_tables = functions[linking_func](question, descriptions)
#
#         # Sort tables and convert to string representation of list
#         gold_tables_str = str(sorted(tables))
#         predicted_tables_str = str(sorted(list(predicted_tables)))
#         print(question, tables, predicted_tables)
#         # Add the result to the DataFrame
#         new_row = pd.DataFrame({
#             'question_id': [question_id],
#             'db_id': [db_id],
#             'gold_tables': [gold_tables_str],
#             'predicted_tables': [predicted_tables_str],
#             'total_tables': [total_tables_in_db]
#         })
#         results_df = pd.concat([results_df, new_row], ignore_index=True)
#
#     # Save the results to a CSV file
#     results_df.to_csv(output_path, index=False)
#     print(f"Results saved to {output_path}")


if __name__ == '__main__':
    print("original")
    # process_questions_and_save_linked_tables_json(
    #     subset_dataset_path="dev/dev_subset.json",
    #     tables_descriptions_path="dev/tables_description.json",
    #     linking_func="original",
    #     output_path="linked_tables_original.csv"
    # )
    # print("conservative")
    # process_questions_and_save_linked_tables_json(
    #     subset_dataset_path="dev/dev_subset.json",
    #     tables_descriptions_path="dev/tables_description.json",
    #     linking_func="conservative",
    #     output_path="linked_tables_conservative.csv"
    # )





