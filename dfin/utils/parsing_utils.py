import json
from typing import List
from sqlglot import exp, parse_one
from sqlglot.dialects import Dialects


def get_query_tables(query: str) -> List[str]:
    # Returns all the table names which are used in the query
    parsed = parse_one(query, read=Dialects.SQLITE)
    return sorted(list({str(table.this).strip() for table in parsed.find_all(exp.Table)}))


def get_query_columns(query: str) -> List[str]:
    # Returns all the table names which are used in the query
    parsed = parse_one(query, read=Dialects.SQLITE)
    return sorted(list({str(table.this).strip() for table in parsed.find_all(exp.Column)}))


def get_columns_usage(query):
    parsed = parse_one(query, read=Dialects.SQLITE)
    table_aliases = {}
    for table in parsed.find_all(exp.Table):
        if table.alias:
            alias_name = table.alias.this if isinstance(table.alias, exp.Alias) else table.alias
            table_aliases[alias_name] = str(table.this)
        else:
            alias_name = str(table.this)
            table_aliases[alias_name] = alias_name

            # Step 2: Create the result dictionary
    result = {}
    for column in parsed.find_all(exp.Column):
        # Resolve the table name using the aliases dictionary
        if column.table:
            table_name = table_aliases.get(column.table)
        else:
            table_name = str(column.parent_select.find(exp.Table))
        column_name = column.this

        # Add the column to the list of columns for that table
        if table_name not in result:
            result[table_name] = set()
        result[table_name].add(column_name.this)

    for key, val in result.items():
        result[key] = list(val)
    return result


def process_tables_and_columns_in_dataset(dataset_path: str = "../../dev/dev.json", output_path: str = '../../db_preprocessing/dev_gold_links.json'):
    # Read the JSON data from the file
    with open(dataset_path, 'r') as file:
        data = json.load(file)

    # Iterate through all entries, get the tables and columns, and update the entry
    updated_data = []
    for entry in data:
        query = entry["SQL"]
        tables = get_query_tables(query)
        columns = get_columns_usage(query)
        question_id = entry["question_id"]

        # Create a new dictionary with only the required keys
        new_entry = {
            "question_id": question_id,
            "SQL": query,
            "tables": tables,
            "columns": columns
        }
        updated_data.append(new_entry)

    # Write the updated data back to the JSON file
    with open(output_path, 'w') as file:
        json.dump(updated_data, file, indent=4)


# process_tables_and_columns_in_dataset()