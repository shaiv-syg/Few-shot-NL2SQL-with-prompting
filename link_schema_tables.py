import json
import re
import pandas as pd
from typing import Set
from azure_openai import get_completion_4


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
    tables = [table.strip(" '").lower() for table in matched_str.group(1).split(",")]

    # Convert list to set and return
    return set(tables)


def get_table_links_1(question: str, tables_descriptions: str) -> Set[str]:
    prompt = f"""
You are a sql query assistant schema linker. You are provided with a natural language question and a list of table descriptions.
Your task is to pick the minimum relevant tables for the query. Each table you pick is costly so be cautious and weary for each table you consider.
the final output of tables should look like a pytho set -> {{'table_a', 'table_b'}}

Question: {question}

Table Descriptions:
####
{tables_descriptions}####

tables for query, Let's think step by step."""

    result = get_completion_4(prompt)
    return extract_tables_result(result)

def get_table_links_2(question: str, tables_descriptions: str) -> Set[str]:
    prompt = f"""
You are a sql query assistant schema linker with a strong focus on high recall. You are provided with a natural language question and a list of table descriptions.
Your task is to identify all the tables that could possibly be relevant to the query. It is crucial that you do not miss any relevant tables, even if it means including a few that might not be strictly necessary. Each missed table is a significant issue, while including an extra table is a minor inconvenience.

Question: {question}

Table Descriptions:
####
{tables_descriptions}####

Considering the importance of high recall and the relative cost of missing a table versus including an extra one, list the tables for the query. Aim for completeness and be err on the side of inclusion. Your output should be formatted as a python set -> {{'table_a', 'table_b'}}.

Tables for query, Let's think step by step."""

    result = get_completion_4(prompt)
    return extract_tables_result(result)


functions = {
    "original": get_table_links_1,
    "conservative": get_table_links_2
}


def process_questions_and_save_linked_tables(subset_dataset_path: str = "dev/dev_subset.json",
                                             tables_descriptions_path: str = "dev/tables_description.csv",
                                             linking_func: str = "original"):
    # Read the subset dataset
    with open(subset_dataset_path, 'r') as file:
        subset_data = json.load(file)

    # Read the tables descriptions CSV file
    tables_descriptions_df = pd.read_csv(tables_descriptions_path)

    # Create a DataFrame to store the results
    results_df = pd.DataFrame(columns=['question_id', 'db_id', 'gold_tables', 'predicted_tables', 'total_tables'])

    # Process each question
    for entry in subset_data:
        question_id = entry["question_id"]
        db_id = entry["db_id"]
        question = entry["question"]
        tables = entry.get("tables", [])  # Default to an empty list if "tables" is not present

        # Filter the table descriptions based on the db_id
        db_tables_descriptions = tables_descriptions_df[tables_descriptions_df['db_name'] == db_id]
        descriptions = "\n".join(
            "Table " + db_tables_descriptions['table_name'] + ": " + db_tables_descriptions['description'])
        total_tables_in_db = len(db_tables_descriptions)

        predicted_tables = functions[linking_func](question, descriptions)

        # Add the result to the DataFrame
        # Sort tables and convert to string representation of list
        gold_tables_str = str(sorted(tables))
        predicted_tables_str = str(sorted(list(predicted_tables)))
        print(question, tables, predicted_tables)
        # Add the result to the DataFrame
        new_row = pd.DataFrame({
            'question_id': [question_id],
            'db_id': [db_id],
            'gold_tables': [gold_tables_str],
            'predicted_tables': [predicted_tables_str],
            'total_tables': total_tables_in_db
        })
        results_df = pd.concat([results_df, new_row], ignore_index=True)

    # Save the results to a CSV file
    results_df.to_csv("linked_tables.csv", index=False)
    print("Results saved to linked_tables.csv")


def process_questions_and_save_linked_tables_json(subset_dataset_path: str = "dev/dev_subset.json",
                                             tables_descriptions_path: str = "dev/dev_tables_description_2.json",
                                             linking_func: str = "original",
                                            output_path: str = "linked_tables.csv"):
    # Read the subset dataset
    with open(subset_dataset_path, 'r') as file:
        subset_data = json.load(file)

    # Read the tables descriptions JSON file
    with open(tables_descriptions_path, 'r') as file:
        tables_descriptions = json.load(file)

    # Create a DataFrame to store the results
    results_df = pd.DataFrame(columns=['question_id', 'db_id', 'gold_tables', 'predicted_tables', 'total_tables'])

    # Process each question
    for entry in subset_data:
        question_id = entry["question_id"]
        db_id = entry["db_id"]
        question = entry["question"]
        tables = entry.get("tables", [])  # Default to an empty list if "tables" is not present

        # Filter the table descriptions based on the db_id
        db_tables_descriptions = [desc for desc in tables_descriptions if desc['db_name'] == db_id]
        descriptions = "\n".join(["Table {}: {}".format(desc['table_name'], desc['description']) for desc in db_tables_descriptions])
        total_tables_in_db = len(db_tables_descriptions)

        predicted_tables = functions[linking_func](question, descriptions)

        # Sort tables and convert to string representation of list
        gold_tables_str = str(sorted(tables))
        predicted_tables_str = str(sorted(list(predicted_tables)))
        print(question, tables, predicted_tables)
        # Add the result to the DataFrame
        new_row = pd.DataFrame({
            'question_id': [question_id],
            'db_id': [db_id],
            'gold_tables': [gold_tables_str],
            'predicted_tables': [predicted_tables_str],
            'total_tables': [total_tables_in_db]
        })
        results_df = pd.concat([results_df, new_row], ignore_index=True)

    # Save the results to a CSV file
    results_df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")


if __name__ == '__main__':
    print("original")
    process_questions_and_save_linked_tables_json(
        subset_dataset_path="dev/dev_subset.json",
        tables_descriptions_path="dev/dev_tables_description_2.json",
        linking_func="original",
        output_path="linked_tables_json_original.csv"
    )
    print("conservative")
    process_questions_and_save_linked_tables_json(
        subset_dataset_path="dev/dev_subset.json",
        tables_descriptions_path="dev/dev_tables_description_2.json",
        linking_func="conservative",
        output_path="linked_tables_json_conservative.csv"
    )
    # process_questions_and_save_linked_tables()

# question = "how many accounts who choose issuance after transaction are staying in East Bohemia region?"
# question = "The average unemployment ratio of 1995 and 1996, which one has higher percentage?"
# descriptions = """
# Table: loan, "The 'loan' table in the 'financial' database stores information about various loans. Each record includes the loan ID, the associated account ID, the date the loan was approved, the approved amount in US dollars, the loan duration in months, the monthly payments, and the repayment status. The status can indicate whether the contract is finished or running and if there have been any payment issues."
# Table: client, "The 'client' table in the 'financial' database stores information about the bank's clients. Each row represents a unique client, identified by a unique 'client_id', and includes details such as the client's gender, birth date, and the ID of the district where their branch is located. The 'district_id' is a foreign key that references the 'district' table."
# Table: district, "The 'district' table in the 'financial' database contains detailed information about different districts. It includes data such as the district name, region, municipality hierarchy, ratio of urban inhabitants, average salary, unemployment rates for 1995 and 1996, number of entrepreneurs per 1000 inhabitants, and the number of committed crimes for 1995 and 1996. The 'district_id' serves as the primary key for this table."
# Table: trans, "The 'trans' table in the 'financial' database stores information about various transactions. Each record includes details such as transaction id, account id, date of transaction, type of transaction (credit or withdrawal), mode of operation, amount of money involved in the transaction, and the balance after the transaction. The table also contains optional fields for the symbol, bank, and account related to the transaction."
# Table: account, "The 'account' table in the 'financial' database holds information about various accounts. It includes details such as the account id, the location of the branch (district_id), the frequency of the account, and the creation date of the account. The account_id serves as the primary key, and the district_id is a foreign key referencing the district table."
# Table: card, "The 'card' table in the 'financial' database stores information about credit cards. It includes details such as the card's unique ID, the associated disposition ID, the type of credit card (junior, classic, or gold), and the date the card was issued. The 'disp_id' column is a foreign key that references the 'disp' table."
# Table: order, "The ""order"" table in the ""financial"" database is used to track individual financial transactions. Each row represents a unique order, identified by the ""order_id"", and includes details such as the account initiating the transaction (""account_id""), the recipient's bank and account (""bank_to"" and ""account_to""), the debited amount (""amount""), and the purpose of the payment (""k_symbol""). The ""k_symbol"" can indicate various types of payments such as insurance, household, leasing, or loan payments."
# Table: disp, "The 'disp' table in the 'financial' database is used to manage the relationship between clients and their accounts. It contains unique identifiers for each record (disp_id), the client (client_id), and the account (account_id), as well as the type of disposition (type) which can be 'OWNER', 'USER', or 'DISPONENT'. The type of disposition indicates the level of control a client has over an account, such as the ability to issue orders or apply for loans."
# """
#
#
# tables = get_table_links(question, descriptions)




