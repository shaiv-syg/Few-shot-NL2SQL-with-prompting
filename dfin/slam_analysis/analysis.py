
import pandas as pd
import json


def analyze_linked_tables(gold_links_json_path: str = '../../db_preprocessing/dev_gold_links.json',
                          linked_tables_json_path: str = '../../output/linked_tables/linked_tables_results_mode_minimal.json',
                          output_csv_path: str = "table_analysis.csv"):   # Load the gold tables data
    with open(gold_links_json_path, 'r') as file:
        gold_data = json.load(file)
    gold_tables_dict = {entry["question_id"]: set(entry["tables"]) for entry in gold_data}

    # Load the linked tables data
    with open(linked_tables_json_path, 'r') as file:
        linked_tables_data = json.load(file)
    linked_tables_dict = {entry["question_id"]: set(entry["linked_tables"]) for entry in linked_tables_data}

    # Filter gold_tables_dict to only include questions that are also in linked_tables_dict
    gold_tables_dict = {k: v for k, v in gold_tables_dict.items() if k in linked_tables_dict}

    # Initialize counters and lists
    total_questions = len(gold_tables_dict)
    fully_correct_predictions = 0
    missed_tables_count = 0
    extra_tables_count = 0
    number_of_questions_with_too_little_tables = 0

    # Prepare a DataFrame to store the detailed analysis
    analysis_df = pd.DataFrame(columns=["question_id", "missed_tables_count", "extra_tables_count", "missed_tables", "extra_tables", "recall", "tables_filtered_percentage"])

    # Analyze each entry
    for question_id, gold_tables in gold_tables_dict.items():
        linked_tables = linked_tables_dict.get(question_id, set())

        # Check if all gold tables are included in the prediction
        missed_tables = gold_tables - linked_tables
        extra_tables = linked_tables - gold_tables

        missed_count = len(missed_tables)
        extra_count = len(extra_tables)

        # if missed_count > 0:
        #     print(question_id)
        #     print(gold_tables, linked_tables)
        # Calculate recall for this question
        recall = (len(gold_tables) - missed_count) / len(gold_tables) if gold_tables else 1
        if len(missed_tables):
            print(question_id, gold_tables, linked_tables)

        if not missed_tables and not extra_tables:
            fully_correct_predictions += 1
        else:
            missed_tables_count += missed_count
            extra_tables_count += extra_count
        if missed_count:
            number_of_questions_with_too_little_tables += 1

        # Append data to the analysis DataFrame
        new_row = pd.DataFrame({
            "question_id": [question_id],
            "missed_tables_count": [missed_count],
            "extra_tables_count": [extra_count],
            "missed_tables": [missed_tables],
            "extra_tables": [extra_tables],
            "recall": [recall],
            "tables_filtered_percentage": [None]  # Not calculated in this context
        })
        analysis_df = pd.concat([analysis_df, new_row], ignore_index=True)

    # Save the detailed analysis to a CSV file
    analysis_df.to_csv(output_csv_path, index=False)

    # Calculate and print summary results
    avg_recall = analysis_df['recall'].mean()
    print(f"Total Questions: {total_questions}")
    print(f"Fully Correct Predictions: {fully_correct_predictions} ({(fully_correct_predictions / total_questions) * 100:.2f}%)")
    print(f"Average Recall: {avg_recall:.2f}")
    print(f"Average Missed Tables per Question: {missed_tables_count / total_questions:.2f}")
    print(f"Average Extra Tables per Question: {extra_tables_count / total_questions:.2f}")
    print("Failed in questions", number_of_questions_with_too_little_tables)


def analyze_linked_columns(gold_links_json_path: str, linked_columns_json_path: str, output_csv_path: str):
    with open(gold_links_json_path, 'r') as file:
        gold_data = json.load(file)
    gold_columns_dict = {entry["question_id"]: {table.lower(): [col.lower().strip() for col in columns] for table, columns in entry["columns"].items()} for entry in gold_data}

    # Load the linked columns data
    with open(linked_columns_json_path, 'r') as file:
        linked_columns_data = json.load(file)
    linked_columns_dict = {entry["question_id"]: {table.lower(): [col.lower().strip() for col in columns] for table, columns in entry["column_links"].items()} for entry in linked_columns_data}

    # Filter gold_columns_dict to only include questions that are also in linked_columns_dict
    gold_columns_dict = {k: v for k, v in gold_columns_dict.items() if k in linked_columns_dict}


    # Initialize counters and lists
    total_questions = len(gold_columns_dict)
    fully_correct_predictions = 0
    missed_columns_count = 0
    extra_columns_count = 0
    number_of_questions_with_too_little_columns = 0

    # Prepare a DataFrame to store the detailed analysis
    analysis_df = pd.DataFrame(columns=["question_id", "missed_columns_count", "extra_columns_count", "missed_columns", "extra_columns", "recall"])

    # Analyze each entry
    for question_id, gold_columns in gold_columns_dict.items():
        linked_columns = linked_columns_dict.get(question_id, {})

        total_missed_columns = 0
        total_extra_columns = 0
        total_recall = 0

        missed_columns = {}
        extra_columns = {}
        recalls = {}

        for table, columns in gold_columns.items():
            gold_set = set(columns)
            linked_set = set(linked_columns.get(table, []))

            missed = gold_set - linked_set
            extra = linked_set - gold_set

            recall = (len(gold_set) - len(missed)) / len(gold_set) if gold_set else 1

            total_missed_columns += len(missed)
            total_extra_columns += len(extra)
            total_recall += recall

            if missed:
                missed_columns[table] = list(missed)
                print(question_id, gold_set, linked_set)
            if extra:
                extra_columns[table] = list(extra)

            recalls[table] = recall

        avg_recall = total_recall / len(gold_columns) if gold_columns else 1

        if not total_missed_columns and not total_extra_columns:
            fully_correct_predictions += 1
        else:
            missed_columns_count += total_missed_columns
            extra_columns_count += total_extra_columns
        if total_missed_columns:
            number_of_questions_with_too_little_columns += 1


        # Append data to the analysis DataFrame
        new_row = pd.DataFrame({
            "question_id": [question_id],
            "missed_columns_count": [total_missed_columns],
            "extra_columns_count": [total_extra_columns],
            "missed_columns": [missed_columns],
            "extra_columns": [extra_columns],
            "recall": [avg_recall]
        })
        analysis_df = pd.concat([analysis_df, new_row], ignore_index=True)

    # Save the detailed analysis to a CSV file
    analysis_df.to_csv(output_csv_path, index=False)

    # Calculate and print summary results
    avg_recall = analysis_df['recall'].mean()
    print(f"Total Questions: {total_questions}")
    print(f"Fully Correct Predictions: {fully_correct_predictions} ({(fully_correct_predictions / total_questions) * 100:.2f}%)")
    print(f"Average Recall: {avg_recall:.2f}")
    print(f"Average Missed Columns per Question: {missed_columns_count / total_questions:.2f}")
    print(f"Average Extra Columns per Question: {extra_columns_count / total_questions:.2f}")
    print("Failed in questions", number_of_questions_with_too_little_columns)

# Example usage
analyze_linked_columns(
    gold_links_json_path='../../db_preprocessing/dev_gold_links.json',
    linked_columns_json_path='../../output/linked_columns/column_links_results_conservative_top_k_15.json',
    output_csv_path="column_analysis.csv"
)


analyze_linked_tables()
