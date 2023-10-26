import pandas as pd

def analyze_linked_tables(linked_tables_csv_path: str = "linked_tables.csv", output_csv_path: str = "table_analysis_2.csv"):
    # Read the linked tables CSV file
    df = pd.read_csv(linked_tables_csv_path)

    # Initialize counters and lists
    total_questions = 0
    fully_correct_predictions = 0
    missed_tables_count = 0
    extra_tables_count = 0
    number_of_questions_with_too_little_tables = 0
    tables_filtered_percentages = []

    # Prepare a DataFrame to store the detailed analysis
    analysis_df = pd.DataFrame(columns=["question_id", "missed_tables_count", "extra_tables_count", "missed_tables", "extra_tables", "recall", "tables_filtered_percentage"])

    # Analyze each row
    for _, row in df.iterrows():
        total_questions += 1
        gold_tables = eval(row['gold_tables'])
        predicted_tables = eval(row['predicted_tables'])
        total_tables = row['total_tables']

        # Check if all gold tables are included in the prediction
        missed_tables = set(gold_tables) - set(predicted_tables)
        extra_tables = set(predicted_tables) - set(gold_tables)

        missed_count = len(missed_tables)
        extra_count = len(extra_tables)

        # Calculate recall for this question
        recall = (len(gold_tables) - missed_count) / len(gold_tables) if gold_tables else 1

        # Calculate percentage of tables filtered
        filtered_tables = total_tables - len(predicted_tables)
        tables_filtered_percentage = (filtered_tables / total_tables) * 100 if total_tables else 0
        tables_filtered_percentages.append(tables_filtered_percentage)

        if not missed_tables and not extra_tables:
            fully_correct_predictions += 1
        else:
            missed_tables_count += missed_count
            extra_tables_count += extra_count
        if missed_count:
            number_of_questions_with_too_little_tables += 1

        # Append data to the analysis DataFrame
        new_row = pd.DataFrame({
            "question_id": [row['question_id']],
            "missed_tables_count": [missed_count],
            "extra_tables_count": [extra_count],
            "missed_tables": [missed_tables],
            "extra_tables": [extra_tables],
            "recall": [recall],
            "tables_filtered_percentage": [tables_filtered_percentage]
        })
        analysis_df = pd.concat([analysis_df, new_row], ignore_index=True)

    # Save the detailed analysis to a CSV file
    analysis_df.to_csv(output_csv_path, index=False)

    # Calculate and print summary results
    avg_recall = analysis_df['recall'].mean()
    avg_tables_filtered_percentage = sum(tables_filtered_percentages) / total_questions
    print(f"Total Questions: {total_questions}")
    print(f"Fully Correct Predictions: {fully_correct_predictions} ({(fully_correct_predictions / total_questions) * 100:.2f}%)")
    print(f"Average Recall: {avg_recall:.2f}")
    print(f"Average Missed Tables per Question: {missed_tables_count / total_questions:.2f}")
    print(f"Average Extra Tables per Question: {extra_tables_count / total_questions:.2f}")
    print(f"Average Tables Filtered Percentage: {avg_tables_filtered_percentage:.2f}%")
    print("Failed in questions", number_of_questions_with_too_little_tables)




if __name__ == "__main__":
    print("original")
    analyze_linked_tables(
        linked_tables_csv_path="../../output/linked_tables_original.csv",
        output_csv_path="linked_tables_original_analysis.csv"
    )
    print("conservative")
    analyze_linked_tables(
        linked_tables_csv_path="../../output/linked_tables_conservative.csv",
        output_csv_path="linked_tables_conservative_analysis.csv"
    )