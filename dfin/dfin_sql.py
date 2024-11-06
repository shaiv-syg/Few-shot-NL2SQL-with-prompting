from typing import List, Dict
import datetime

import pandas as pd
import json

from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from consts import *
from utils.azure_openai import get_embedding, get_langchain_llm_4_32_k, get_langchain_llm_4
from original_din_sql_utils.din_sql_original_prompts import SYSTEM_SCHEMA_LINKING_TEMPLATE, HUMAN_SCHEMA_LINKING_TEMPLATE, \
    SYSTEM_CLASSIFICATION_TEMPLATE, HUMAN_CLASSIFICATION_TEMPLATE, SYSTEM_EASY_CLASS_TEMPLATE, \
    HUMAN_EASY_CLASS_TEMPLATE, SYSTEM_NON_NESTED_CLASS_TEMPLATE, HUMAN_NON_NESTED_CLASS_TEMPLATE, \
    SYSTEM_NESTED_CLASS_TEMPLATE, HUMAN_NESTED_CLASS_TEMPLATE, SYSTEM_SELF_CORRECTION_PROMPT, \
    HUMAN_SELF_CORRECTION_PROMPT
from original_din_sql_utils.din_sql_original_utils import extract_schema_links, extract_label_and_sub_questions, extract_sql_query, \
    extract_revised_sql_query, update_json_file
from link_columns import link_schema_and_get_focused_context, get_focused_schema_context_for_links
from link_schema_tables import predict_linked_tables


CHAT = get_langchain_llm_4()
# Store the current time as the last reset time
last_reset_time = datetime.datetime.now()
dev_df = pd.read_json(BIRD_DEV_JSON_PATH)

# ----------------------- #

with open(f'{LINKED_COLUMNS_DIR}/column_links_results_minimal_top_k_15.json', 'r') as f:
    preprocessed_linked_columns = json.load(f)


def get_dfin_focused_schema_context(
        annotated_db_descriptions_path: str,
        db_uri: str,
        question_id: int,
):
    existing_entry = next((entry for entry in preprocessed_linked_columns if entry["question_id"] == question_id), None)

    if existing_entry:
        return get_focused_schema_context_for_links(
            annotated_db_descriptions_path=annotated_db_descriptions_path,
            db_uri=db_uri,
            table_columns_dict=existing_entry["column_links"]
        )
    raise Exception("Missing question id in preprocessed links")


start_index = 0


if __name__ == "__main__":
    logs_df = pd.DataFrame(
        columns=["question","gold_query","db_id","final_query","schema_linking","classification","sql_generation","self_correction"])

    system_schema_linking_prompt = SystemMessagePromptTemplate.from_template(SYSTEM_SCHEMA_LINKING_TEMPLATE)  # noqa: E501
    human_schema_linking_prompt = HumanMessagePromptTemplate.from_template(HUMAN_SCHEMA_LINKING_TEMPLATE) # noqa: E501
    schema_linking_prompt = ChatPromptTemplate.from_messages([system_schema_linking_prompt, human_schema_linking_prompt]) # noqa: E501
    system_classification_prompt = SystemMessagePromptTemplate.from_template(SYSTEM_CLASSIFICATION_TEMPLATE) # noqa: E501
    human_classification_prompt = HumanMessagePromptTemplate.from_template(HUMAN_CLASSIFICATION_TEMPLATE) # noqa: E501
    classification_prompt = ChatPromptTemplate.from_messages([system_classification_prompt, human_classification_prompt]) # noqa: E501
    system_easy_prompt = SystemMessagePromptTemplate.from_template(SYSTEM_EASY_CLASS_TEMPLATE) # noqa: E501
    human_easy_prompt = HumanMessagePromptTemplate.from_template(HUMAN_EASY_CLASS_TEMPLATE) # noqa: E501
    easy_prompt = ChatPromptTemplate.from_messages([system_easy_prompt, human_easy_prompt]) # noqa: E501
    system_easy_prompt = SystemMessagePromptTemplate.from_template(SYSTEM_EASY_CLASS_TEMPLATE) # noqa: E501
    human_easy_prompt = HumanMessagePromptTemplate.from_template(HUMAN_EASY_CLASS_TEMPLATE) # noqa: E501
    easy_prompt = ChatPromptTemplate.from_messages([system_easy_prompt, human_easy_prompt]) # noqa: E501
    system_medium_prompt = SystemMessagePromptTemplate.from_template(SYSTEM_NON_NESTED_CLASS_TEMPLATE) # noqa: E501
    human_medium_prompt = HumanMessagePromptTemplate.from_template(HUMAN_NON_NESTED_CLASS_TEMPLATE) # noqa: E501
    medium_prompt = ChatPromptTemplate.from_messages([system_medium_prompt, human_medium_prompt]) # noqa: E501
    system_hard_prompt = SystemMessagePromptTemplate.from_template(SYSTEM_NESTED_CLASS_TEMPLATE) # noqa: E501
    human_hard_prompt = HumanMessagePromptTemplate.from_template(HUMAN_NESTED_CLASS_TEMPLATE) # noqa: E501
    hard_prompt = ChatPromptTemplate.from_messages([system_hard_prompt, human_hard_prompt]) # noqa: E501
    system_correction_prompt = SystemMessagePromptTemplate.from_template(SYSTEM_SELF_CORRECTION_PROMPT) # noqa: E501
    human_correction_prompt = HumanMessagePromptTemplate.from_template(HUMAN_SELF_CORRECTION_PROMPT) # noqa: E501
    correction_prompt = ChatPromptTemplate.from_messages([system_correction_prompt, human_correction_prompt]) # noqa: E501

    with open(PREPROCESSING_DEV_TABLE_DESCRIPTIONS, 'r') as file:
        tables_descriptions = json.load(file)

    accuracy = 0
    for index,row in dev_df.iterrows():
        if index < start_index:
            continue

        CHAT = get_langchain_llm_4()
        time_diff = datetime.datetime.now() - last_reset_time
        # Check if 50 minutes have passed
        if time_diff.total_seconds() >= 100:
            # If yes, update the chat variable and reset the last reset time
            last_reset_time = datetime.datetime.now()
            print('RESET',last_reset_time)
        else:
            print('SKIP, ', last_reset_time, time_diff.total_seconds())

        print("Processing row: ", index)
        db_uri = BIRD_DEV_DATABASES_PATH + "/" + row["db_id"] + "/" + row["db_id"] + ".sqlite"
        question = row["question"]
        db_id = row["db_id"]
        hint = str(row["evidence"])
        db_descriptions = BIRD_DEV_DATABASES_PATH + "/" + row["db_id"] + "/" + BIRD_DATABASE_DB_DESCRIPTION_DIR  # noqa: E501
        print("Database: ", db_uri)
        # columns_descriptions = table_descriptions_parser(db_descriptions)
        # schema = get_database_schema(db_uri)
        print("Question: ", question)
        question_id = row["question_id"]

        # --------- DFIN -> Focused context --------------

        schema, columns_descriptions = get_dfin_focused_schema_context(
            annotated_db_descriptions_path=db_descriptions,
            db_uri=db_uri,
            question_id=question_id
        )

        # -------------------------------------------------

        chain = LLMChain(llm=CHAT, prompt=schema_linking_prompt, verbose=False)
        schema_linking = chain.run(question=question, schema=schema, hint=hint, columns_descriptions=columns_descriptions) # noqa: E501
        schema_links = extract_schema_links(schema_linking)
        print(schema_links)
        chain = LLMChain(llm=CHAT, prompt=classification_prompt)
        classification = chain.run(
            question=question,
            schema=schema,
            hint=hint,
            columns_descriptions=columns_descriptions,
            schema_links=schema_links)
        label, sub_questions = extract_label_and_sub_questions(classification)
        print("Label: ", label)
        sql_generation = None
        if "EASY" in label:
            chain = LLMChain(llm=CHAT, prompt=easy_prompt)
            easy = chain.run(
                question=question,
                schema=schema,
                hint=hint,
                columns_descriptions=columns_descriptions,
                schema_links=schema_links)
            sql_query = extract_sql_query(easy)
            sql_generation = easy
        elif "NON-NESTED" in label:     
            chain = LLMChain(llm=CHAT, prompt=medium_prompt)
            medium = chain.run(
                question=question,
                schema=schema,
                hint=hint,
                columns_descriptions=columns_descriptions,
                schema_links=schema_links)
            sql_query = extract_sql_query(medium)
            sql_generation = medium
        else:
            chain = LLMChain(llm=CHAT, prompt=hard_prompt)
            hard = chain.run(
                question=question,
                schema=schema,
                hint=hint,
                columns_descriptions=columns_descriptions,
                schema_links=schema_links,
                sub_questions=sub_questions)
            sql_query = extract_sql_query(hard)
            sql_generation = hard
        chain = LLMChain(llm=CHAT, prompt=correction_prompt)
        correction = chain.run(
            question=question,
            schema=schema,
            columns_descriptions=columns_descriptions,
            hint=hint,
            sql_query=sql_query)
        finall_sql = extract_revised_sql_query(correction)
        if finall_sql is not None:
            one_liner_sql_query = finall_sql.replace('\n', '').replace('\r', '')
        else:
            if sql_query is not None:
                one_liner_sql_query = sql_query.replace('\n', '').replace('\r', '')
            else:
                one_liner_sql_query = "SELECT * FROM table" # no query generated, placeholder to avoid errors # noqa: E501
        new_row_df = pd.DataFrame(
            [[question,row["SQL"],row["db_id"],one_liner_sql_query,schema_linking, classification, sql_generation, correction]],  # noqa: E501
            columns=["question","gold_query","db_id","final_query","schema_linking","classification","sql_generation","self_correction"])
        logs_df = pd.concat([logs_df, new_row_df], ignore_index=True)
        logs_df.to_csv(LOGS_PATH, index=False)
        update_json_file(OUTPUT_PREDICT_JSON, index, one_liner_sql_query, row["db_id"])
        print("final sql query: ", one_liner_sql_query)
        print("Gold sql query: ", row["SQL"])
        print("--------------------------------------------------")


