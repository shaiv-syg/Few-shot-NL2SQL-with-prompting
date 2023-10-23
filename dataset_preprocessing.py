import os
import glob
import json
import re
import pandas as pd
from langchain.utilities.sql_database import SQLDatabase
from azure_openai import get_completion_4


def table_description_parser(database_dir, table_name):
    """
    Returns a description string for the given table.

    Args:
    - database_dir (str): Path to the directory containing the CSV files.
    - table_name (str): Name of the table to be processed.

    Returns:
    - str: Description of the table.
    """
    file_path = os.path.join(database_dir, f"{table_name}.csv")
    if not os.path.exists(file_path):
        return f"No CSV found for table: {table_name}"

    db_description = f""
    table_df = pd.read_csv(file_path, encoding='latin-1')

    for _, row in table_df.iterrows():
        try:
            if pd.notna(row[2]):
                col_description = re.sub(r'\s+', ' ', str(row[2]))
                val_description = re.sub(r'\s+', ' ', str(row[4]))
                if pd.notna(row[4]):
                    db_description += f"Column {row[0]}: column description -> {col_description}, value description -> {val_description}\n"
                else:
                    db_description += f"Column {row[0]}: column description -> {col_description}\n"
        except Exception as e:
            print(e)
            db_description += "No column description"

    db_description += "\n"
    return db_description


def get_table_names(database_dir):
    """
    Returns a list of table description files present in the given directory of the bird dataset.

    Args:
    - database_dir (str): Path to the directory containing the CSV files.

    Returns:
    - List[str]: List of table names.
    """
    csv_files = glob.glob(f"{database_dir}/*.csv")
    return [os.path.basename(file_path).replace(".csv", "") for file_path in csv_files]


def get_table_create_statement_with_sample(db_uri, table_name):
    db = SQLDatabase.from_uri("sqlite:///"+db_uri)
    db._sample_rows_in_table_info = 3
    return db.get_table_info_no_throw([table_name])


def get_table_description(db_name, table_name, create_statement, annotated_columns_description):
    prompt = f"""
You are a helpful database inspection assistant you are given details about the database name and a specific table information within the database.
Your task is to generate a short description of the table (2-3 sentences long).

You are provided with the following: database name, table name, create table statement, sample of 3 rows and annotated information of each column in the table.

Database name: {db_name}
Table Name: {table_name}
Create Table Statement + Sample:
```{create_statement}
```

Columns annotated information:
```
{annotated_columns_description}```
Table description: """
    return get_completion_4(prompt)


def get_table_description_2(db_name, table_name, create_statement, annotated_columns_description):
    prompt = f"""
You are a helpful database inspection assistant tasked with generating concise and clear descriptions of database tables. The description should highlight the unique aspects of each table, helping someone to understand its role and how it might be used in a query. Focus on key identifiers and relationships with other tables if applicable.

Example Input ->
Database name: mubi
Table Name: movies
Create Table Statement + Sample:
```
CREATE TABLE movies (
        movie_id INTEGER NOT NULL, 
        movie_title TEXT, 
        movie_release_year INTEGER, 
        movie_url TEXT, 
        movie_title_language TEXT, 
        movie_popularity INTEGER, 
        movie_image_url TEXT, 
        director_id TEXT, 
        director_name TEXT, 
        director_url TEXT, 
        PRIMARY KEY (movie_id)
)
3 rows from movies table:
movie_id        movie_title     movie_release_year      movie_url       movie_title_language    movie_popularity        movie_image_url director_id     director_namedirector_url
1       La Antena       2007    http://mubi.com/films/la-antena en      105     https://images.mubicdn.net/images/film/1/cache-7927-1581389497/image-w1280.jpg  131  Esteban Sapir    http://mubi.com/cast/esteban-sapir
2       Elementary Particles    2006    http://mubi.com/films/elementary-particles      en      23      https://images.mubicdn.net/images/film/2/cache-512179-1581389841/image-w1280.jpg      73      Oskar Roehler   http://mubi.com/cast/oskar-roehler
3       It's Winter     2006    http://mubi.com/films/its-winter        en      21      https://images.mubicdn.net/images/film/3/cache-7929-1481539519/image-w1280.jpg82      Rafi Pitts      http://mubi.com/cast/rafi-pitts
*/
```

Columns annotated information:
```
Column movie_id: column description -> ID related to the movie on Mubi
Column movie_title: column description -> Name of the movie
Column movie_release_year: column description -> Release year of the movie
Column movie_url: column description -> URL to the movie page on Mubi
Column movie_title_language: column description -> By default, the title is in English., value description -> Only contains one value which is 'en'
Column movie_popularity: column description -> Number of Mubi users who love this movie
Column movie_image_url: column description -> Image URL to the movie on Mubi
Column director_id: column description -> ID related to the movie director on Mubi
Column director_name: column description -> Full Name of the movie director
Column director_url : column description -> URL to the movie director page on Mubi
``` 

Table description: 
The "movies" table in "mubi" database is a central repository for films, uniquely identified by "movie_id". It encompasses vital attributes like "movie_title," "movie_release_year," and "movie_popularity," along with direct links to the film and director's pages on Mubi through "movie_url" and "director_url". The uniform use of English for "movie_title_language" ensures consistency across entries. This table is pivotal for queries retrieving movie details, evaluating popularity, and linking movies to their directors.

Your turn ->

Database name: {db_name}
Table Name: {table_name}
Create Table Statement + Sample:
```{create_statement}
```

Columns annotated information:
```
{annotated_columns_description}```
Table description: 
"""
    return get_completion_4(prompt)


def generate_general_table_descriptions(db_path: str = "dev/dev_databases"):
    # Create an empty dataframe with desired columns
    df = pd.DataFrame(columns=['db_name', 'table_name', 'description'])

    databases = sorted(glob.glob(f"{db_path}/*"))
    for db in databases:
        db_name = os.path.basename(db)
        db_uri = f"{db_path}/{db_name}/{db_name}.sqlite"
        db_descriptions_path = f"{db}/database_description"
        tables = get_table_names(db_descriptions_path)
        print(db_descriptions_path)
        for table in tables:
            create_statement = get_table_create_statement_with_sample(db_uri, table)
            annotated_columns_description = table_description_parser(db_descriptions_path, table)
            table_description = get_table_description_2(db_name, table, create_statement, annotated_columns_description)

            # Append data to the dataframe using pandas.concat
            new_row = pd.DataFrame({
                'db_name': [db_name],
                'table_name': [table],
                'description': [table_description],
            })
            df = pd.concat([df, new_row], ignore_index=True)

    df.to_csv('dev/dev_tables_description_2.csv', index=False)


def generate_general_table_descriptions_json(db_path: str = "dev/dev_databases"):
    data = []

    databases = sorted(glob.glob(f"{db_path}/*"))
    for db in databases:
        db_name = os.path.basename(db)
        db_uri = f"{db_path}/{db_name}/{db_name}.sqlite"
        db_descriptions_path = f"{db}/database_description"
        tables = get_table_names(db_descriptions_path)
        print(db_descriptions_path)

        for table in tables:
            create_statement = get_table_create_statement_with_sample(db_uri, table)
            annotated_columns_description = table_description_parser(db_descriptions_path, table)
            table_description = get_table_description_2(db_name, table, create_statement, annotated_columns_description)

            data.append({
                'db_name': db_name,
                'table_name': table,
                'description': table_description,
            })

    with open('dev/dev_tables_description_2.json', 'w') as f:
        json.dump(data, f, indent=4)

    # Testing: Try reading back the file to ensure everything is okay
    with open('dev/dev_tables_description_2.json', 'r') as f:
        loaded_data = json.load(f)
    print(loaded_data[:2])


def create_subset_of_dataset(input_dataset_path: str = "dev/dev.json", output_dataset_path: str = "dev/dev_subset.json"):
    # Read the JSON data from the file
    with open(input_dataset_path, 'r') as file:
        data = json.load(file)

    # A dictionary to hold the questions for each db_id
    db_questions = {}

    # Iterate through all entries, collecting questions for each db_id
    for entry in data:
        db_id = entry["db_id"]
        if db_id not in db_questions:
            db_questions[db_id] = []
        db_questions[db_id].append(entry)

    # Take the first 10 questions from each db_id
    subset_data = []
    for questions in db_questions.values():
        subset_data.extend(questions[:10])

    # Write the subset data to the new JSON file
    with open(output_dataset_path, 'w') as file:
        json.dump(subset_data, file, indent=4)

    print(f"Subset created and saved to {output_dataset_path}")


if __name__ == "__main__":
    # pass
    generate_general_table_descriptions_json()
    # create_subset_of_dataset()
