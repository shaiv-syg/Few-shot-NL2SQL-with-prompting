from dfin.utils.azure_openai import get_completion_4
def extract_query_skeleton(question):
    prompt = f"""
Your task is to identify and extract potential column names from a natural language SQL query. Focus on retaining words that could correspond to column names in a database, while discarding common words, aggregate functions, and other irrelevant terms.

For example, in the query "Find the total revenue for each product category", the words "find", "the", "total", "for", "each" are irrelevant or any other word which is a sql function or an operator. The potential column names or key terms are:
revenue, product, category

Now, identify and list the potential column names or key terms from the following query, pick at most 5 terms:

    Query: "{question}"
    """

    # Make an API call to OpenAI to get the completion
    response_text = get_completion_4(prompt)

    return response_text.strip().split('\n')

# Example usage
question = "Show me the average number of people in each city where the population is greater than 100,000."
question = "What is the highest eligible free rate for K-12 students in the schools in Alameda County?"
extracted_terms = extract_query_skeleton(question)
print(extracted_terms)


from nltk.tokenize import word_tokenize

import inflect
p = inflect.engine()

import re

def split_camel_snake_case(text):
    # Split snake_case
    text = re.sub(r'_', ' ', text)
    # Split CamelCase
    text = re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', text)
    return text


def convert_to_singular(word):
    singular = p.singular_noun(word)
    return singular if singular else word

def clean_up_terms(terms):
    cleaned_terms = set()
    for term in terms:
        # Split camel case and snake case
        term = split_camel_snake_case(term)
        # Tokenize the term
        tokens = word_tokenize(term)
        # Filter out any unwanted characters or words, convert to lowercase, and convert to singular form
        cleaned_tokens = [convert_to_singular(token.lower()) for token in tokens if token.isalpha()]
        cleaned_terms.update(cleaned_tokens)
    return list(cleaned_terms)



# # Example usage
# cleaned_terms = clean_up_terms(extracted_terms)
# print(cleaned_terms)

print(clean_up_terms(["DoctorsName"]))


def keyword_match(query_tokens, preprocessed_columns):
    matches = {}
    for token in query_tokens:
        for column_name, column_tokens in preprocessed_columns.items():
            if token in column_tokens:
                matches[token] = column_name
                break  # Stop searching once a match is found for this token
    return matches
# Example usage

query = "Which doctor is the best?"
extracted_terms = extract_query_skeleton(query)
cleaned_terms = clean_up_terms(extracted_terms)

# Assuming preprocessed_columns is a dictionary of preprocessed column names from your database
preprocessed_columns = {

    "doctorsNames": ['doctor', 'name'],
    "politics": ['politics']
}

matches = keyword_match(cleaned_terms, preprocessed_columns)
print(matches)

