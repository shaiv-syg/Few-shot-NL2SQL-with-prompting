from azure_openai import get_completion_4

question = "how many accounts who choose issuance after transaction are staying in East Bohemia region?"

descriptions = """
Table: loan, "The 'loan' table in the 'financial' database stores information about various loans. Each record includes the loan ID, the associated account ID, the date the loan was approved, the approved amount in US dollars, the loan duration in months, the monthly payments, and the repayment status. The status can indicate whether the contract is finished or running and if there have been any payment issues."
Table: client, "The 'client' table in the 'financial' database stores information about the bank's clients. Each row represents a unique client, identified by a unique 'client_id', and includes details such as the client's gender, birth date, and the ID of the district where their branch is located. The 'district_id' is a foreign key that references the 'district' table."
Table: district, "The 'district' table in the 'financial' database contains detailed information about different districts. It includes data such as the district name, region, municipality hierarchy, ratio of urban inhabitants, average salary, unemployment rates for 1995 and 1996, number of entrepreneurs per 1000 inhabitants, and the number of committed crimes for 1995 and 1996. The 'district_id' serves as the primary key for this table."
Table: trans, "The 'trans' table in the 'financial' database stores information about various transactions. Each record includes details such as transaction id, account id, date of transaction, type of transaction (credit or withdrawal), mode of operation, amount of money involved in the transaction, and the balance after the transaction. The table also contains optional fields for the symbol, bank, and account related to the transaction."
Table: account, "The 'account' table in the 'financial' database holds information about various accounts. It includes details such as the account id, the location of the branch (district_id), the frequency of the account, and the creation date of the account. The account_id serves as the primary key, and the district_id is a foreign key referencing the district table."
Table: card, "The 'card' table in the 'financial' database stores information about credit cards. It includes details such as the card's unique ID, the associated disposition ID, the type of credit card (junior, classic, or gold), and the date the card was issued. The 'disp_id' column is a foreign key that references the 'disp' table."
Table: order, "The ""order"" table in the ""financial"" database is used to track individual financial transactions. Each row represents a unique order, identified by the ""order_id"", and includes details such as the account initiating the transaction (""account_id""), the recipient's bank and account (""bank_to"" and ""account_to""), the debited amount (""amount""), and the purpose of the payment (""k_symbol""). The ""k_symbol"" can indicate various types of payments such as insurance, household, leasing, or loan payments."
Table: disp, "The 'disp' table in the 'financial' database is used to manage the relationship between clients and their accounts. It contains unique identifiers for each record (disp_id), the client (client_id), and the account (account_id), as well as the type of disposition (type) which can be 'OWNER', 'USER', or 'DISPONENT'. The type of disposition indicates the level of control a client has over an account, such as the ability to issue orders or apply for loans."
"""

prompt = f"""
You are a sql query assistant schema linker. You are provided with a natural language question and a list of table descriptions.
Your task is to pick the relevant tables for the query.

Question: {question}

Table Descriptions:
{descriptions}

tables for query: """

print(get_completion_4(prompt))