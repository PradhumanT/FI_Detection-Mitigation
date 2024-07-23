from dotenv import load_dotenv
from genai.client import Client
from genai.credentials import Credentials
from genai.schema import (
    TextGenerationParameters,
    TextGenerationReturnOptions,
)

import pandas as pd
import re


file_path = 'Complete_dataset_mitigation_input.csv'
df = pd.read_csv(file_path)


def heading(text: str) -> str:
    """Helper function for centering text."""
    return "\n" + f" {text} ".center(80, "=") + "\n"

load_dotenv()
client = Client(credentials=Credentials.from_env())
print(heading("Simple Text Generation"))
prompt_sentence = "You are a helpful AI assistant. You are given a response to the question, \"{{Question}}\". This response should be based on the information present inside the following table:\n{{Table}}\nThis response is divided into the following sentences:\n{{Response}}\nSome of these sentences, indicated at their start with (Factually Inconcistent), are factually inconsistent. Your task is to correct these factually inconsistent sentences using the information given in the table. Correct sentences are included for context and do not need to be mentioned in your output. Provide only the corrected sentences in the order they appear in the response.\nStrictly follow this output Format (in order):\nS<index of first incosistent sentence>: <Corrected sentence1>\nS<index of second incosistent sentence>: <Corrected sentence2>\n.\n.\nS<index of last incosistent sentence>: <Last corrected sentence>"


def find_fi_indexes(input_string):
    pattern = r'L(\d+):\s+FI'
    fi_indexes = []
    
    matches = re.finditer(pattern, input_string)
    for match in matches:
        index = int(match.group(1))
        fi_indexes.append(index)
    
    return fi_indexes

def extract_reasons(input_string, index_list):
    # Compile regex to match lines starting with S followed by digits and a colon
    pattern = re.compile(r'^(R\d+): (.+)$', re.MULTILINE)
    reasons = []
    
    # Find all matches in the input string
    matches = pattern.findall(input_string)
    
    # Iterate over the matches and filter based on the index list
    for index_part, reason in matches:
        index = int(index_part[1:])  # Extract numeric part and convert to int
        if index in index_list:
            reasons.append(f'{index_part}: {reason}')
    
    return '\n'.join(reasons)


def create_sentences_dict(input_string):
    # Define the regex pattern to match sentences
    sentence_pattern = r'S(\d+):\s*(.*?)\s*(?=S\d+:|$)'

    # Find all matches
    matches = re.findall(sentence_pattern, input_string, re.DOTALL)

    # Define the regex pattern to remove unwanted phrases
    cleanup_pattern = r'^\s*\((?:Factually Consistent|Factually Inconsistent)\)\s*'

    # Create the dictionary and clean the sentences
    sentences_dict = {int(key): re.sub(cleanup_pattern, '', sentence).strip() for key, sentence in matches}

    return sentences_dict

def create_reasons_dict(input_string):
    # Define the regex pattern
    reason_pattern = r'R(\d+):\s*(.*?)\s*(?=R\d+:|$)'

    # Find all matches
    matches = re.findall(reason_pattern, input_string, re.DOTALL)

    # Create the dictionary
    reasons_dict = {int(key): reason for key, reason in matches}

    return reasons_dict

def create_Labels_dict(input_string):
    # Define the regex pattern
    Label_pattern = r'L(\d+):\s*(.*?)\s*(?=L\d+:|$)'

    # Find all matches
    matches = re.findall(Label_pattern, input_string, re.DOTALL)

    # Create the dictionary
    Labels_dict = {int(key): Label for key, Label in matches}

    return Labels_dict

import csv

mitigation_output_file = 'Complete_dataset_mitigation_output.csv'

# Define the header for the CSV file
header = ["Index", "Question_Index", "Question", "Table", "Sentence", "Label", "Reason", "Corrected_Sentence"]

# Open the CSV file in write mode and write the header
with open(mitigation_output_file, 'w', newline='') as f1:
    writer = csv.writer(f1)
    writer.writerow(header)

ind = 0
index = 0
num_extra_calls_allowed = 0
extra_calls_allowed = num_extra_calls_allowed
total_calls = 0
no_response_calls = 0
num_fi_responses = 0


# Iterate through the DataFrame and populate the CSV file

with open(mitigation_output_file, 'a', newline='') as f1:
    writer = csv.writer(f1)
    while index < len(df):
        row = df.iloc[index]
        print(f"{index}")
        FI_indices = find_fi_indexes(row.to_list()[4])
        Labels_dict = create_Labels_dict(row.to_list()[4])
        Sentences_dict = create_sentences_dict(row.to_list()[3])
        Reasons_dict = create_reasons_dict(row.to_list()[5])
        Corrected_sentences_dict = {}

        if len(FI_indices) != 0:
            total_calls += 1
            num_fi_responses += 1
            print(f"Total calls: {total_calls}")
            for response in client.text.generation.create(
                model_id="meta-llama/llama-3-70b-instruct",
                input=prompt_sentence,
                data={"Question": row.to_list()[1], "Table": row.to_list()[2], "Response": row.to_list()[3]},
                parameters=TextGenerationParameters(
                    max_new_tokens=250,
                    min_new_tokens=20,
                    return_options=TextGenerationReturnOptions(
                        input_text=True,
                    ),
                ),
            ):
                result = response.results[0]
                input_text = result.input_text
                generated_text = result.generated_text

                # Print to console
                print(f"Generated Answer: {generated_text}\n")

            Corrected_sentences_dict = create_sentences_dict(generated_text)

            if not all(index in Corrected_sentences_dict for index in FI_indices) and extra_calls_allowed>=0:
                if extra_calls_allowed==0:
                    no_response_calls += 1
                else:
                    extra_calls_allowed -= 1
                    num_fi_responses -= 1
                    continue
        
            ind += 1

        extra_calls_allowed = num_extra_calls_allowed
        index += 1
        

print(f"Number of datapoints: {num_fi_responses}")
print(f"Total number of calls made: {total_calls}")
print(f"Number of datapoints with incorrect output format: {no_response_calls}")


