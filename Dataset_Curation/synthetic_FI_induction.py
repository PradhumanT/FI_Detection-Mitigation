from dotenv import load_dotenv
from genai.client import Client
from genai.credentials import Credentials
from genai.schema import (
    TextGenerationParameters,
    TextGenerationReturnOptions,
)
import openpyxl
import pandas as pd
import re

import pandas as pd

# Read the list of indexes from the .txt file
index_file = '/Users/praddy/code/dataset_curation3/temp_FI.txt'
with open(index_file, 'r') as file:
    selected_indexes = [int(line.strip()) for line in file]



# Read the CSV file
csv_file = '/Users/praddy/code/dataset_curation3/sent_tokenized_all_FC_dataset.csv'
big_df = pd.read_csv(csv_file)

# Select the rows corresponding to the last 600 indexes and the first 3 columns
selected_rows = big_df.iloc[selected_indexes, :4]

# Create a DataFrame with the selected rows and columns
df = pd.DataFrame(selected_rows)


def heading(text: str) -> str:
    """Helper function for centering text."""
    return "\n" + f" {text} ".center(80, "=") + "\n"

load_dotenv()
client = Client(credentials=Credentials.from_env())
print(heading("Simple Text Generation"))
prompt_sentence = "Rewrite the given sentence which is part of the answer to the question, \"{{Question}}\" to make it factually inconsistent with the provided table. Ensure that the newly generated statement does not mean the same thing as the original sentence, even if the original sentence is already factually inconsistent with the table. No need of giving any explanation.\n Table: {{Data}}\n Sentence: {{Sentence}} \nOutput Format:\nFactually inconsistent statement: <Factually inconsistent statement>"

factually_inconcistent_output_file = '/Users/praddy/code/Dataset_curation3/all_factually_inconcistent_outputs_file.txt'
factually_inconcistent_statements_file = '/Users/praddy/code/Dataset_curation3/all_factually_inconcistent_statements_file.txt'

def split_into_sentences(text):
    # Step 1: Replace periods after numbers with a unique token
    text = re.sub(r'(?:(?<=^)|(?<=\n))(\d+)\.\s+', r'\1__PERIOD__', text)
    
    # Step 2: Replace periods in abbreviations with a unique token
    text = re.sub(r'(\b[A-Z])\.\s*', r'\1__DOT__', text)
    
    # Step 3: Replace periods after any abbreviation that are followed by a lowercase letter with a unique token
    text = re.sub(r'(\b[A-Za-z]{1,4})\.(?=\s*[a-z])', r'\1__DOT__', text)
    
    # Step 4: Tokenize the text into sentences
    sentences = re.split(r'(?<=[.?!])\s+', text)
    
    # Step 5: Replace the unique tokens back to periods in sentences
    corrected_sentences = []
    for sentence in sentences:
        corrected_sentence = re.sub(r'(\d+)__PERIOD__', r'\1. ', sentence)
        corrected_sentence = re.sub(r'(\b[A-Z])__DOT__', r'\1. ', corrected_sentence)
        corrected_sentence = re.sub(r'(\b[A-Za-z]{1,4})__DOT__', r'\1.', corrected_sentence)
        corrected_sentences.append(corrected_sentence.strip())
    
    return corrected_sentences


with open(factually_inconcistent_output_file, 'a') as f1, open(factually_inconcistent_statements_file, 'a') as f2:
    current_indexes = set()
    for index, row in df.iterrows():  # Convert set to list here
    
        for response in client.text.generation.create(
            model_id="meta-llama/llama-3-70b-instruct",
            input=prompt_sentence,
            data={"Question": row.to_list()[1]  ,"Data": row.to_list()[2], "Sentence": row.to_list()[3]},
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
            
            # Write to file
            
    
            f1.write(f"Input Text: {input_text}\n")
            f1.write(f"Generated Text: {generated_text}\n")
            f1.write("\n")

            placeholder = "Factually inconsistent statement: ".lower()
            if placeholder in generated_text.lower():
                print("Hi")
                # Find the start index of the placeholder in a case-insensitive manner
                start_index = generated_text.lower().index(placeholder) + len(placeholder)
                # Extract the substring starting from the end of the placeholder
                inconcistent_statement = generated_text[start_index:].strip()
                # Split at the first period to get only the first sentence
                inconcistent_statement = split_into_sentences(inconcistent_statement)[0]
                print(inconcistent_statement)
                f2.write(f"{row.to_list()[0]} {inconcistent_statement}\n")
            else:
                continue



# You can generate better quality and better representing factual inconcistencies by also passing questions.


