

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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report


load_dotenv()
client = Client(credentials=Credentials.from_env())


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

def format_response(sentences):
    # Create the S0, S1, ... format
    formatted_sentences = '\n'.join([f'S{i}: {sentence.strip()}' for i, sentence in enumerate(sentences)])
    return formatted_sentences


def extract_labels(text):
    """Extract labels from the text where the word after 'Li:' is either 'FC' or 'FI'."""
    # Find all matches where the word after 'Li:' is either 'FC' or 'FI'
    matches = re.findall(r'L(\d+): (FC|FI)', text)
    
    # Initialize a dictionary to store labels with their indices
    labels_dict = {}
    
    # Iterate over the matches
    for match in matches:
        label_index = match[0]  # Get the index of the label (e.g., '1' for L1, '2' for L2)
        label = match[1]  # Get the word after 'Li:' (either 'FC' or 'FI')
        labels_dict[label_index] = label  # Store label with its index in the dictionary
    
    return labels_dict


def detection(Question, Table, Response, num_extra_calls):
    for response in client.text.generation.create(
        model_id="meta-llama/llama-3-70b-instruct",
        input="You are a helpful AI assistant. You are given an answer to the question, \"{{Question}}\". This answer is divided into constituent individual sentence(s) (answer can also be just 1 sentence) in the following format:\nS0: <sentence0>\nS1: <sentence1>\nS2: <sentence2>\n.\n.\n.\nYou are also provided with information in the following table:\n{{Table}}\nAnswer:\n{{Response}}\nYour task is to evaluate each sentence and determine whether it is factually consistent (FC) or factually inconsistent (FI) with the information provided in the table.\nAlso provide one line reasoning for each of your decisions. Please provide your evaluation in the following format:\nL0: FC/FI, <reason> (corresponds to S0)\nL1: FC/FI, <reason> (corresponds to S1)\nL2: FC/FI, <reason> (corresponds to S2)\n.\n.\n.",
        data={"Question": Question, "Table": Table, "Response": Response},
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

        sentence_pattern = r'S(\d+):\s*(.*?)\s*(?=S\d+:|$)'
        num_sentences = len(re.findall(sentence_pattern, Response))

        labels_dict = extract_labels(generated_text)
        

        if len(labels_dict)!=num_sentences and num_extra_calls>0:
            print("Extra Call is being made in detection module")
            num_extra_calls -= 1
            labels_dict = detection(Question, Table, Response, num_extra_calls)
        return labels_dict
            

def reformat_mitigation_input(string, label_dictionary):

    sentence_pattern = r'S(\d+):\s*(.*?)\s*(?=S\d+:|$)'
    sentences = re.findall(sentence_pattern, string)
    processed_sentences = []

    for sentence in sentences:
        sentence_number = sentence[0]
        sentence_text = sentence[1]

        # Check if the sentence number is in the label_dictionary
        if sentence_number in label_dictionary:
            label_text = "Factually Inconsistent" if label_dictionary[sentence_number] == "FI" else "Factually Consistent"
        else:
            label_text = "Factually Inconsistent"

        processed_sentence = f"S{sentence_number}: ({label_text}) {sentence_text}"
        processed_sentences.append(processed_sentence)

    formatted_input = '\n'.join(processed_sentences)

    return formatted_input

def create_sentences_dict(input_string):
    # Define the regex pattern to match sentences
    sentence_pattern = r'S(\d+):\s*(.*?)\s*(?=S\d+:|$)'

    # Find all matches
    matches = re.findall(sentence_pattern, input_string)

    # Define the regex pattern to remove unwanted phrases
    cleanup_pattern = r'^\s*\((?:Factually Consistent|Factually Inconsistent)\)\s*'

    # Create the dictionary and clean the sentences
    sentences_dict = {int(key): re.sub(cleanup_pattern, '', sentence).strip() for key, sentence in matches}

    return sentences_dict

def find_factually_inconsistent_statements_indices(string):

    sentence_pattern = r'S(\d+):\s*(.*?)\s*(?=S\d+:|$)'
    # Find all matches
    matches = re.findall(sentence_pattern, string)

    indices = []

    for match in matches:
        if "(Factually Inconsistent)" in match[1]:
            indices.append(int(match[0]))

    return indices
    

def mitigation(Question, Table, Response, num_extra_calls):
    for response in client.text.generation.create(
        model_id="meta-llama/llama-3-70b-instruct",
        input="You are a helpful AI assistant. You are given a response to the question, \"{{Question}}\". This response should be based on the information present inside the following table:\n{{Table}}\nThis response is divided into the following sentences:\n{{Response}}\nSome of these sentences, indicated at their start with (Factually Inconcistent), are factually inconsistent. Your task is to correct these factually inconsistent sentences using the information given in the table. Correct sentences are included for context and do not need to be mentioned in your output. Provide only the corrected sentences in the order they appear in the response.\nStrictly follow this output Format (in order):\nS<index of first incosistent sentence>: <Corrected sentence1>\nS<index of second incosistent sentence>: <Corrected sentence2>\n.\n.\nS<index of last incosistent sentence>: <Last corrected sentence>",
        data={"Question": Question, "Table": Table, "Response": Response},
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


        original_sentences_dict = create_sentences_dict(Response)
        corrected_sentences_dict = create_sentences_dict(generated_text)

        factually_inconsistent_statements_indices = find_factually_inconsistent_statements_indices(Response)

        final_sentences_dict = {}

        if len(corrected_sentences_dict)!=len(factually_inconsistent_statements_indices) and num_extra_calls>0:
            print("Extra Call is being made in mitigation module")
            num_extra_calls -= 1
            final_sentences_dict = mitigation(Question, Table, Response, num_extra_calls)
            print("Output is coming from the extra mitigation call")
        else:
            for key in original_sentences_dict:
                if key in factually_inconsistent_statements_indices:
                    final_sentences_dict[key] = corrected_sentences_dict.get(key, original_sentences_dict.get(key, ""))
                else:
                    final_sentences_dict[key] = original_sentences_dict.get(key, "")
            
            
            # print(f"Generated Answer: {generated_text}\n")
        return final_sentences_dict
    
    
def concatenate_sentences(sentence_dict):
    # Sort the dictionary by keys to ensure the order is correct
    sorted_sentences = [sentence_dict[key] for key in sorted(sentence_dict.keys())]
    
    # Join the sentences with a space
    result_string = ' '.join(sorted_sentences)
    
    return result_string

def Detection_Mitigation_wrapper(Question, Table, Response):
    detection_num_extra_calls_allowed = 1
    mitigation_num_extra_calls_allowed = 1
    sentences = split_into_sentences(Response)
    formatted_response = format_response(sentences)
    labels_dict = detection(Question, Table, formatted_response, detection_num_extra_calls_allowed)
    print(labels_dict)
    if 'FI' in labels_dict.values():
        mitigation_input = reformat_mitigation_input(formatted_response, labels_dict)
        mitigation_output = mitigation(Question, Table, mitigation_input, mitigation_num_extra_calls_allowed)
        final_output = concatenate_sentences(mitigation_output)
    else:
        print("Mitigation call not made.")
        final_output = Response
    return final_output

input_file = '/Users/praddy/code/Fast_fit/end_to_end_testing_data.csv'
df = pd.read_csv(input_file)

num_detection_extra_calls_allowed = 1
num_mitigation_extra_calls_allowed = 1

check_count = 100
output_file = 'End_to_end_results.txt'

# Open the output file in write mode
with open(output_file, 'w') as f:
    # Iterate through the first 100 rows
    for i in range(check_count):
        # Extract index, third and fourth column values
        row = df.iloc[i].to_list()
        Question = row[0]
        Table = row[1]
        Answer = row[2]
        Generated_answer = Detection_Mitigation_wrapper(Question, Table, Answer)
        Correct_answer = row[3]

        print(f"Input answer:\n {Answer}\n")
        print(f"Generated Answer:\n {Generated_answer}\n")

        # Write index, third and fourth column values to the file
        f.write(f'Row index: {i}\n')
        f.write(f'Original response:\n {Answer}\n')
        f.write(f'Generated response:\n {Generated_answer}\n')
        f.write(f'Correct response:\n {Correct_answer}\n\n')