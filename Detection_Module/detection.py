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


# Read the CSV file
csv_file = 'Complete_dataset_detection_input.csv'
df = pd.read_csv(csv_file)


def heading(text: str) -> str:
    """Helper function for centering text."""
    return "\n" + f" {text} ".center(80, "=") + "\n"

load_dotenv()
client = Client(credentials=Credentials.from_env())
print(heading("Simple Text Generation"))
prompt_sentence = '''
You are a helpful AI assistant. You are given an answer to the question, \"{{Question}}\". This answer is divided into constituent individual sentence(s) (answer can also be just 1 sentence) in the following format:
S0: <sentence0>
S1: <sentence1>
S2: <sentence2>
.
.
.
You are also provided with information in the following table:
{{Table}}
Answer:
{{Response}}
Your task is to evaluate each sentence and determine whether it is factually consistent (FC) or factually inconsistent (FI) with the information provided in the table.\nAlso provide one line reasoning for each of your decisions. Please provide your evaluation in the following format:
L0: FC/FI, <reason> (corresponds to S0)
L1: FC/FI, <reason> (corresponds to S1)
L2: FC/FI, <reason> (corresponds to S2)
.
.
.
'''

incorrect_output = 'Incorrect_detection_output.txt'


def extract_labels(text):
    """Extract labels from the text where the word after 'Li:' is either 'FC' or 'FI'."""
    # Find all matches where the word after 'Li:' is either 'FC' or 'FI'
    matches = re.findall(r'L(\d+): (FC|FI)', text)
    
    # Initialize a dictionary to store labels with their indices
    labels_dict = {}
    
    # Iterate over the matches
    for match in matches:
        label_index = int(match[0])  # Get the index of the label (e.g., '1' for L1, '2' for L2)
        label = match[1]  # Get the word after 'Li:' (either 'FC' or 'FI')
        labels_dict[label_index] = label  # Store label with its index in the dictionary
    
    return labels_dict

def process_list(text):
    lines = text.strip().split('\n')
    processed_lines = [line.split(': ')[0] + ':' for line in lines]
    return '\n'.join(processed_lines)

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


def extract_reason(input_string):
    # Initialize an empty dictionary
    result = {}

    # Regular expression to find lines in the required format
    pattern = re.compile(r'L(\d+):\s*(FC|FI),\s*(.*)')

    # Find all matches in the input string
    matches = pattern.findall(input_string)

    # Process each match
    for match in matches:
        line_number = int(match[0])
        result_part = match[2].strip()
        result_part = split_into_sentences(result_part)[0]
        result[line_number] = result_part

    return result

import csv

# Assuming the extract_labels and extract_reason functions are defined
def extract_sentences(text):
    # This function extracts sentences from the given text
    sentences = {}
    
    # Regex pattern to match "S<number>: <sentence>"
    pattern = r'S(\d+):\s*(.*?)\s*(?=S\d+:|$)'

    # Find all matches in the text
    matches = re.findall(pattern, text, re.DOTALL)

    # Process each match
    for match in matches:
        key = int(match[0])
        sentence = match[1].strip()
        sentences[key] = sentence

    return sentences

true_labels = []
predicted_labels = []

no_answer_calls = 0

index = 0
num_extra_calls_allowed = 4
extra_calls_allowed = num_extra_calls_allowed
Total_calls = 0
Extra_calls = 0

with open(incorrect_output, 'a') as f1, open('Complete_dataset_detection_output.csv', 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['Index', 'Question', 'Table', 'Sentence', 'Label', 'Reason', 'Quest_index'])
    
    current_indexes = set()
    csv_index = 0
    
    while index < len(df):
        print(f"Total Calls: {Total_calls}")
        print(f"Extra Calls: {Extra_calls}")
        print(f"No answer calls: {no_answer_calls}")
        row = df.iloc[index]
        print(index)
        Total_calls += 1
        for response in client.text.generation.create(
            model_id="meta-llama/llama-3-70b-instruct",
            input=prompt_sentence,
            data={"Question": row.to_list()[0], "Table": row.to_list()[1], "Response": row.to_list()[2]},
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
            
            # Extract labels and reasons
            true_label_list = extract_labels(row.to_list()[3])
            predicted_label_list = extract_labels(generated_text)
            reason_list = extract_reason(generated_text)

            # Handle extra call flag logic
            if len(true_label_list) != len(predicted_label_list) and extra_calls_allowed > 0:
                Extra_calls += 1
                extra_calls_allowed -= 1
                continue

            print(f"Ground truth labels: {true_label_list}")
            print(f"Predicted labels: {predicted_label_list}")
            
            common_indices = set(true_label_list.keys()).intersection(predicted_label_list.keys())
            no_answer_calls += len(true_label_list) - len(common_indices)

            # Write to CSV for each sentence in row.to_list()[2]
            sentences = extract_sentences(row.to_list()[2])
            for key in sentences:
                if key not in predicted_label_list:
                    predicted_label_list[key]="FC"
                sentence = sentences[key]
                label = predicted_label_list.get(key, "")
                reason = reason_list.get(key, "Answer in inappropriate format")
                csvwriter.writerow([csv_index, row.to_list()[0], row.to_list()[1], sentence, label, reason, index])
                csv_index += 1

            # Extend true_labels and predicted_labels based on common indices
            write_flag = 0
            for key in true_label_list:
                if true_label_list[key] != predicted_label_list[key] and write_flag == 0:
                    write_flag = 1
                    f1.write(f"Input Text: {input_text}\n")
                    f1.write(f"Generated Answer: {generated_text}\n")
                    f1.write(f"Correct Labels: {true_label_list}")
                    f1.write("\n")
                true_labels.append(true_label_list[key])
                predicted_labels.append(predicted_label_list[key])

            index += 1
            extra_calls_allowed = num_extra_calls_allowed

           

print(f"Total Calls: {Total_calls}")
print(f"Extra Calls: {Extra_calls}")
print(f"No answer calls: {no_answer_calls}")

# Calculate metrics
accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels, average='weighted')
recall = recall_score(true_labels, predicted_labels, average='weighted')
f1 = f1_score(true_labels, predicted_labels, average='weighted')
report = classification_report(true_labels, predicted_labels, target_names=['FC', 'FI'])

# Print metrics
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print("Classification Report:")
print(report)

# Write metrics to a file``
with open('metrics_output_prompt4_v4.txt', 'w') as f:

    f.write(f"Total Calls: {Total_calls}\n")
    f.write(f"Extra Callse: {Extra_calls}\n")
    f.write(f"No answer calls: {no_answer_calls}\n")
    f.write(f"Accuracy: {accuracy}\n")
    f.write(f"Precision: {precision}\n")
    f.write(f"Recall: {recall}\n")
    f.write(f"F1 Score: {f1}\n")
    f.write("Classification Report:\n")
    f.write(report)



