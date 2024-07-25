import ollama
import os
import time
import json

models = ['phi', 'llama2-uncensored', 'llama2:13b', 'gemma:7b', 'mistral']
medical_reports_folder = os.path.join(os.getcwd(), 'Medical Reports')
medical_reports_results_folder = os.path.join(os.getcwd(), 'Medical Reports Results')

# Create the folders if they do not exist
if not os.path.exists(medical_reports_folder):
    os.makedirs(medical_reports_folder)
if not os.path.exists(medical_reports_results_folder):
    os.makedirs(medical_reports_results_folder)

# Search all medical reports in the directory
def list_files(directory):
    if not os.path.exists(directory):
        return "The directory does not exist."
    all_entries = os.listdir(directory)
    files = [entry for entry in all_entries if os.path.isfile(os.path.join(directory, entry))]
    return files

file_names = list_files(medical_reports_folder)

# Dictionary to store file names and their corresponding processing times
processing_times = {}

for model in models:  # Iterate over each model
    for medical_report in file_names:  # Iterate over each medical report
        # Start timing
        start_time = time.time()

        with open(os.path.join(medical_reports_folder, medical_report), 'r') as file:
            print(f"I'm reading: {medical_report}")
            medical_report_content = file.read()

        with open("prompt.txt", 'r') as file:
            prompt = file.read()

        # Setting up the model, enabling streaming responses, and defining the input messages
        ollama_response = ollama.chat(model=model, messages=[
            {'role': 'system', 'content': prompt},
            {'role': 'user', 'content': medical_report_content},
        ],
        options={'temperature': 0})

        # Printing out of the generated response
        response = ollama_response['message']['content']

        # Create filename with model name included
        model_name = model.replace(':', '-')
        new_filename = f"{os.path.splitext(medical_report)[0]}_{model_name}{os.path.splitext(medical_report)[1]}"
        with open(os.path.join(medical_reports_results_folder, new_filename), 'w') as file:
            file.write(response)
            print(f"I'm writing the response for {medical_report} using model {model_name}")

        # End timing
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Total time taken for {medical_report}: {elapsed_time} seconds")

        # Store the file name and elapsed time in the dictionary
        processing_times[(model, medical_report)] = elapsed_time

# Output final processing times
print("===== TIME TO COMPLETE =====")
print(processing_times)

# Convert the dictionary with string keys to a JSON formatted string
processing_times_str_keys = {f"{model}|{report}": time for (model, report), time in processing_times.items()}

with open(os.path.join(medical_reports_results_folder, "processing_times.json"), 'w') as file:
    json_str = json.dumps(processing_times_str_keys, indent=4)
    file.write(json_str)
