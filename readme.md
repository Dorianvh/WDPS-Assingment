
# Entity Linking Script

This repository contains the `main.py` script, which processes input questions, generates answers using a language model, and links recognized entities to relevant Wikidata pages. The results are saved in a structured output file.

## How to Use

The `main.py` script processes input questions and generates answers along with linked entities. It requires two command-line arguments:

### Command-Line Arguments

#### 1. `-infile` or `-if`  
Specifies the path and name of the input file containing the questions to be processed.  
- The input file should follow this format:
  ```
  question-1\tWhat is the capital of France?
  question-2\tWho wrote 'To Kill a Mockingbird'?
  ```

#### 2. `-outfile` or `-of`  
Specifies the path and name of the output file where the script will save the results.  
- The output includes raw responses, entity links, and other information. If the file already exists, it will be overwritten.

### Example Usage

```bash
python code/main.py -if code/example_input.txt -of code/outputfile.txt
```

This command processes the file `code/example_input.txt` and saves the results to `code/outputfile.txt`.

## Features

- **Named Entity Recognition (NER):** The script uses spaCy to extract entities from the language model's answers.
- **Entity Linking:** Extracted entities are matched with relevant Wikidata pages based on popularity (measured by the number of sitelinks).
- **Customizable Language Model:** The script integrates with `llama_cpp` for generating answers to questions.

## Input File Format

The input file should be a tab-separated text file with each line containing:
- A unique question ID
- A question text

**Example Input File:**
```
question-1\tWhat is the capital of France?
question-2\tWho is the CEO of Tesla?
```

## Output File Format

The output file contains the following information:
- **Raw Responses:** The language model's answers to the questions.
- **Extracted Entities:** Entities recognized in the response.
- **Linked Wikidata Pages:** URLs to relevant Wikidata pages for the recognized entities.

**Example Output File:**
```
question-1\tR"What is the capital of France?"
question-1\tE"France"\t"https://en.wikipedia.org/wiki/France"
question-1\tA"Paris"
```

## Prerequisites

- Python 3.8 or higher
- Required Python libraries:  
  - `spacy`
  - `argparse`
  - `subprocess`
  - `requests`
  - `llama_cpp`

## Installation

1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure the spaCy model is installed:
   ```bash
   python -m spacy download en_core_web_sm
   ```

## How It Works

1. The script reads the input file for questions.
2. For each question:
   - It generates a response using the specified language model.
   - Performs Named Entity Recognition (NER) on the response.
   - Links extracted entities to Wikidata pages using the Wikidata API.
3. The results are saved to the specified output file.

