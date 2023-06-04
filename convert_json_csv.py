import json
import csv
import codecs
import sys

input = sys.argv[1]
output_csv = sys.argv[2]

# Load the JSON data from the SPARQL query result
with open(input) as f:
    data = json.load(f)

# Get the list of bindings from the "results" section of the data
bindings = data['results']['bindings']

# Open a new CSV file for writing
with open(output_csv, 'w', newline='') as f:
    writer = csv.writer(f)

    # Write the header row
    writer.writerow(['abstract', 'title2'])

    # Iterate over the bindings and write each row to the CSV
    for binding in bindings:
        # Convert escape sequences to real characters except for double quotes
        abstract = binding['abstract']['value'].encode('utf-8').decode('utf-8').replace('"', ' ')
        title2 = binding['title2']['value'].encode('utf-8').decode('utf-8').replace('"', ' ')
        writer.writerow([abstract, title2])