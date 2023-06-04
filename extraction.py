import sys
import pandas as pd
import re

def extract_year(title):
    years = re.findall(r'\b\d{4}\b', title)
    if years:
        return max(years)
    else:
        return None

def main(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    df['year'] = df['title2'].apply(extract_year)
    df = df[['title2', 'abstract', 'year']]
    df = df.rename(columns={'title2': 'title', 'abstract': 'content'})
    df.to_csv(output_csv, index=False)


if __name__ == "__main__":
    input_csv = sys.argv[1]
    output_csv = sys.argv[2]
    main(input_csv, output_csv)