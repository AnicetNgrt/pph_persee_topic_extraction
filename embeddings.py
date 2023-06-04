import sys
import pandas as pd
from sentence_transformers import SentenceTransformer
import time
import math
import json

def to_json(col):
    return json.dumps(col.tolist())

def main(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

    # Calculate the total number of iterations
    total_iters = len(df)

    # Set the batch size for encoding
    batch_size = 100

    # Initialize the embeddings list and the current iteration
    embeddings = []
    current_iter = 0

    # Start the timer
    start_time = time.time()

    # Loop over the data in batches
    for i in range(0, total_iters, batch_size):        # Get the batch of data to encode
        batch_data = df['content'][i:i+batch_size].to_list()
        
        # Encode the batch and append to the embeddings list
        batch_embeddings = model.encode(batch_data)

        embeddings.extend(batch_embeddings)

        # Update the current iteration
        current_iter += len(batch_data)

        # Calculate the elapsed time and estimated remaining time
        elapsed_time = time.time() - start_time
        remaining_time = elapsed_time / current_iter * (total_iters - current_iter)

        # Print progress and estimated remaining time
        print(f"Processed {current_iter}/{total_iters} ({current_iter/total_iters*100:.2f}%), Estimated Remaining Time: {math.floor(remaining_time/60)} min {math.floor(remaining_time%60)} sec")

    # Assign the embeddings to the dataframe
    df['embeddings'] = embeddings
    df['embeddings'] = df['embeddings'].apply(to_json)

    df.to_csv(output_csv, index=False)


if __name__ == "__main__":
    input_csv = sys.argv[1]
    output_csv = sys.argv[2]
    main(input_csv, output_csv)