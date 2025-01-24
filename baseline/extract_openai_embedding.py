from openai import OpenAI
import torch 

os.environ["OPENAI_API_KEY"] = "<MASKED>"
openai.api_key = "<MASKED>"
client = OpenAI()

# Initialize tokenizer for the specific model
tokenizer = tiktoken.encoding_for_model("text-embedding-3-large")  # Use the correct embedding model

def truncate_text(text, max_tokens, tokenizer):
    """
    Truncates text to fit within the specified max_tokens limit.
    """
    tokens = tokenizer.encode(text)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]  # Truncate to max_tokens
    return tokenizer.decode(tokens)


def get_embedding(text, model="text-embedding-3-large"):
    text = truncate_text(text, 8100, tokenizer) # set to a value slightly under the max token count value 
    return client.embeddings.create(input = [text], model=model).data[0].embedding

#embedding = get_embedding(sample_text)  
#embedding = torch.tensor(embedding).float() 
#embedding = embedding.unsqueeze(0)  # Shape: [1, 3072]
#embedding.shape

# Check date range
min_date = min(grouped_news.keys())
max_date = max(grouped_news.keys())
print(f"Date range in grouped_news: {min_date} to {max_date}")

# Create embeddings
all_embeddings = []
for date in tqdm(sorted(grouped_news.keys()), desc="Processing embeddings"):
    text = grouped_news[date]
    embedding = get_embedding(text)  # Generate embedding
    embedding_tensor = torch.tensor(embedding).float().unsqueeze(0)  # Shape: [1, 3072]
    all_embeddings.append(embedding_tensor)

# Stack embeddings into a tensor of shape [n, 3072]
final_embeddings = torch.cat(all_embeddings, dim=0)

# Save embeddings
torch.save(final_embeddings, "/content/drive/MyDrive/grouped_news_embeddings.pt")
print(f"Final embeddings shape: {final_embeddings.shape}")
print("Embeddings saved to '/content/drive/MyDrive/grouped_news_embeddings.pt'")
