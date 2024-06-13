from huggingface_hub import snapshot_download

def download_pretrained_language_model(model_type, cache_dir):
    snapshot_download(repo_id=model_type, ignore_patterns=["*.h5", "*.ot", "*.msgpack"], cache_dir=cache_dir)

def main():
    output_cache_dir = "cache/"
    model_list = [
        "sentence-transformers/all-MiniLM-L6-v2", # Embedding model for similarity
        "cointegrated/roberta-large-cola-krishna2020", # CoLA fluency classifier
        "hallisky/cds_style_classifier" # Style classifier for the CDS dataset
    ]

    for model_name in model_list:
        download_pretrained_language_model(model_name, output_cache_dir)

if __name__ == "__main__":
    main()