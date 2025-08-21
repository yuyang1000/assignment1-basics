import transformers
from transformers import pipeline

if __name__ == "__main__":
    generator = pipeline("text-generation", model="HuggingFaceTB/SmolLM2-360M")
    result = generator(
        "life is full of hard，",
        max_length=128,
        num_return_sequences=2,
    )
    print(result)



