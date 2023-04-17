import sys
from llama_index import GPTSimpleVectorIndex, LLMPredictor
from langchain import OpenAI

llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, max_tokens=350))
index = GPTSimpleVectorIndex.load_from_disk("index.json")

args = sys.argv
question = args[1]
print("Q: "+ question)
output = index.query(question)
print("A: ", end="")
print(output)
