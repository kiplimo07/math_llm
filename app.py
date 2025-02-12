# app.py
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain import LLMChain, PromptTemplate

app = FastAPI()

# Load the fine-tuned Mistral 7B model and tokenizer
model = AutoModelForCausalLM.from_pretrained("./models/fine-tuned-mistral-7b")
tokenizer = AutoTokenizer.from_pretrained("./models/fine-tuned-mistral-7b")

# Define the prompt template
prompt_template = """
Solve the following problem step by step:
Problem: {problem}
Solution:
"""

prompt = PromptTemplate(template=prompt_template, input_variables=["problem"])

# Initialize the LLMChain
llm_chain = LLMChain(llm=model, prompt=prompt)

# Define input schema
class ProblemRequest(BaseModel):
    problem: str

# Define the API endpoint
@app.post("/solve")
def solve(request: ProblemRequest):
    problem = request.problem
    solution = llm_chain.run(problem)
    return {"solution": solution}

# Run the API
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)