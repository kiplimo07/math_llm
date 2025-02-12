# solver.py
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain import LLMChain, PromptTemplate

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

# Example usage
problem = "Find the derivative of $f(x) = x^2 + 3x + 2$."
solution = llm_chain.run(problem)
print(solution) 