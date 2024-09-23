import openai
import json
import pandas as pd
import random
from typing import List, Dict, Any, Optional
from tqdm import tqdm

class AdvancedChainOfThoughtAnalyzer:
    def __init__(self, api_key: Optional[str] = None):
        if api_key is None:
            api_key = ""
        
        if not api_key:
            raise ValueError("API key not provided and OPENAI_API_KEY environment variable not set")
        
        self.client = openai.OpenAI(api_key=api_key)
    
    def _make_api_call(self, messages: List[Dict[str, str]], max_retries: int = 3) -> str:
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4",
                    messages=messages,
                    temperature=0.7,
                    max_tokens=3000
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"API call failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    raise Exception(f"Max retries reached. API call failed: {e}")
        return ""

    def _parse_json(self, content: str, max_retries: int = 3) -> Dict:
        for attempt in range(max_retries):
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                if attempt < max_retries - 1:
                    print(f"Error parsing JSON (attempt {attempt + 1}/{max_retries}). Attempting to reformat...")
                    content = self._reformat_output(content)
                else:
                    print(f"Failed to parse JSON after {max_retries} attempts.")
                    return {}
        return {}

    def _reformat_output(self, content: str) -> str:
        system_prompt = "You are an AI assistant that reformats text into valid JSON. Your task is to take the given content and ensure it is properly formatted as JSON, preserving all the information and structure."
        
        user_prompt = f"""
        The following content needs to be reformatted to ensure it's valid JSON:

        {content}

        Please reformat this content into valid JSON, preserving all the information. 
        Ensure all quotes are properly escaped and the structure is correct.
        If you encounter any issues, try to fix them or omit problematic parts while maintaining the overall structure.
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        return self._make_api_call(messages)

    def generate_advanced_cot_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        results = []
        
        for _, row in tqdm(df.iterrows(), total=len(df)):
            question = row['question']
            answer = row['answer']
            
            cot_data = self._generate_advanced_cot(question, answer)
            
            for cot in cot_data:
                results.append({
                    'question': question,
                    'answer': answer,
                    'cot_reasoning': self._format_cot(cot),
                    'cot_type': cot.get('type', 'unknown'),
                    'reflection': cot.get('reflection', ''),
                    'corrected_cot': cot.get('corrected_cot', ''),
                    'solution': cot.get('solution', '')
                })
        
        return pd.DataFrame(results)

    def _generate_advanced_cot(self, question: str, answer: str) -> List[Dict[str, Any]]:
        system_prompt = """You are an AI assistant that generates multiple chains of thought for reflection and learning purposes. Your output should include one correct chain, two incorrect chains, and two subtly incorrect chains."""
        
        user_prompt = f"""
        Given the following question and answer:
        Question: {question}
        Answer: {answer}
        
        Generate 5 different chains of thought to solve this question:
        1. One correct chain
        2. Two clearly incorrect chains
        3. Two subtly incorrect chains

        For each chain:
        1. Start with the initial approach or understanding of the problem.
        2. Detail the step-by-step reasoning process.
        3. Include any assumptions made during the process.
        4. Conclude with the final answer derived from the chain.
        5. Provide a reflection on the chain, discussing its strengths or weaknesses.
        6. For incorrect chains, suggest a corrected chain of thought.
        7. Provide the correct solution to the problem.

        Format each chain of thought as a JSON object with the following structure:
        {{
            "type": "correct" | "incorrect" | "subtly_incorrect",
            "approach": "Initial understanding or approach to the problem",
            "steps": [
                "Step 1: ...",
                "Step 2: ...",
                ...
            ],
            "assumptions": [
                "Assumption 1: ...",
                "Assumption 2: ...",
                ...
            ],
            "conclusion": "Final answer derived from this chain",
            "reflection": "Reflection on the strengths or weaknesses of this chain",
            "corrected_cot": "Suggested correct chain of thought (for incorrect chains)",
            "solution": "The correct solution to the problem"
        }}

        Ensure that your response is a valid JSON array containing 5 chain objects.
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            content = self._make_api_call(messages)
            chains = self._parse_json(content)
            
            if not isinstance(chains, list):
                print("Error: Generated content is not a list. Attempting to fix...")
                chains = [chains] if isinstance(chains, dict) else []
            
            return chains
        except Exception as e:
            print(f"Error generating advanced chains of thought: {e}")
            return []

    def _format_cot(self, chain: Dict[str, Any]) -> str:
        cot = f"Approach: {chain.get('approach', 'N/A')}\n\n"
        cot += "Steps:\n" + "\n".join(chain.get('steps', ['N/A'])) + "\n\n"
        cot += "Assumptions:\n" + "\n".join(chain.get('assumptions', ['N/A'])) + "\n\n"
        cot += f"Conclusion: {chain.get('conclusion', 'N/A')}"
        return cot

def generate_dummy_data(num_questions: int = 5) -> pd.DataFrame:
    questions = [
        "What is the capital of France?",
        "If a train travels at 60 mph for 2 hours, how far does it go?",
        "What is the square root of 144?",
        "Who wrote 'Romeo and Juliet'?",
        "What is the boiling point of water in Celsius?",
        "How many continents are there on Earth?",
        "What is the chemical symbol for gold?",
        "Who painted the Mona Lisa?",
        "What is the largest planet in our solar system?",
        "What year did World War II end?"
    ]
    
    answers = [
        "Paris",
        "120 miles",
        "12",
        "William Shakespeare",
        "100 degrees Celsius",
        "7",
        "Au",
        "Leonardo da Vinci",
        "Jupiter",
        "1945"
    ]
    
    selected_indices = random.sample(range(len(questions)), min(num_questions, len(questions)))
    
    dummy_data = {
        'question': [questions[i] for i in selected_indices],
        'answer': [answers[i] for i in selected_indices]
    }
    
    return pd.DataFrame(dummy_data)

def main():
    try:
        analyzer = AdvancedChainOfThoughtAnalyzer()

        # Generate dummy data
        print("Generating dummy data...")
        dummy_df = generate_dummy_data(num_questions=3)
        print("Dummy data generated:")
        print(dummy_df)

        print("\nGenerating Advanced CoT dataset...")
        advanced_cot_dataset = analyzer.generate_advanced_cot_dataset(dummy_df)

        advanced_cot_dataset.to_csv('advanced_cot_dataset.csv', index=False)
        print("\nAdvanced CoT dataset generated and saved to 'advanced_cot_dataset.csv'")
        print(advanced_cot_dataset.head())

    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
