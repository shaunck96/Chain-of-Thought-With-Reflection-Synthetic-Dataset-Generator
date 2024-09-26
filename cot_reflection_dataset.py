import openai
import json
import pandas as pd
import random
from typing import List, Dict, Any, Optional
from tqdm import tqdm

import os
import json
import time
import logging
import uuid
from typing import List, Dict, Any, Optional
import argparse
import yaml
import openai
import pandas as pd
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from pydantic import BaseModel, Field, ValidationError
from concurrent.futures import ThreadPoolExecutor, as_completed

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Config(BaseModel):
    api_key: str = Field(..., env="OPENAI_API_KEY")
    model: str = Field(default="gpt-4", env="OPENAI_MODEL")
    temperature: float = Field(default=0.8, env="TEMPERATURE")
    max_tokens: int = Field(default=3000, env="MAX_TOKENS")
    rate_limit_pause: float = Field(default=1.0, env="RATE_LIMIT_PAUSE")
    max_retries: int = Field(default=3, env="MAX_RETRIES")
    num_threads: int = Field(default=5, env="NUM_THREADS")

class Chain(BaseModel):
    reasoning: str = Field(..., min_length=50)
    conclusion: str = Field(..., min_length=10)
    confidence: float = Field(..., ge=0, le=1)

class OutputValidator(BaseModel):
    chains: List[Chain] = Field(..., min_items=5, max_items=5)

class OpenAIClient:
    def __init__(self, config: Config):
        self.client = openai.OpenAI(api_key=config.api_key)
        self.config = config

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=(retry_if_exception_type(openai.RateLimitError) | retry_if_exception_type(openai.APIError))
    )
    def make_api_call(self, messages: List[Dict[str, str]]) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            return response.choices[0].message.content
        except openai.RateLimitError:
            logger.warning("Rate limit reached. Retrying...")
            raise
        except openai.APIError as e:
            if "model_not_found" in str(e):
                logger.error(f"Model '{self.config.model}' not found. Trying fallback model.")
                self.config.model = "gpt-3.5-turbo"  # Fallback to a different model
                raise
            logger.error(f"API error: {e}")
            raise
        except Exception as e:
            logger.error(f"API call failed: {e}")
            raise

class AdvancedChainOfThoughtAnalyzer:
    def __init__(self, config: Config):
        self.config = config
        self.client = OpenAIClient(config)

    def generate_advanced_cot_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        results = []
        
        with ThreadPoolExecutor(max_workers=self.config.num_threads) as executor:
            future_to_row = {executor.submit(self._process_row, row): row for _, row in df.iterrows()}
            for future in tqdm(as_completed(future_to_row), total=len(df)):
                results.extend(future.result())
        
        if not results:
            logger.warning("No results generated. Returning empty DataFrame.")
            return pd.DataFrame(columns=['question_id', 'question', 'answer', 'cot_reasoning', 'cot_conclusion', 'cot_confidence', 'cot_index', 'is_corrected'])
        
        return pd.DataFrame(results)

    def _process_row(self, row):
        question_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, row['question']))
        question = row['question']
        answer = row['answer']
        
        cot_data = self._generate_advanced_cot(question, answer)
        
        results = []
        for i, cot in enumerate(cot_data):
            corrected_cot = self.reflect_and_correct(question, answer, cot)
            results.append({
                'question_id': question_id,
                'question': question,
                'answer': answer,
                'cot_reasoning': corrected_cot.reasoning,
                'cot_conclusion': corrected_cot.conclusion,
                'cot_confidence': corrected_cot.confidence,
                'cot_index': i,
                'is_corrected': corrected_cot != cot
            })
        
        time.sleep(self.config.rate_limit_pause)
        return results

    def _generate_advanced_cot(self, question: str, answer: str) -> List[Chain]:
        if not isinstance(question, str) or not isinstance(answer, str):
            raise ValueError("Question and answer must be strings")

        system_prompt = """You are an AI assistant that generates multiple chains of thought for a given question. Your task is to simulate realistic reasoning processes, including occasional subtle errors or misconceptions that might naturally occur in human or AI reasoning."""

        user_prompt = f"""
        Given the following question and answer:
        Question: {question}
        Answer: {answer}

        Generate 5 different chains of thought to answer this question. Each chain should include:
        1. A detailed reasoning process
        2. A conclusion based on that reasoning
        3. A confidence score (0-1) representing how sure you are about the conclusion

        Your output should be a valid JSON object with a 'chains' key containing an array of 5 objects. Each object should have the following structure:

        {{
          "chains": [
            {{
              "reasoning": "Detailed step-by-step thought process...",
              "conclusion": "The final answer derived from the reasoning",
              "confidence": 0.85
            }},
            // ... (4 more similar objects)
          ]
        }}

        Important guidelines:
        1. Vary the level of correctness and confidence across the chains.
        2. Include some chains with subtle errors or misconceptions.
        3. Make the errors natural and not obviously wrong.
        4. Ensure that higher confidence doesn't always correlate with correctness.
        5. Vary the length and detail of the reasoning processes.
        6. Use a conversational tone in the reasoning, as if thinking out loud.
        """

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        max_retries = self.config.max_retries
        for attempt in range(max_retries):
            try:
                content = self.client.make_api_call(messages)
                parsed_content = json.loads(content)
                validated_output = OutputValidator(**parsed_content)
                return validated_output.chains

            except json.JSONDecodeError as e:
                logger.warning(f"Attempt {attempt + 1}: Invalid JSON response: {e}")
            except ValidationError as e:
                logger.warning(f"Attempt {attempt + 1}: Pydantic validation error: {e}")
            except Exception as e:
                logger.error(f"Attempt {attempt + 1}: Error generating advanced chains of thought: {e}")

        logger.error("Failed to generate valid advanced chains of thought after multiple attempts")
        return []

    def reflect_and_correct(self, question: str, answer: str, chain: Chain) -> Chain:
        if chain.conclusion.lower() != answer.lower():
            system_prompt = """You are an AI assistant tasked with reflecting on and correcting an incorrect chain of thought. Your goal is to identify the error in the reasoning, explain why it's incorrect, and provide a corrected chain of thought."""

            user_prompt = f"""
            Given the following question, correct answer, and incorrect chain of thought:
            
            Question: {question}
            Correct Answer: {answer}
            
            Incorrect Chain of Thought:
            Reasoning: {chain.reasoning}
            Conclusion: {chain.conclusion}
            Confidence: {chain.confidence}
            
            Please provide:
            1. A reflection on why the original reasoning was incorrect
            2. A corrected chain of thought with the correct conclusion
            3. An updated confidence score

            Your response should be a valid JSON object with the following structure:
            {{
              "reflection": "Explanation of why the original reasoning was incorrect...",
              "corrected_reasoning": "Step-by-step correct reasoning process...",
              "corrected_conclusion": "The correct conclusion based on the new reasoning",
              "corrected_confidence": 0.95
            }}
            """

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]

            try:
                content = self.client.make_api_call(messages)
                correction_data = json.loads(content)
                
                return Chain(
                    reasoning=f"Original reasoning: {chain.reasoning}\n\nReflection: {correction_data['reflection']}\n\nCorrected reasoning: {correction_data['corrected_reasoning']}",
                    conclusion=correction_data['corrected_conclusion'],
                    confidence=correction_data['corrected_confidence']
                )
            except Exception as e:
                logger.error(f"Error in reflection and correction: {e}")
                return chain
        else:
            return chain

    @staticmethod
    def print_cot_dataset(df: pd.DataFrame):
        for _, group in df.groupby('question_id'):
            question = group['question'].iloc[0]
            answer = group['answer'].iloc[0]
            
            print(f"\n{'='*80}")
            print(f"Question: {question}")
            print(f"Answer: {answer}")
            print(f"{'='*80}")
            
            for _, row in group.iterrows():
                print(f"\nChain of Thought #{row['cot_index'] + 1}")
                print(f"{'='*40}")
                print(f"Reasoning:\n{row['cot_reasoning']}")
                print(f"\nConclusion: {row['cot_conclusion']}")
                print(f"Confidence: {row['cot_confidence']:.2f}")
                print(f"Is Corrected: {row['is_corrected']}")
                print(f"{'='*40}")
            
            print("\n")

    @staticmethod
    def evaluate_cot_dataset(df: pd.DataFrame, correct_answers: Dict[str, str]) -> pd.DataFrame:
        print(f"Overall Evaluation Metrics:")
        print(f"{'='*40}")
        
        evaluation_results = []
        
        total_chains = len(df)
        total_correct = 0
        total_confidence = 0
        high_confidence_correct = 0
        low_confidence_incorrect = 0
        total_corrected = 0
        
        high_confidence_threshold = 0.8
        low_confidence_threshold = 0.3
        
        for question, correct_answer in correct_answers.items():
            question_df = df[df['question'] == question]
            
            correct_conclusions = sum(question_df['cot_conclusion'].str.lower() == correct_answer.lower())
            total_correct += correct_conclusions
            total_confidence += question_df['cot_confidence'].sum()
            
            high_conf_correct = sum((question_df['cot_confidence'] >= high_confidence_threshold) & 
                                    (question_df['cot_conclusion'].str.lower() == correct_answer.lower()))
            low_conf_incorrect = sum((question_df['cot_confidence'] <= low_confidence_threshold) & 
                                     (question_df['cot_conclusion'].str.lower() != correct_answer.lower()))
            
            high_confidence_correct += high_conf_correct
            low_confidence_incorrect += low_conf_incorrect
            
            corrected_chains = sum(question_df['is_corrected'])
            total_corrected += corrected_chains
            
            avg_confidence = question_df['cot_confidence'].mean()
            avg_reasoning_length = question_df['cot_reasoning'].str.len().mean()
            
            evaluation_results.append({
                'question': question,
                'correct_answer': correct_answer,
                'total_chains': len(question_df),
                'correct_conclusions': correct_conclusions,
                'correct_percentage': correct_conclusions / len(question_df),
                'average_confidence': avg_confidence,
                'high_confidence_correct': high_conf_correct,
                'low_confidence_incorrect': low_conf_incorrect,
                'corrected_chains': corrected_chains,
                'average_reasoning_length': avg_reasoning_length
            })
            
            print(f"\nQuestion: {question}")
            print(f"Correct Answer: {correct_answer}")
            print(f"Total Chains: {len(question_df)}")
            print(f"Correct Conclusions: {correct_conclusions} ({correct_conclusions/len(question_df):.2%})")
            print(f"Corrected Chains: {corrected_chains}")
            print(f"Average Confidence: {avg_confidence:.2f}")
            print(f"{'='*40}")
        
        avg_confidence = total_confidence / total_chains
        
        print(f"\nOverall Statistics:")
        print(f"Total Questions: {len(correct_answers)}")
        print(f"Total Chains of Thought: {total_chains}")
        print(f"Overall Correct Conclusions: {total_correct} ({total_correct/total_chains:.2%})")
        print(f"Overall Corrected Chains: {total_corrected} ({total_corrected/total_chains:.2%})")
        print(f"Overall Average Confidence: {avg_confidence:.2f}")
        print(f"High Confidence & Correct: {high_confidence_correct}")
        print(f"Low Confidence & Incorrect: {low_confidence_incorrect}")
        
        reasoning_lengths = df['cot_reasoning'].str.len()
        print(f"\nReasoning Analysis:")
        print(f"Min Reasoning Length: {reasoning_lengths.min()} characters")
        print(f"Max Reasoning Length: {reasoning_lengths.max()} characters")
        
        print(f"\nConfidence Distribution:")
        confidence_bins = pd.cut(df['cot_confidence'], bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0])
        print(confidence_bins.value_counts().sort_index())

        return pd.DataFrame(evaluation_results)

    @staticmethod
    def save_to_csv(df: pd.DataFrame, filename: str):
        try:
            df.to_csv(filename, index=False)
            logger.info(f"Data successfully saved to {filename}")
        except Exception as e:
            logger.error(f"Error saving data to CSV: {e}")

def load_config(config_path: str) -> Dict[str, Any]:
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return {}

def run_cot_analysis(config_path: str = "config.yaml", num_samples: int = 10):
    config_dict = load_config(config_path)
    config = Config(**config_dict)

    # Initialize analyzer
    analyzer = AdvancedChainOfThoughtAnalyzer(config)

    # Load and preprocess data
    splits = {'test': 'data/test-00000-of-00001.parquet', 'validation': 'data/validation-00000-of-00001.parquet', 'train': 'data/train-00000-of-00001.parquet'}
    df = pd.read_parquet("hf://datasets/microsoft/wiki_qa/" + splits["test"])
    df = df[['question', 'answer']].head(num_samples)

    # Generate CoT dataset
    logger.info("Generating Chain of Thought dataset")
    advanced_cot_dataset = analyzer.generate_advanced_cot_dataset(df)

    # Save the dataset to a CSV file
    output_file = "cot_dataset.csv"
    analyzer.save_to_csv(advanced_cot_dataset, output_file)

    # Evaluate the dataset
    logger.info("Evaluating the Generated Dataset:")
    correct_answers = dict(zip(df['question'], df['answer']))
    evaluation_results = analyzer.evaluate_cot_dataset(advanced_cot_dataset, correct_answers)

    # Save the evaluation results to a CSV file
    eval_output_file = "evaluation_results.csv"
    analyzer.save_to_csv(evaluation_results, eval_output_file)

    return advanced_cot_dataset, evaluation_results

# Uncomment and run the following lines in your notebook:
config_path = "/Workspace/Users/shaunshib96@outlook.com/ai_news/cot_dataset_generator/config.yaml"
num_samples = 3000  # Adjust as needed
cot_dataset, evaluation_results = run_cot_analysis(config_path, num_samples)
