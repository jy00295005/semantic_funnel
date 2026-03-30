"""
Round 1-2 Semantic Funnel pipeline.

Round 1 extracts raw technical terms from titles and abstracts.
Round 2 filters the Round 1 output and retains specific technical terms.
"""

import os
import json
import ast
import pandas as pd
from openai import OpenAI
from tqdm import tqdm
from pathlib import Path
from dotenv import load_dotenv
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import argparse

# Prompt templates
from prompts import (
    ROUND1_SYSTEM_PROMPT, 
    ROUND1_USER_PROMPT,
    ROUND2_SYSTEM_PROMPT,
    ROUND2_USER_PROMPT
)

# Load environment variables
load_dotenv()

# Configuration
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY') or os.getenv('ds_api_key')
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com")
MODEL_NAME = os.getenv("ROUND12_MODEL", "deepseek-chat")
DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
MAX_WORKERS = 5

# Ensure output directory exists
RESULTS_DIR.mkdir(exist_ok=True)

# Lock for safe incremental writes
write_lock = threading.Lock()


class TechnicalTermExtractor:
    """LLM wrapper for Round 1 and Round 2 extraction."""
    
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        """Initialize the extractor."""
        self.client = OpenAI(api_key=api_key, base_url=OPENAI_BASE_URL)
        self.model = model
    
    def call_llm(self, system_prompt: str, user_prompt: str, max_retries: int = 3) -> dict:
        """Call the LLM and parse the JSON response."""
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.2,
                )
                
                result = json.loads(response.choices[0].message.content)
                return result
                
            except Exception as e:
                print(f"\nAttempt {attempt + 1}/{max_retries} failed: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise
    
    def extract_round1(self, paper_id: int, title: str, abstract: str, 
                      research_area: str, research_topic: str, domain_name: str) -> dict:
        """Round 1: extract raw technical terms from a paper."""
        user_prompt = ROUND1_USER_PROMPT.format(
            research_area=research_area,
            research_topic=research_topic,
            domain_name=domain_name,
            title=title,
            abstract=abstract
        )
        
        result = self.call_llm(ROUND1_SYSTEM_PROMPT, user_prompt)
        
        return {
            'paper_id': paper_id,
            'title': title,
            'abstract': abstract,
            'research_area': research_area,
            'research_topic': research_topic,
            'domain': domain_name,
            'technical_terms': result.get('technical_terms', []),
            'terms_count': len(result.get('technical_terms', []))
        }
    
    def extract_round2(self, paper_id: int, terms_list: list, 
                      research_area: str, research_topic: str, domain_name: str) -> dict:
        """Round 2: filter Round 1 terms and retain specific technical terms."""
        if not terms_list:
            return {
                'paper_id': paper_id,
                'research_area': research_area,
                'research_topic': research_topic,
                'domain': domain_name,
                'important_terms': [],
                'removed_terms': [],
                'filtering_summary': {
                    'total_input': 0,
                    'retained': 0,
                    'removed': 0,
                    'retention_rate': '0%'
                }
            }
        
        # Format the term list
        terms_str = "\n".join([f"{i+1}. {term}" for i, term in enumerate(terms_list)])
        
        user_prompt = ROUND2_USER_PROMPT.format(
            research_area=research_area,
            research_topic=research_topic,
            domain_name=domain_name,
            terms_list=terms_str
        )
        
        result = self.call_llm(ROUND2_SYSTEM_PROMPT, user_prompt)
        
        return {
            'paper_id': paper_id,
            'research_area': research_area,
            'research_topic': research_topic,
            'domain': domain_name,
            'important_terms': result.get('important_terms', []),
            'removed_terms': result.get('removed_terms', []),
            'filtering_summary': result.get('filtering_summary', {})
        }


def process_domain(csv_file: Path, domain_name: str, extractor: TechnicalTermExtractor):
    """Process one input CSV through Round 1 and Round 2."""
    print(f"\n{'='*80}")
    print(f"Processing domain: {domain_name}")
    print(f"Input file: {csv_file}")
    print(f"{'='*80}\n")
    
    # Load input data
    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} papers")
    
    # Use the CSV stem as the dataset slug
    domain_slug = csv_file.stem
    round1_output = RESULTS_DIR / f"{domain_slug}_round1_results.csv"
    round2_output = RESULTS_DIR / f"{domain_slug}_round2_results.csv"
    
    # Resume from existing partial outputs if available
    processed_round1 = set()
    processed_round2 = set()
    
    if round1_output.exists():
        existing_df = pd.read_csv(round1_output)
        processed_round1 = set(existing_df['paper_id'].tolist())
        print(f"Round 1 already processed: {len(processed_round1)} papers")
    
    if round2_output.exists():
        existing_df = pd.read_csv(round2_output)
        processed_round2 = set(existing_df['paper_id'].tolist())
        print(f"Round 2 already processed: {len(processed_round2)} papers")
    
    # ==================== Round 1 ====================
    print(f"\n{'='*80}")
    print("Round 1: Raw term extraction")
    print(f"{'='*80}\n")
    
    remaining_papers = df[~df['ID'].isin(processed_round1)]
    
    if len(remaining_papers) > 0:
        print(f"Papers remaining: {len(remaining_papers)}")
        print(f"Workers: {MAX_WORKERS}")
        
        round1_results = []
        
        def process_paper_round1(row):
            """Run Round 1 for one paper."""
            try:
                result = extractor.extract_round1(
                    paper_id=row['ID'],
                    title=row['TI'],
                    abstract=row['AB'],
                    research_area=row['RaName_EN'],
                    research_topic=row['GroupName'],
                    domain_name=domain_name
                )
                return result
            except Exception as e:
                print(f"\nError processing paper {row['ID']}: {str(e)}")
                return None
        
        # Process papers concurrently
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_row = {executor.submit(process_paper_round1, row): row for _, row in remaining_papers.iterrows()}
            
            with tqdm(total=len(remaining_papers), desc="Round 1") as pbar:
                for future in as_completed(future_to_row):
                    result = future.result()
                    if result:
                        round1_results.append(result)
                        
                        # Flush every 10 papers
                        if len(round1_results) >= 10:
                            with write_lock:
                                temp_df = pd.DataFrame(round1_results)
                                temp_df.to_csv(round1_output, mode='a', header=not round1_output.exists(), index=False)
                                round1_results = []
                    
                    pbar.update(1)
        
        # Flush any remaining results
        if round1_results:
            with write_lock:
                temp_df = pd.DataFrame(round1_results)
                temp_df.to_csv(round1_output, mode='a', header=not round1_output.exists(), index=False)
        
        print(f"\nRound 1 complete. Results saved to: {round1_output}")
    else:
        print("Round 1 already complete for all papers")
    
    # ==================== Round 2 ====================
    print(f"\n{'='*80}")
    print("Round 2: Specificity filtering")
    print(f"{'='*80}\n")
    
    # Load the full Round 1 output
    if not round1_output.exists():
        print("Error: Round 1 results do not exist, so Round 2 cannot run")
        return
    
    round1_df = pd.read_csv(round1_output)
    remaining_papers_r2 = round1_df[~round1_df['paper_id'].isin(processed_round2)]
    
    if len(remaining_papers_r2) > 0:
        print(f"Papers remaining: {len(remaining_papers_r2)}")
        print(f"Workers: {MAX_WORKERS}")
        
        round2_results = []
        
        def process_paper_round2(row):
            """Run Round 2 for one paper."""
            try:
                # Parse the serialized term list from the Round 1 CSV
                terms_list = ast.literal_eval(row['technical_terms']) if isinstance(row['technical_terms'], str) else row['technical_terms']
                
                result = extractor.extract_round2(
                    paper_id=row['paper_id'],
                    terms_list=terms_list,
                    research_area=row['research_area'],
                    research_topic=row['research_topic'],
                    domain_name=row['domain']
                )
                return result
            except Exception as e:
                print(f"\nError processing paper {row['paper_id']}: {str(e)}")
                return None
        
        # Process papers concurrently
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_row = {executor.submit(process_paper_round2, row): row for _, row in remaining_papers_r2.iterrows()}
            
            with tqdm(total=len(remaining_papers_r2), desc="Round 2") as pbar:
                for future in as_completed(future_to_row):
                    result = future.result()
                    if result:
                        round2_results.append(result)
                        
                        # Flush every 10 papers
                        if len(round2_results) >= 10:
                            with write_lock:
                                temp_df = pd.DataFrame(round2_results)
                                temp_df.to_csv(round2_output, mode='a', header=not round2_output.exists(), index=False)
                                round2_results = []
                    
                    pbar.update(1)
        
        # Flush any remaining results
        if round2_results:
            with write_lock:
                temp_df = pd.DataFrame(round2_results)
                temp_df.to_csv(round2_output, mode='a', header=not round2_output.exists(), index=False)
        
        print(f"\nRound 2 complete. Results saved to: {round2_output}")
    else:
        print("Round 2 already complete for all papers")
    
    # ==================== Summary ====================
    print(f"\n{'='*80}")
    print("Processing summary")
    print(f"{'='*80}\n")
    
    round1_df = pd.read_csv(round1_output)
    round2_df = pd.read_csv(round2_output)
    
    print(f"Round 1 total papers: {len(round1_df)}")
    print(f"Round 1 total terms: {round1_df['terms_count'].sum()}")
    print(f"Round 1 mean terms per paper: {round1_df['terms_count'].mean():.2f}")
    
    print(f"\nRound 2 total papers: {len(round2_df)}")
    # Parse the serialized term lists to compute summary counts
    important_counts = []
    for _, row in round2_df.iterrows():
        terms = ast.literal_eval(row['important_terms']) if isinstance(row['important_terms'], str) else row['important_terms']
        important_counts.append(len(terms))
    
    print(f"Round 2 total retained terms: {sum(important_counts)}")
    print(f"Round 2 mean retained terms per paper: {sum(important_counts)/len(important_counts):.2f}")


def main():
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description='Round 1-2 Semantic Funnel extraction')
    parser.add_argument('--domain', type=str, required=True, help='Human-readable domain label')
    parser.add_argument('--input', type=str, required=True, help='Input CSV path')
    args = parser.parse_args()
    
    print("="*80)
    print("Round 1-2 Semantic Funnel extraction")
    print("="*80)
    
    if not DEEPSEEK_API_KEY:
        raise ValueError("Missing DEEPSEEK_API_KEY. Set it in the environment or .env file.")

    extractor = TechnicalTermExtractor(api_key=DEEPSEEK_API_KEY, model=MODEL_NAME)
    print(f"Model: {MODEL_NAME}\n")

    input_file = Path(args.input)
    if not input_file.exists():
        print(f"Error: input file not found - {input_file}")
        return 1

    process_domain(input_file, args.domain, extractor)
    
    print("\n" + "="*80)
    print("Round 1-2 processing complete")
    print("="*80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
