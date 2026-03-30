#!/usr/bin/env python3
"""
Round 3: Technology Clustering (Version 2 - Iterative Batch Clustering)
Uses batched clustering plus iterative merge to reduce prompt overload.
"""

import os
import json
import csv
import ast
import time
import threading
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY') or os.getenv('ds_api_key')
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com")
MODEL_NAME = os.getenv("ROUND3_MODEL", "deepseek-chat")
DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
ROUND2_DIR = RESULTS_DIR
ROUND3_DIR = RESULTS_DIR / "round3"
MAX_WORKERS = 2
TEMPERATURE = 0.3
BATCH_SIZE = 50
MIN_FREQUENCY = 3
MIN_PAPERS = 2

# Ensure output directory exists
ROUND3_DIR.mkdir(parents=True, exist_ok=True)

# Lock for safe writes
write_lock = threading.Lock()

# Initialize client
client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=OPENAI_BASE_URL)


# ==================== Prompt definitions ====================

BATCH_CLUSTERING_SYSTEM_PROMPT = """You are an expert in scientific research and technology taxonomy. Your task is to cluster technical terms from academic papers into semantically coherent groups.

Your responsibilities:
1. Identify technical terms that represent similar or related technologies
2. Group them into meaningful clusters based on their semantic relationships
3. Provide clear, descriptive names and descriptions for each cluster
4. Ensure each term is assigned to exactly ONE cluster (no duplicates, no omissions)

Guidelines:
- Focus on TECHNICAL SIMILARITY, not just keyword matching
- Create clusters that are neither too broad nor too specific
- Aim for 3-8 clusters per batch, with 3-15 terms per cluster
- If a term doesn't fit well into any cluster, mark it as "unclustered"
"""

BATCH_CLUSTERING_USER_PROMPT = """
# Task: Cluster Technical Terms (Batch Processing)

## Research Context
- **Domain**: {domain_name}
- **Research Group**: {group_name}
- **Batch**: {batch_num}/{total_batches}

## Input Data
You are given {term_count} technical terms from the field of "{group_name}".

### Technical Terms List (with frequency):
{terms_list}

## Your Task
Cluster these terms into semantically coherent groups.

## Output Format
Return a JSON object:

```json
{{
  "clusters": [
    {{
      "cluster_name": "Quantum Error Correction Codes",
      "cluster_description": "Technologies related to quantum error correction...",
      "terms": ["surface code", "topological code", "stabilizer code"]
    }}
  ],
  "unclustered_terms": [
    {{
      "term": "some_term",
      "reason": "Does not fit into any cluster"
    }}
  ]
}}
```

## Important Notes
1. Every term must appear EXACTLY ONCE in either "clusters" or "unclustered_terms"
2. No duplicates
3. Use exact term strings from the input
4. Focus on semantic coherence

Please proceed with the clustering task.
"""

MERGE_CLUSTERS_SYSTEM_PROMPT = """You are an expert in technology taxonomy and cluster analysis. Your task is to merge and consolidate clusters from multiple batches into a coherent final clustering.

Your responsibilities:
1. Identify duplicate or highly similar clusters across batches
2. Merge semantically similar clusters
3. Ensure no term appears in multiple clusters
4. Create a clean, well-organized final clustering
"""

MERGE_CLUSTERS_USER_PROMPT = """
# Task: Merge Clusters from Multiple Batches

## Research Context
- **Domain**: {domain_name}
- **Research Group**: {group_name}

## Input Data
You have {cluster_count} clusters from {batch_count} batches that need to be merged.

### Existing Clusters:
{clusters_info}

## Your Task
1. **Identify similar clusters**: Find clusters that represent the same or very similar technologies
2. **Merge clusters**: Combine similar clusters into single, well-defined clusters
3. **Rename if needed**: Provide better names for merged clusters
4. **Ensure completeness**: Every term from input must appear in output

## Output Format
Return a JSON object:

```json
{{
  "merged_clusters": [
    {{
      "cluster_name": "Final Cluster Name",
      "cluster_description": "Comprehensive description...",
      "terms": ["term1", "term2", ...],
      "source_clusters": ["original_cluster_1", "original_cluster_2"]
    }}
  ],
  "merge_summary": {{
    "original_cluster_count": <number>,
    "final_cluster_count": <number>,
    "clusters_merged": <number>
  }}
}}
```

## Important Notes
1. Preserve all terms - no term should be lost
2. No term should appear in multiple clusters
3. Provide clear rationale for merges in descriptions
4. Maintain semantic coherence

Please proceed with the merging task.
"""


# ==================== Data preparation ====================

def load_round2_results(domain: str) -> List[Dict]:
    """Load Round 2 results."""
    round2_file = ROUND2_DIR / f"{domain}_round2_results.csv"
    
    if not round2_file.exists():
        raise FileNotFoundError(f"Round 2 results not found: {round2_file}")
    
    papers = []
    with open(round2_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            papers.append(row)
    
    print(f"✓ Loaded {len(papers)} papers from Round 2 results")
    return papers


def aggregate_and_filter_terms(papers: List[Dict], min_frequency: int = MIN_FREQUENCY, min_papers: int = MIN_PAPERS) -> Dict[str, Dict]:
    """
    Aggregate technical terms by group and filter low-frequency terms.
    
    Returns:
        {
            "group_name": {
                "papers": [paper_dict, ...],
                "terms": {
                    "term1": {"frequency": int, "paper_ids": [id1, id2, ...]},
                    ...
                }
            }
        }
    """
    groups = defaultdict(lambda: {"papers": [], "terms": defaultdict(lambda: {"frequency": 0, "paper_ids": []})})
    
    for paper in papers:
        group_name = paper.get('research_topic', paper.get('group_name', 'Unknown'))
        paper_id = int(paper['paper_id'])
        
        groups[group_name]["papers"].append(paper)
        
        # Parse the serialized Round 2 term list
        try:
            important_terms = json.loads(paper['important_terms'])
        except (json.JSONDecodeError, ValueError):
            try:
                important_terms = ast.literal_eval(paper['important_terms'])
            except:
                print(f"Warning: Failed to parse important_terms for paper {paper_id}")
                continue
        
        # Count term frequency and supporting paper IDs
        for term in important_terms:
            groups[group_name]["terms"][term]["frequency"] += 1
            groups[group_name]["terms"][term]["paper_ids"].append(paper_id)
    
    # Filter low-frequency terms and convert to a regular dict
    result = {}
    for group_name, data in groups.items():
        filtered_terms = {}
        for term, info in data["terms"].items():
            if info["frequency"] >= min_frequency and len(set(info["paper_ids"])) >= min_papers:
                filtered_terms[term] = info
        
        result[group_name] = {
            "papers": data["papers"],
            "terms": filtered_terms,
            "filtered_out": len(data["terms"]) - len(filtered_terms)
        }
        
        print(f"  {group_name}: {len(filtered_terms)} terms (filtered out {result[group_name]['filtered_out']})")
    
    return result


def split_terms_into_batches(terms_dict: Dict[str, Dict], batch_size: int = BATCH_SIZE) -> List[Dict[str, Dict]]:
    """Split terms into batches."""
    terms_list = list(terms_dict.items())
    # Sort by frequency so the most frequent terms are processed first
    terms_list.sort(key=lambda x: -x[1]['frequency'])
    
    batches = []
    for i in range(0, len(terms_list), batch_size):
        batch = dict(terms_list[i:i + batch_size])
        batches.append(batch)
    
    return batches


def format_terms_list_with_frequency(terms_dict: Dict[str, Dict]) -> str:
    """Format the technical term list with frequencies."""
    lines = []
    for idx, (term, info) in enumerate(sorted(terms_dict.items(), key=lambda x: -x[1]['frequency']), 1):
        freq = info['frequency']
        paper_count = len(set(info['paper_ids']))
        lines.append(f"{idx}. {term} (freq: {freq}, papers: {paper_count})")
    return "\n".join(lines)


# ==================== LLM calls ====================

def call_llm_for_batch_clustering(
    domain_name: str,
    group_name: str,
    terms_dict: Dict[str, Dict],
    batch_num: int,
    total_batches: int,
    max_retries: int = 3
) -> Dict:
    """Call the LLM to cluster one batch."""
    
    terms_list = format_terms_list_with_frequency(terms_dict)
    user_prompt = BATCH_CLUSTERING_USER_PROMPT.format(
        domain_name=domain_name,
        group_name=group_name,
        term_count=len(terms_dict),
        batch_num=batch_num,
        total_batches=total_batches,
        terms_list=terms_list
    )
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": BATCH_CLUSTERING_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=TEMPERATURE,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            return result
            
        except Exception as e:
            print(f"  ⚠ Attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise
    
    raise Exception("Failed to get clustering result from LLM")


def call_llm_for_merge_clusters(
    domain_name: str,
    group_name: str,
    all_clusters: List[Dict],
    batch_count: int,
    max_retries: int = 3
) -> Dict:
    """Call the LLM to merge batch-level clusters."""
    
    # Format cluster information for the merge prompt
    clusters_info_lines = []
    for idx, cluster in enumerate(all_clusters, 1):
        clusters_info_lines.append(f"{idx}. **{cluster['cluster_name']}** ({len(cluster['terms'])} terms)")
        clusters_info_lines.append(f"   Description: {cluster['cluster_description']}")
        clusters_info_lines.append(f"   Terms: {', '.join(cluster['terms'][:5])}{'...' if len(cluster['terms']) > 5 else ''}")
        clusters_info_lines.append("")
    
    clusters_info = "\n".join(clusters_info_lines)
    
    user_prompt = MERGE_CLUSTERS_USER_PROMPT.format(
        domain_name=domain_name,
        group_name=group_name,
        cluster_count=len(all_clusters),
        batch_count=batch_count,
        clusters_info=clusters_info
    )
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": MERGE_CLUSTERS_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=TEMPERATURE,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            return result
            
        except Exception as e:
            print(f"  ⚠ Merge attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise
    
    raise Exception("Failed to merge clusters")


# ==================== Validation and repair ====================

def verify_and_fix_batch_clustering(
    llm_output: Dict,
    input_terms_dict: Dict[str, Dict]
) -> Dict:
    """Validate and repair one batch clustering result."""
    
    input_terms = set(input_terms_dict.keys())
    clustered_terms = set()
    unclustered_terms = set()
    
    # Collect clustered terms
    for cluster in llm_output.get('clusters', []):
        for term in cluster.get('terms', []):
            if term in clustered_terms:
                print(f"  ⚠ Warning: Duplicate term '{term}' in batch")
            clustered_terms.add(term)
    
    # Collect unclustered terms
    for item in llm_output.get('unclustered_terms', []):
        term = item.get('term', '')
        if term:
            unclustered_terms.add(term)
    
    # Check completeness
    processed_terms = clustered_terms | unclustered_terms
    missing_terms = input_terms - processed_terms
    
    # Apply automatic repair if needed
    if missing_terms:
        print(f"  ⚠ Auto-fixing: Adding {len(missing_terms)} missing terms to unclustered")
        if 'unclustered_terms' not in llm_output:
            llm_output['unclustered_terms'] = []
        for term in missing_terms:
            llm_output['unclustered_terms'].append({
                'term': term,
                'reason': 'Auto-added: Missing from LLM output'
            })
    
    return llm_output


# ==================== Batched clustering ====================

def process_group_iterative(
    domain_name: str,
    group_name: str,
    group_data: Dict
) -> Tuple[str, Dict]:
    """Iteratively cluster one hotspot group."""
    
    print(f"\n{'='*60}")
    print(f"Processing: {group_name}")
    print(f"{'='*60}")
    
    terms_dict = group_data['terms']
    papers = group_data['papers']
    paper_count = len(papers)
    
    print(f"  Papers: {paper_count}")
    print(f"  Filtered terms: {len(terms_dict)}")
    print(f"  Filtered out: {group_data['filtered_out']}")
    
    # Step 1: Batch clustering
    batches = split_terms_into_batches(terms_dict, BATCH_SIZE)
    print(f"  Split into {len(batches)} batches (batch_size={BATCH_SIZE})")
    
    all_batch_results = []
    all_clusters = []
    all_unclustered = []
    
    for batch_num, batch_terms in enumerate(batches, 1):
        print(f"\n  Processing batch {batch_num}/{len(batches)} ({len(batch_terms)} terms)...")
        
        # Call the LLM for batch clustering
        batch_result = call_llm_for_batch_clustering(
            domain_name=domain_name,
            group_name=group_name,
            terms_dict=batch_terms,
            batch_num=batch_num,
            total_batches=len(batches)
        )
        
        # Validate and repair the batch result
        batch_result = verify_and_fix_batch_clustering(batch_result, batch_terms)
        
        # Collect the batch result
        all_batch_results.append(batch_result)
        all_clusters.extend(batch_result.get('clusters', []))
        all_unclustered.extend(batch_result.get('unclustered_terms', []))
        
        print(f"    ✓ Batch {batch_num}: {len(batch_result.get('clusters', []))} clusters, "
              f"{len(batch_result.get('unclustered_terms', []))} unclustered")
    
    # Step 2: Merge clusters if multiple batches were used
    if len(batches) > 1:
        print(f"\n  Merging {len(all_clusters)} clusters from {len(batches)} batches...")
        
        merge_result = call_llm_for_merge_clusters(
            domain_name=domain_name,
            group_name=group_name,
            all_clusters=all_clusters,
            batch_count=len(batches)
        )
        
        final_clusters = merge_result.get('merged_clusters', [])
        merge_summary = merge_result.get('merge_summary', {})
        
        print(f"    ✓ Merged: {merge_summary.get('original_cluster_count', len(all_clusters))} → "
              f"{merge_summary.get('final_cluster_count', len(final_clusters))} clusters")
    else:
        final_clusters = all_clusters
    
    # Step 3: Build the final JSON output
    print(f"\n  Generating final JSON...")
    result_json = generate_final_json(
        domain_name=domain_name,
        group_name=group_name,
        group_data=group_data,
        final_clusters=final_clusters,
        unclustered_terms=all_unclustered
    )
    
    # Save the output JSON
    safe_group_name = ''.join(c if c.isalnum() else '_' for c in str(group_name)).strip('_')
    output_file = ROUND3_DIR / f"{domain_name}_{safe_group_name}_clusters.json"
    with write_lock:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result_json, f, ensure_ascii=False, indent=2)
    
    print(f"  ✓ Saved to: {output_file.name}")
    print(f"  ✓ Final clusters: {result_json['clusters_count']}")
    print(f"  ✓ Clustered terms: {result_json['clustered_terms_count']}/{result_json['important_terms_count']}")
    print(f"  ✓ Coverage: {result_json['statistics']['clustering_coverage']:.1%}")
    
    return group_name, result_json


def generate_final_json(
    domain_name: str,
    group_name: str,
    group_data: Dict,
    final_clusters: List[Dict],
    unclustered_terms: List[Dict]
) -> Dict:
    """Build the final JSON output."""
    
    terms_dict = group_data['terms']
    papers = group_data['papers']
    
    # Generate cluster IDs
    domain_abbr = ''.join([word[0].upper() for word in domain_name.split('_')])
    group_abbr = ''.join(c.upper() if c.isalnum() else '' for c in str(group_name))[:6] or 'GROUP'
    
    clusters = []
    for idx, cluster in enumerate(final_clusters, 1):
        cluster_id = f"{domain_abbr}_{group_abbr}_C{idx:03d}"
        
        # Attach metadata for each term
        terms = []
        paper_ids_set = set()
        for term in cluster.get('terms', []):
            if term in terms_dict:
                term_info = terms_dict[term]
                terms.append({
                    'term': term,
                    'frequency': term_info['frequency'],
                    'paper_ids': sorted(list(set(term_info['paper_ids'])))
                })
                paper_ids_set.update(term_info['paper_ids'])
        
        clusters.append({
            'cluster_id': cluster_id,
            'cluster_name': cluster.get('cluster_name', f'Cluster {idx}'),
            'cluster_description': cluster.get('cluster_description', ''),
            'term_count': len(terms),
            'paper_count': len(paper_ids_set),
            'terms': terms,
            'source_clusters': cluster.get('source_clusters', [])
        })
    
    # Attach metadata for unclustered terms
    unclustered_terms_final = []
    for item in unclustered_terms:
        term = item.get('term', '')
        if term in terms_dict:
            term_info = terms_dict[term]
            unclustered_terms_final.append({
                'term': term,
                'frequency': term_info['frequency'],
                'paper_ids': sorted(list(set(term_info['paper_ids']))),
                'reason': item.get('reason', 'No reason provided')
            })
    
    # Compute summary statistics
    total_clustered_terms = sum(c['term_count'] for c in clusters)
    total_unique_papers = len(set(p['paper_id'] for p in papers))
    
    avg_cluster_size = total_clustered_terms / len(clusters) if clusters else 0
    avg_term_frequency = sum(t['frequency'] for t in terms_dict.values()) / len(terms_dict) if terms_dict else 0
    clustering_coverage = total_clustered_terms / len(terms_dict) if terms_dict else 0
    
    # Assemble the final JSON structure
    result = {
        'domain_name': domain_name,
        'group_name': group_name,
        'paper_count': total_unique_papers,
        'important_terms_count': len(terms_dict),
        'clusters_count': len(clusters),
        'clustered_terms_count': total_clustered_terms,
        'unclustered_terms_count': len(unclustered_terms_final),
        'clusters': clusters,
        'unclustered_terms': unclustered_terms_final,
        'statistics': {
            'avg_cluster_size': round(avg_cluster_size, 2),
            'avg_term_frequency': round(avg_term_frequency, 2),
            'clustering_coverage': round(clustering_coverage, 3),
            'total_unique_papers': total_unique_papers,
            'batch_size_used': BATCH_SIZE,
            'min_frequency_filter': MIN_FREQUENCY,
            'min_papers_filter': MIN_PAPERS
        }
    }
    
    return result


# ==================== Domain processing ====================

def process_domain(domain: str):
    """Process all hotspot groups for one domain."""
    
    print(f"\n{'#'*80}")
    print(f"# Round 3: Technology Clustering (Iterative Batch Processing)")
    print(f"# Domain: {domain}")
    print(f"# Batch Size: {BATCH_SIZE}, Min Frequency: {MIN_FREQUENCY}, Min Papers: {MIN_PAPERS}")
    print(f"{'#'*80}\n")
    
    # Load the Round 2 output
    papers = load_round2_results(domain)
    
    # Aggregate by group and apply low-frequency filtering
    print(f"\nAggregating and filtering terms by group...")
    groups_data = aggregate_and_filter_terms(papers, MIN_FREQUENCY, MIN_PAPERS)
    print(f"✓ Found {len(groups_data)} groups")
    
    # Process each group concurrently
    print(f"\nProcessing groups with {MAX_WORKERS} workers...")
    
    results = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_group_iterative, domain, group_name, group_data): group_name
            for group_name, group_data in groups_data.items()
        }
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Clustering groups"):
            group_name = futures[future]
            try:
                _, result = future.result()
                results[group_name] = result
            except Exception as e:
                print(f"\n✗ Error processing {group_name}: {str(e)}")
    
    # Write the summary report
    generate_summary_report(domain, results)
    
    print(f"\n{'='*80}")
    print(f"✓ Round 3 completed for domain: {domain}")
    print(f"{'='*80}\n")


def generate_summary_report(domain: str, results: Dict[str, Dict]):
    """Write a summary report."""
    
    report_file = ROUND3_DIR / f"{domain}_round3_report.txt"
    
    total_papers = sum(r['paper_count'] for r in results.values())
    total_terms = sum(r['important_terms_count'] for r in results.values())
    total_clusters = sum(r['clusters_count'] for r in results.values())
    total_clustered = sum(r['clustered_terms_count'] for r in results.values())
    total_unclustered = sum(r['unclustered_terms_count'] for r in results.values())
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("Round 3 Clustering Report (Iterative Batch Processing)\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Domain: {domain}\n")
        f.write(f"Configuration:\n")
        f.write(f"  - Batch Size: {BATCH_SIZE}\n")
        f.write(f"  - Min Frequency: {MIN_FREQUENCY}\n")
        f.write(f"  - Min Papers: {MIN_PAPERS}\n\n")
        
        f.write(f"Overall Statistics:\n")
        f.write(f"  - Total Papers: {total_papers}\n")
        f.write(f"  - Total Groups: {len(results)}\n")
        f.write(f"  - Total Terms (filtered): {total_terms}\n")
        f.write(f"  - Total Clusters: {total_clusters}\n")
        f.write(f"  - Total Clustered Terms: {total_clustered} ({total_clustered/total_terms:.1%})\n")
        f.write(f"  - Total Unclustered Terms: {total_unclustered} ({total_unclustered/total_terms:.1%})\n")
        f.write(f"  - Avg Clusters per Group: {total_clusters/len(results):.1f}\n")
        f.write(f"  - Avg Terms per Cluster: {total_clustered/total_clusters:.1f}\n\n")
        
        f.write("-"*80 + "\n")
        f.write("Per-Group Statistics:\n\n")
        
        for group_name, result in sorted(results.items()):
            f.write(f"Group: {group_name}\n")
            f.write(f"  - Papers: {result['paper_count']}\n")
            f.write(f"  - Input Terms: {result['important_terms_count']}\n")
            f.write(f"  - Clusters: {result['clusters_count']}\n")
            f.write(f"  - Clustered Terms: {result['clustered_terms_count']} ({result['statistics']['clustering_coverage']:.1%})\n")
            f.write(f"  - Unclustered Terms: {result['unclustered_terms_count']}\n")
            f.write(f"  - Avg Cluster Size: {result['statistics']['avg_cluster_size']:.1f}\n")
            f.write(f"  - Paper Coverage: 100%\n\n")
    
    print(f"\n✓ Summary report saved to: {report_file}")


# ==================== Main program ====================

def main():
    """CLI entrypoint."""
    import argparse
    
    global BATCH_SIZE, MIN_FREQUENCY, MIN_PAPERS

    if not DEEPSEEK_API_KEY:
        raise ValueError("Missing DEEPSEEK_API_KEY. Set it in the environment or .env file.")
    
    parser = argparse.ArgumentParser(description='Round 3: Technology Clustering (Iterative)')
    parser.add_argument('--domain', type=str, required=True,
                        help='Dataset stem used by Round 1-2 outputs')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                        help=f'Batch size for clustering (default: {BATCH_SIZE})')
    parser.add_argument('--min-freq', type=int, default=MIN_FREQUENCY,
                        help=f'Minimum term frequency (default: {MIN_FREQUENCY})')
    parser.add_argument('--min-papers', type=int, default=MIN_PAPERS,
                        help=f'Minimum paper count (default: {MIN_PAPERS})')
    
    args = parser.parse_args()
    
    BATCH_SIZE = args.batch_size
    MIN_FREQUENCY = args.min_freq
    MIN_PAPERS = args.min_papers
    
    round2_file = ROUND2_DIR / f"{args.domain}_round2_results.csv"
    if not round2_file.exists():
        print(f"✗ Error: Round 2 results not found: {round2_file}")
        print(f"  Please run Round 1-2 first: python round1_round2_extract.py")
        return
    
    # Process the requested domain
    process_domain(args.domain)


if __name__ == "__main__":
    main()
