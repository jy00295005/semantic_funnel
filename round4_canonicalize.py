#!/usr/bin/env python3
"""
Round 4: Cross-topic canonicalization
Canonicalize local Round 3 clusters into comparable cross-topic technologies
"""

import os
import json
import csv
import time
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY') or os.getenv('ds_api_key')
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com")
MODEL_NAME = os.getenv("ROUND4_MODEL", "deepseek-chat")
RESULTS_DIR = Path("results")
ROUND3_DIR = RESULTS_DIR / "round3"
ROUND4_DIR = RESULTS_DIR / "round4"

# Ensure output directory exists
ROUND4_DIR.mkdir(parents=True, exist_ok=True)

# Initialize client
client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=OPENAI_BASE_URL)


# ==================== Prompt definitions ====================

ROUND4_SYSTEM_PROMPT = """You are an expert in semantic standardization for scientific and technical terminology.

Your task is to canonicalize Round 3 local technology clusters into reusable cross-topic technology labels.

Requirements:
1. Use SPECIFIC, method-level or architecture-level names.
2. Merge only terms that refer to the same underlying technology family.
3. Do not merge across clearly different mechanisms, architectures, or methodological paradigms.
4. Prefer concise English names that can be reused across topics.

Examples of appropriate granularity:
✅ "Transformer Models" (specific architecture family)
✅ "Graph Neural Networks" (specific model family)
✅ "Federated Learning" (specific training paradigm)
✅ "Generative Adversarial Networks" (specific generative framework)

❌ "Machine Learning" (too broad)
❌ "Optimization" (too broad)
❌ "Prediction System" (too vague)
"""

ROUND4_USER_PROMPT = """
# Task: Canonicalize Local Clusters into Cross-Topic Technologies

## Context
You are analyzing {total_clusters} local technology clusters from {total_groups} hotspot groups within one scientific domain.

## Input Data

{clusters_by_group}

## Your Task

Analyze these local clusters and identify canonical technologies that recur across multiple hotspot groups.

**Goal**: produce a high-quality canonical list, not the largest possible list.

1. **Canonicalization requirement**:
   - Each canonical label must represent one identifiable technology family.
   - Use implementation-level names rather than broad field labels.
   - Good examples: "Transformer Models", "Graph Neural Networks", "Federated Learning", "Diffusion Models".

2. **Merging constraint**:
   - Merge local clusters only when they refer to the same technology or a clear naming variant.
   - Do not merge merely because two methods are often used together.
   - Do not merge parent and child concepts if the child is a materially distinct method family.

3. **Evidence requirement**:
   - Each canonical technology must map back to specific Round 3 cluster IDs.
   - Include exact matching terms that justify the mapping.
   - A technology should normally appear in at least two hotspot groups.

4. **Naming rules**:
   - Standardized names must be in English.
   - Prefer plural technology-family names when appropriate.
   - Remove application-specific suffixes unless they are intrinsic to the technology.
   - Keep names interpretable to readers outside one subfield.

5. **Exclusions**:
   - Broad umbrella concepts such as "Artificial Intelligence", "Machine Learning", or "Optimization".
   - Evaluation goals, performance claims, and application areas.
   - Generic tools that are not themselves focal technologies.

## Output Format

Return a JSON object with this structure:

```json
{{
  "canonical_technologies": [
    {{
      "technology_id": "CRT_001",
      "standardized_name": "Transformer Models",
      "category": "Model Architectures",
      "subcategory": "Attention-Based Neural Architectures",
      "appears_in_groups": ["Language Modeling", "Vision Foundation Models"],
      "related_clusters": [
        {{
          "cluster_id": "AI_LANG_C003",
          "cluster_name": "Transformer Architectures",
          "group_name": "Language Modeling",
          "matching_terms": ["transformer", "self-attention", "encoder-decoder transformer"]
        }}
      ],
      "naming_variations": ["transformer", "vision transformer", "transformer architecture"],
      "support_score": 8.5,
      "technology_description": "Attention-based neural architectures used across multiple hotspot groups.",
      "canonicalization_notes": "Merged local clusters that referred to the same transformer family with minor naming differences."
    }}
  ],
  "statistics": {{
    "total_technologies": 25,
    "avg_groups_per_technology": 2.3,
    "avg_clusters_per_technology": 2.8
  }}
}}
```

## Important
- Each technology must map to specific Round 3 cluster_ids
- Provide exact matching terms from the clusters
- Rate support_score (1-10) based on cross-topic recurrence and naming confidence

Please analyze the clusters and return canonical technologies only.
"""


# ==================== Data loading ====================

def load_all_clusters(domain: str):
    """Load all Round 3 cluster JSON files for one dataset."""
    json_files = list(ROUND3_DIR.glob(f"{domain}_*_clusters.json"))
    
    if not json_files:
        raise FileNotFoundError(f"No Round 3 results found in {ROUND3_DIR}")
    
    all_clusters = []
    groups_info = {}
    
    for json_file in sorted(json_files):
        print(f"Loading: {json_file.name}")
        
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        group_name = data['group_name']
        groups_info[group_name] = {
            'paper_count': data['paper_count'],
            'clusters_count': data['clusters_count']
        }
        
        for cluster in data['clusters']:
            # Keep only a small representative term set to control prompt length
            terms = [t['term'] for t in cluster['terms'][:10]]
            
            all_clusters.append({
                'cluster_id': cluster['cluster_id'],
                'cluster_name': cluster['cluster_name'],
                'cluster_description': cluster['cluster_description'],
                'group_name': group_name,
                'term_count': cluster['term_count'],
                'paper_count': cluster['paper_count'],
                'terms': terms
            })
    
    return all_clusters, groups_info


def format_clusters_for_llm(clusters, groups_info):
    """Format clusters into a compact text block for the LLM."""
    
    # Group clusters by hotspot label
    clusters_by_group = defaultdict(list)
    for cluster in clusters:
        clusters_by_group[cluster['group_name']].append(cluster)
    
    formatted = []
    for group_name, group_clusters in sorted(clusters_by_group.items()):
        group_info = groups_info[group_name]
        formatted.append(f"\n### Hotspot Group: {group_name}")
        formatted.append(f"Papers: {group_info['paper_count']}, Clusters: {group_info['clusters_count']}\n")
        
        for cluster in group_clusters:
            formatted.append(f"**{cluster['cluster_id']}**: {cluster['cluster_name']}")
            formatted.append(f"  Description: {cluster['cluster_description']}")
            formatted.append(f"  Key Terms ({cluster['term_count']} total): {', '.join(cluster['terms'])}")
            formatted.append(f"  Papers: {cluster['paper_count']}\n")
    
    return "\n".join(formatted)


# ==================== LLM call ====================

def extract_json_from_response(content: str) -> dict:
    """Extract JSON from the response, including fenced code blocks."""
    import re
    
    if not content or not content.strip():
        raise ValueError("Empty response content")
    
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    
    json_pattern = r'```(?:json)?\s*\n?([\s\S]*?)\n?```'
    matches = re.findall(json_pattern, content)
    
    for match in matches:
        try:
            return json.loads(match.strip())
        except json.JSONDecodeError:
            continue
    
    start = content.find('{')
    end = content.rfind('}')
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(content[start:end+1])
        except json.JSONDecodeError:
            pass
    
    raise ValueError(f"Failed to extract JSON from response: {content[:200]}...")


def call_llm_for_analysis(clusters_text, total_clusters, total_groups, max_retries=3):
    """Run Round 4 canonicalization with the LLM."""
    
    user_prompt = ROUND4_USER_PROMPT.format(
        total_clusters=total_clusters,
        total_groups=total_groups,
        clusters_by_group=clusters_text
    )
    
    print(f"\nCalling LLM for Round 4 canonicalization...")
    print(f"Model: {MODEL_NAME}")
    print(f"Input: {total_clusters} clusters from {total_groups} groups")
    print(f"Prompt length: {len(user_prompt):,} characters")
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": ROUND4_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=16000,
                temperature=0.3,
            )
            
            content = response.choices[0].message.content
            
            if hasattr(response.choices[0].message, 'reasoning_content') and response.choices[0].message.reasoning_content:
                reasoning = response.choices[0].message.reasoning_content
                print(f"  Reasoning trace length: {len(reasoning)} characters")
            
            result = extract_json_from_response(content)
            print(f"✓ LLM analysis completed")
            print(f"  Tokens used: {response.usage.total_tokens}")
            return result
            
        except Exception as e:
            print(f"⚠ Attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise
    
    raise Exception("Failed to get analysis result from LLM")


# ==================== Validation ====================

def validate_and_enrich_results(result, all_clusters):
    """Validate cluster references and enrich the Round 4 output."""
    
    print("\nValidating results...")
    
    cluster_map = {c['cluster_id']: c for c in all_clusters}
    
    technologies = result.get('canonical_technologies', [])
    validated_technologies = []
    
    for tech in technologies:
        valid_clusters = []
        for cluster_ref in tech.get('related_clusters', []):
            cluster_id = cluster_ref.get('cluster_id')
            if cluster_id in cluster_map:
                cluster_info = cluster_map[cluster_id]
                cluster_ref['term_count'] = cluster_info['term_count']
                cluster_ref['paper_count'] = cluster_info['paper_count']
                valid_clusters.append(cluster_ref)
            else:
                print(f"  Warning: cluster_id '{cluster_id}' not found")
        
        if valid_clusters:
            tech['related_clusters'] = valid_clusters
            tech['total_clusters_involved'] = len(valid_clusters)
            validated_technologies.append(tech)
        else:
            print(f"  Skipping technology '{tech.get('standardized_name')}' because no valid clusters were found")
    
    result['canonical_technologies'] = validated_technologies
    
    if validated_technologies:
        result['statistics']['total_technologies'] = len(validated_technologies)
        result['statistics']['avg_groups_per_technology'] = round(
            sum(len(set(t.get('appears_in_groups', []))) for t in validated_technologies) / len(validated_technologies), 2
        )
        result['statistics']['avg_clusters_per_technology'] = round(
            sum(t['total_clusters_involved'] for t in validated_technologies) / len(validated_technologies), 2
        )
    
    print(f"✓ Validated {len(validated_technologies)} technologies")
    return result


# ==================== Output writers ====================

def save_results(result, domain):
    """Save Round 4 outputs."""
    
    json_file = ROUND4_DIR / f"{domain}_canonical_technologies.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n✓ JSON saved to: {json_file}")
    
    report_file = ROUND4_DIR / f"{domain}_canonical_report.md"
    generate_markdown_report(result, report_file, domain)
    print(f"✓ Report saved to: {report_file}")
    
    csv_file = ROUND4_DIR / f"{domain}_technology_mapping.csv"
    generate_csv_mapping(result, csv_file)
    print(f"✓ Mapping saved to: {csv_file}")


def generate_markdown_report(result, report_file, domain):
    """Write a Markdown summary of Round 4 results."""
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(f"# Round 4 Canonical Technologies\n\n")
        f.write(f"**Domain**: {domain}\n\n")
        f.write(f"---\n\n")
        
        f.write("## Summary Statistics\n\n")
        stats = result.get('statistics', {})
        f.write(f"- **Total Technologies**: {stats.get('total_technologies', 0)}\n")
        f.write(f"- **Average Groups per Technology**: {stats.get('avg_groups_per_technology', 0):.1f}\n")
        f.write(f"- **Average Clusters per Technology**: {stats.get('avg_clusters_per_technology', 0):.1f}\n\n")
        
        if 'technology_categories' in stats:
            f.write("### Technology Categories\n\n")
            for category, count in sorted(stats['technology_categories'].items(), key=lambda x: -x[1]):
                f.write(f"- **{category}**: {count}\n")
            f.write("\n")
        
        f.write("---\n\n")
        
        f.write("## Detailed Technologies\n\n")
        
        technologies = result.get('canonical_technologies', [])
        technologies.sort(key=lambda x: x.get('support_score', 0), reverse=True)
        
        for i, tech in enumerate(technologies, 1):
            f.write(f"### {i}. {tech.get('standardized_name', 'Unknown')}\n\n")
            
            f.write(f"**Technology ID**: `{tech.get('technology_id', 'N/A')}`\n")
            f.write(f"**Category**: {tech.get('category', 'N/A')} / {tech.get('subcategory', 'N/A')}\n")
            f.write(f"**Support Score**: {tech.get('support_score', 0)}/10\n\n")
            
            groups = tech.get('appears_in_groups', [])
            f.write(f"#### Appears in {len(set(groups))} Hotspot Group(s)\n\n")
            for group in set(groups):
                f.write(f"- {group}\n")
            f.write("\n")
            
            clusters = tech.get('related_clusters', [])
            f.write(f"#### Related Clusters ({len(clusters)})\n\n")
            for cluster in clusters:
                f.write(f"- **{cluster.get('cluster_id')}**: {cluster.get('cluster_name')}\n")
                f.write(f"  - Hotspot Group: {cluster.get('group_name')}\n")
                f.write(f"  - Matching Terms: {', '.join(cluster.get('matching_terms', [])[:5])}\n")
            f.write("\n")
            
            variations = tech.get('naming_variations', [])
            if variations:
                f.write("#### Naming Variations\n\n")
                for var in variations:
                    f.write(f"- `{var}`\n")
                f.write("\n")
            
            f.write("#### Description\n\n")
            f.write(f"{tech.get('technology_description', 'N/A')}\n\n")
            
            f.write("#### Canonicalization Notes\n\n")
            f.write(f"{tech.get('canonicalization_notes', 'N/A')}\n\n")
            
            f.write("---\n\n")


def generate_csv_mapping(result, csv_file):
    """Write a flat technology-to-cluster mapping table."""
    
    with open(csv_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'technology_id',
            'standardized_name',
            'category',
            'cluster_id',
            'cluster_name',
            'group_name',
            'support_score'
        ])
        
        for tech in result.get('canonical_technologies', []):
            tech_id = tech.get('technology_id', '')
            tech_name = tech.get('standardized_name', '')
            category = tech.get('category', '')
            score = tech.get('support_score', 0)
            
            for cluster in tech.get('related_clusters', []):
                writer.writerow([
                    tech_id,
                    tech_name,
                    category,
                    cluster.get('cluster_id', ''),
                    cluster.get('cluster_name', ''),
                    cluster.get('group_name', ''),
                    score
                ])


# ==================== Main program ====================

def main():
    """CLI entrypoint."""
    import argparse

    if not DEEPSEEK_API_KEY:
        raise ValueError("Missing DEEPSEEK_API_KEY. Set it in the environment or .env file.")
    
    parser = argparse.ArgumentParser(description='Round 4: Cross-topic canonicalization')
    parser.add_argument('--domain', type=str, required=True,
                        help='Dataset stem used by Round 3 outputs')
    
    args = parser.parse_args()
    
    print("="*80)
    print("Round 4: Cross-topic canonicalization")
    print("="*80)
    print(f"Domain: {args.domain}")
    print(f"Model: {MODEL_NAME}")
    print("="*80)
    
    try:
        print("\n[Step 1] Loading Round 3 clusters...")
        all_clusters, groups_info = load_all_clusters(args.domain)
        print(f"✓ Loaded {len(all_clusters)} clusters from {len(groups_info)} groups")
        
        print("\n[Step 2] Formatting data for LLM...")
        clusters_text = format_clusters_for_llm(all_clusters, groups_info)
        print(f"✓ Formatted {len(clusters_text):,} characters")
        
        print("\n[Step 3] Analyzing with LLM...")
        result = call_llm_for_analysis(clusters_text, len(all_clusters), len(groups_info))
        
        print("\n[Step 4] Validating results...")
        result = validate_and_enrich_results(result, all_clusters)
        
        print("\n[Step 5] Saving results...")
        save_results(result, args.domain)
        
        print("\n" + "="*80)
        print("✓ Round 4 Analysis Complete!")
        print("="*80)
        stats = result.get('statistics', {})
        print(f"\nFound {stats.get('total_technologies', 0)} canonical technologies")
        print(f"Average {stats.get('avg_groups_per_technology', 0):.1f} groups per technology")
        print(f"Average {stats.get('avg_clusters_per_technology', 0):.1f} clusters per technology")
        print("\n" + "="*80)
        
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
