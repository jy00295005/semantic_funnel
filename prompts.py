"""
Prompt templates for Round 1 and Round 2 of the Semantic Funnel.

These prompts are written for OpenAI-compatible chat completion APIs and can
be adapted to different scientific domains by changing the runtime metadata.
"""

# ==================== Round 1: Initial Technical Term Extraction ====================

ROUND1_SYSTEM_PROMPT = """You are an expert in identifying technical approaches, methodologies, and technologies in scientific research papers. You have deep knowledge across multiple scientific domains including but not limited to:
- Computer Science and Artificial Intelligence
- Materials Science and Engineering
- Physics and Quantum Technologies
- Chemistry and Chemical Engineering
- Energy and Battery Technologies
- Biomedical and Life Sciences

You are skilled at extracting domain-specific technical terms that represent concrete technologies, methods, and approaches."""


ROUND1_USER_PROMPT = """
Please analyze the following scientific paper and extract ALL meaningful technical terms, methods, and technologies.

**Paper Context:**
- Research Area: {research_area}
- Research Topic: {research_topic}
- Broader Domain: {domain_name}

**Extraction Purpose:**
The extracted terms will be used to:
1. Discover specific technologies and methodologies in this research domain
2. Identify important and breakthrough technologies
3. Analyze common technologies across different research topics
4. Build a technology knowledge graph for this scientific field

**What to Extract:**

1. **Core Technologies and Methods:**
   - Specific algorithms, techniques, and methodologies
   - Novel approaches and implementations
   - Technical frameworks and architectures
   - Material compositions and structures (for materials science)
   - Physical phenomena and mechanisms (for physics)
   - Chemical processes and reactions (for chemistry)
   - Device architectures and designs (for engineering)

2. **Technical Implementations:**
   - Specific experimental methods and setups
   - Fabrication and synthesis techniques
   - Characterization and measurement methods
   - Computational and simulation approaches
   - Processing and treatment methods

3. **Technology Combinations:**
   - Hybrid methods and integrated approaches
   - Multi-technique combinations
   - Cross-domain technology applications

**Term Requirements:**

1. **Specificity Levels:**
   - INCLUDE specific technical terms (e.g., "density functional theory", "lithium-ion intercalation", "quantum entanglement", "convolutional neural networks")
   - INCLUDE method variants and implementations (e.g., "pulsed laser deposition", "molecular dynamics simulation", "transfer learning")
   - INCLUDE domain-specific technologies (e.g., "solid-state electrolyte", "topological insulator", "CRISPR-Cas9")

2. **Length Guidelines:**
   - Terms can be 2-6 words when they represent established technical concepts
   - Prefer complete technical phrases over single generic words
   - Include modifiers that add technical specificity (e.g., "high-temperature superconductor" not just "superconductor")

3. **Technical Meaningfulness:**
   - Terms should represent actual technologies, methods, or technical concepts
   - Terms should be specific enough to be useful for technology analysis
   - Terms should have clear technical definitions in the scientific literature

**DO NOT Extract:**

1. **Generic Concepts:**
   - Overly broad terms (e.g., "technology", "method", "approach", "system" alone)
   - General research activities (e.g., "investigation", "study", "analysis" alone)
   - Common scientific vocabulary without technical specificity

2. **Non-Technical Content:**
   - Application domains without technical detail (e.g., "medical diagnosis", "energy storage" alone)
   - Performance metrics and evaluation measures (e.g., "accuracy", "efficiency", "performance")
   - Dataset names and benchmark information
   - General material names without technical context (e.g., "metal", "polymer" alone)

3. **Redundant Terms:**
   - Near-duplicates with only minor wording differences
   - Terms that are subsets of more complete technical phrases

4. **Pure Domain Knowledge:**
   - Basic scientific concepts taught in textbooks (e.g., "Newton's law", "periodic table")
   - Standard material names without technical modification (e.g., "water", "silicon" alone)
   - Common phenomena without technical context (e.g., "oxidation", "diffusion" alone)

**Important Notes:**
- Focus on TECHNICAL TERMS that represent specific methods, technologies, and approaches
- The terms should be useful for discovering important and breakthrough technologies
- Consider the specific research context: {research_area} - {research_topic}
- Include domain-specific technical vocabulary appropriate for this research area
- Maintain technical precision while being comprehensive
- When in doubt, include the term if it has clear technical meaning in the research context

**Output Format:**
Return a JSON object with a single key "technical_terms" containing an array of extracted terms.

**Paper Content:**

Title: {title}

Abstract: {abstract}

Please extract ALL relevant technical terms following the guidelines above.
"""


# ==================== Round 2: Important Technology Filtering ====================

ROUND2_SYSTEM_PROMPT = """You are an expert in evaluating the importance and significance of technologies in scientific research. You can assess technical terms based on their:
- Technical specificity and precision
- Domain relevance and applicability
- Innovation significance and impact potential
- Usefulness for technology analysis and knowledge discovery

You understand the difference between generic concepts and specific, meaningful technologies across multiple scientific domains."""


ROUND2_USER_PROMPT = """
You are given a list of technical terms extracted from a scientific paper.

**Paper Context:**
- Research Area: {research_area}
- Research Topic: {research_topic}
- Broader Domain: {domain_name} 

Your task is to **STRICTLY filter and retain ONLY the most SPECIFIC, INNOVATIVE, and BREAKTHROUGH technical terms** that represent:
1. **Concrete technological innovations** with specific technical details
2. **Novel materials, methods, or processes** with clear technical characteristics
3. **Breakthrough approaches** that advance the field
4. **Specific technical implementations** that can be precisely identified

## 🎯 STRICT Filtering Criteria

### ✅ **MUST RETAIN - Breakthrough & Specific Technologies:**

**1. Specific Technical Innovations (HIGHEST PRIORITY):**
   - **Named technologies with technical details**: "surface code", "variational quantum eigensolver (VQE)", "BB84 protocol"
   - **Specific materials with composition**: "Li7La3Zr2O12 (LLZO)", "Ni-rich Li[NixCoyMn1-x-y]O2", "MXene Ti3C2Tx"
   - **Concrete methods with parameters**: "localized high-concentration electrolyte", "dendrite-free Zn plating", "operando X-ray absorption spectroscopy"
   - **Novel processes with mechanisms**: "water-lubricated intercalation", "epitaxial electrodeposition", "reversible anionic redox"
   
   ✓ Good examples: "topological code", "solid-state electrolyte Li10GeP2S12", "attention mechanism with multi-head architecture"
   ✗ Bad examples: "optimization method", "new approach", "advanced technique"

**2. Innovative Material Systems:**
   - **Specific compounds**: "polyaniline-intercalated manganese dioxide", "Co-N/G composite", "V2O5·nH2O"
   - **Nanostructures with morphology**: "hierarchical porous carbon", "graphene scroll-coating", "monodisperse cobalt atoms"
   - **Engineered interfaces**: "solid electrolyte interphase (SEI)", "polymer-modified Zn anode", "conformal coating"
   
   ✓ Must include: composition, structure, or modification details
   ✗ Reject: "electrode materials", "cathode materials", "nanomaterials" (too generic)

**3. Breakthrough Methods & Techniques:**
   - **Specific synthesis methods**: "microwave synthesis", "chemical vapor deposition (CVD)", "electrochemical synthesis"
   - **Novel characterization**: "operando neutron depth profiling", "cryogenic TEM", "in situ AFM"
   - **Advanced processing**: "hot-pressing fabrication", "warm isostatic pressing", "vapor-solid strategy"
   
   ✓ Must specify: the exact technique name and key characteristics
   ✗ Reject: "synthesis techniques", "characterization methods", "processing routes"

### ❌ **MUST REMOVE - Generic & Non-Technical Terms:**

**1. Generic Concepts (REMOVE ALL):**
   - ❌ Broad categories: "energy storage", "battery technology", "optimization", "analysis"
   - ❌ Vague descriptors: "high performance", "advanced", "novel", "improved", "efficient"
   - ❌ General terms: "electrode materials", "electrolyte", "cathode", "anode" (without specifics)
   - ❌ Common words: "system", "method", "approach", "technique", "strategy" (alone)

**2. Performance Metrics (REMOVE ALL):**
   - ❌ "high capacity", "energy density", "power density", "cycling stability"
   - ❌ "coulombic efficiency", "capacity retention", "rate capability"
   - ❌ "long cycle life", "fast charging", "high reversibility"

**3. Research Activities (REMOVE ALL):**
   - ❌ "experimental study", "theoretical analysis", "performance evaluation"
   - ❌ "material characterization", "electrochemical testing", "structural analysis"
   - ❌ "design optimization", "process improvement", "performance enhancement"

**4. Application Domains (REMOVE ALL):**
   - ❌ "lithium-ion batteries", "solid-state batteries", "supercapacitors" (as standalone)
   - ❌ "electric vehicles", "energy storage systems", "renewable energy"
   - ❌ "medical diagnosis", "drug discovery", "materials science"

**5. Basic Equipment/Instruments (REMOVE unless specific):**
   - ❌ "X-ray diffraction", "scanning electron microscopy", "transmission electron microscopy" (too common)
   - ✓ KEEP if novel: "operando X-ray diffraction", "cryogenic TEM", "in situ neutron depth profiling"

## 🔍 Strict Evaluation Process

**For EACH term, apply this 3-step filter:**

### Step 1: Specificity Test (MUST PASS)
- ❓ Does it contain specific technical details (composition, structure, parameters)?
- ❓ Can it be distinguished from similar technologies?
- ❓ Would removing one word make it meaningless?

**If NO → REMOVE immediately**

### Step 2: Innovation Test (MUST PASS)
- ❓ Does it represent a novel or breakthrough approach?
- ❓ Is it a specific implementation (not just a category)?
- ❓ Would researchers cite this as a distinct technology?

**If NO → REMOVE immediately**

### Step 3: Context Relevance (MUST PASS)
- ❓ Is it specifically relevant to "{research_topic}"?
- ❓ Does it represent actual technical work in "{research_area}"?
- ❓ Is it a concrete technology (not an abstract concept)?

**If NO → REMOVE immediately**

## 📏 Target Retention Rate

**Aim for 15-30% retention rate** (be STRICT!)
- If retaining >40% → You are too lenient, be more strict
- If retaining <10% → You might be too strict, but err on the side of caution

## 🎯 Examples for This Research Context

**Research Area:** {research_area}
**Research Topic:** {research_topic}

**RETAIN examples (specific, innovative, method-level):**
- ✅ "dendrite-free Zn plating with nano-CaCO3 coating"
- ✅ "localized high-concentration electrolyte with fluorinated ether"
- ✅ "reversible epitaxial electrodeposition on graphene"
- ✅ "operando X-ray absorption spectroscopy"
- ✅ "Li7La3Zr2O12 (LLZO) solid electrolyte"

**REMOVE examples (generic, evaluative, or non-technical):**
- ❌ "high energy density" (performance metric)
- ❌ "electrode materials" (too generic)
- ❌ "battery performance" (evaluation term)
- ❌ "advanced synthesis" (vague descriptor)
- ❌ "electrochemical energy storage" (application domain)

**Special Considerations for this Research Context:**
- Research Area: {research_area}
- Research Topic: {research_topic}
- Consider domain-specific technical vocabulary for this specific research area
- Retain terms that are technically meaningful in the context of {research_topic}
- Focus on terms that represent actual technical implementations
- Prioritize terms that indicate methodological or technological choices

**Input Terms:**

{terms_list}

**Output Format:**
Return a JSON object with:
{{
    "important_terms": [list of retained important technical terms],
    "removed_terms": [list of removed generic/non-technical terms],
    "filtering_summary": {{
        "total_input": <number>,
        "retained": <number>,
        "removed": <number>,
        "retention_rate": "<percentage>"
    }}
}}

Please carefully evaluate each term and return only the IMPORTANT and TECHNICALLY SPECIFIC terms.
"""


# ==================== Round 3: Technology Clustering ====================

ROUND3_SYSTEM_PROMPT = """You are an expert in scientific research and technology taxonomy. Your task is to cluster technical terms from academic papers into semantically coherent groups.

Your responsibilities:
1. Identify technical terms that represent similar or related technologies
2. Group them into meaningful clusters based on their semantic relationships
3. Provide clear, descriptive names and descriptions for each cluster
4. Ensure each term is assigned to exactly ONE cluster (no duplicates, no omissions)

Guidelines:
- Focus on TECHNICAL SIMILARITY, not just keyword matching
- Consider the context of the research domain
- Create clusters that are neither too broad nor too specific
- Aim for 5-15 clusters per group, with 3-20 terms per cluster
- If a term doesn't fit well into any cluster, mark it as "unclustered"
"""


ROUND3_USER_PROMPT = """
# Task: Cluster Technical Terms

## Research Context
- **Domain**: {domain_name}
- **Research Group**: {group_name}
- **Group Description**: {group_description}

## Input Data
You are given {term_count} important technical terms extracted from {paper_count} academic papers in the field of "{group_name}".

### Technical Terms List:
{terms_list}

## Your Task
Please cluster these technical terms into semantically coherent groups. For each cluster:

1. **Cluster Name**: A concise, descriptive name in English (3-8 words)
2. **Cluster Description**: A brief explanation of what technologies this cluster represents (1-2 sentences, in English)
3. **Terms**: List all terms belonging to this cluster (use the exact term strings from the input)

## Output Format
Return your results as a JSON object with the following structure:

```json
{{
  "clusters": [
    {{
      "cluster_name": "Quantum Error Correction Codes",
      "cluster_description": "Technologies related to quantum error correction, including surface codes, topological codes, and stabilizer formalism for protecting quantum information from errors.",
      "terms": [
        "surface code",
        "topological code",
        "stabilizer code"
      ]
    }},
    {{
      "cluster_name": "Variational Quantum Algorithms",
      "cluster_description": "Hybrid quantum-classical algorithms using parameterized quantum circuits for optimization and simulation tasks.",
      "terms": [
        "variational quantum eigensolver",
        "QAOA",
        "VQE"
      ]
    }}
  ],
  "unclustered_terms": [
    {{
      "term": "some_ambiguous_term",
      "reason": "Does not clearly fit into any semantic cluster"
    }}
  ]
}}
```

## Important Notes
1. **Completeness**: Every term in the input list must appear EXACTLY ONCE in either "clusters" or "unclustered_terms"
2. **No Duplicates**: Each term should only appear in one cluster
3. **Semantic Coherence**: Terms in the same cluster should represent related technologies or methods
4. **Appropriate Granularity**: Avoid creating too many tiny clusters or too few giant clusters
5. **Use Exact Terms**: Use the exact term strings from the input list (case-sensitive)

Please proceed with the clustering task.
"""
