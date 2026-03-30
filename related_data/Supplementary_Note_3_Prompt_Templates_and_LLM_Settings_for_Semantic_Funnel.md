# Supplementary Note 3. Prompt templates and LLM settings for the Semantic Funnel

## 3.1 Rationale and reproducibility scope
The Semantic Funnel was designed to reduce lexical noise while preserving method-level semantic specificity. In this study, the funnel transforms paper-level technical expressions into cross-hotspot canonical Core Research Technologies (CRTs) through four rounds: broad extraction (R1), strict denoising (R2), hotspot-local semantic clustering (R3), and cross-hotspot canonicalization (R4). The objective of this note is to provide an auditable, implementation-level description of that process, including complete prompts, runtime parameters, JSON constraints, and post-processing rules. The intent is to make the pipeline reproducible for reviewers who need to verify that the funnel is not a black-box transformation.

All walkthrough examples referenced in this note are documented in `Supplementary_Table_S2_Example_Walkthroughs_of_Four_Round_Funnel.md`.

## 3.2 Execution provenance and implementation alignment
The manuscript pipeline uses prompt-driven LLM calls implemented in the project codebase. For reproducibility, this note records the effective prompt texts used in production scripts and aligns them to the conceptual four-round funnel as follows.

R1 corresponds to paper-level term extraction from title and abstract. R2 corresponds to strict filtering of generic and non-technical terms. R3 corresponds to hotspot-level semantic grouping with technical cluster naming. R4 corresponds to global mapping of local clusters into canonical cross-hotspot technology names.

A key implementation detail is that R4 prompt text in the script is authored in Chinese. To support international peer review, this note provides a full English-equivalent translation of that prompt while preserving the original operational constraints.

## 3.3 Runtime settings and response constraints
Table 3-1 reports the effective runtime settings used in each round.

| Round | Script-level model | temperature | top_p | max_tokens | response format | Retry behavior |
|---|---|---:|---:|---:|---|---|
| R1 | `deepseek-chat` | 0.2 | default | default | `json_object` | up to 3 retries with exponential backoff |
| R2 | `deepseek-chat` | 0.2 | default | default | `json_object` | up to 3 retries with exponential backoff |
| R3-A (filter) | `deepseek-chat` | default | default | default | `json_object` | batch-level continue-on-error |
| R3-B (cluster) | `deepseek-chat` | default | default | default | `json_object` | return empty cluster set on failure |
| R4 | `gpt-4o-mini` | 0.3 | default | default | `json_object` | up to 3 retries with JSON parse validation |

No explicit `top_p` or `max_tokens` values are set in these scripts; therefore provider defaults apply. JSON-mode responses are enforced to reduce format drift.

## 3.4 Round 1 complete prompt (verbatim)
R1 is configured for high recall. The prompt intentionally over-includes candidate method phrases and defers precision control to subsequent rounds.

### 3.4.1 System prompt
```text
You are an expert in identifying technical approaches, methodologies, and technologies in scientific research papers. You have deep knowledge across multiple scientific domains including but not limited to:
- Computer Science and Artificial Intelligence
- Materials Science and Engineering
- Physics and Quantum Technologies
- Chemistry and Chemical Engineering
- Energy and Battery Technologies
- Biomedical and Life Sciences

You are skilled at extracting domain-specific technical terms that represent concrete technologies, methods, and approaches.
```

### 3.4.2 User prompt
```text
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
```

R1 output contract:
```json
{"technical_terms": ["term1", "term2", "..."]}
```

## 3.5 Round 2 complete prompt (verbatim)
R2 is configured for strict precision and intentionally favors false negatives over false positives, because non-specific terms strongly inflate downstream cluster noise.

### 3.5.1 System prompt
```text
You are an expert in evaluating the importance and significance of technologies in scientific research. You can assess technical terms based on their:
- Technical specificity and precision
- Domain relevance and applicability
- Innovation significance and impact potential
- Usefulness for technology analysis and knowledge discovery

You understand the difference between generic concepts and specific, meaningful technologies across multiple scientific domains.
```

### 3.5.2 User prompt
```text
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

## STRICT Filtering Criteria

### MUST RETAIN - Breakthrough and Specific Technologies:

**1. Specific Technical Innovations (HIGHEST PRIORITY):**
   - **Named technologies with technical details**: "surface code", "variational quantum eigensolver (VQE)", "BB84 protocol"
   - **Specific materials with composition**: "Li7La3Zr2O12 (LLZO)", "Ni-rich Li[NixCoyMn1-x-y]O2", "MXene Ti3C2Tx"
   - **Concrete methods with parameters**: "localized high-concentration electrolyte", "dendrite-free Zn plating", "operando X-ray absorption spectroscopy"
   - **Novel processes with mechanisms**: "water-lubricated intercalation", "epitaxial electrodeposition", "reversible anionic redox"

**2. Innovative Material Systems:**
   - **Specific compounds**: "polyaniline-intercalated manganese dioxide", "Co-N/G composite", "V2O5·nH2O"
   - **Nanostructures with morphology**: "hierarchical porous carbon", "graphene scroll-coating", "monodisperse cobalt atoms"
   - **Engineered interfaces**: "solid electrolyte interphase (SEI)", "polymer-modified Zn anode", "conformal coating"

**3. Breakthrough Methods and Techniques:**
   - **Specific synthesis methods**: "microwave synthesis", "chemical vapor deposition (CVD)", "electrochemical synthesis"
   - **Novel characterization**: "operando neutron depth profiling", "cryogenic TEM", "in situ AFM"
   - **Advanced processing**: "hot-pressing fabrication", "warm isostatic pressing", "vapor-solid strategy"

### MUST REMOVE - Generic and Non-Technical Terms:

**1. Generic Concepts (REMOVE ALL):**
   - Broad categories: "energy storage", "battery technology", "optimization", "analysis"
   - Vague descriptors: "high performance", "advanced", "novel", "improved", "efficient"
   - General terms: "electrode materials", "electrolyte", "cathode", "anode" (without specifics)
   - Common words: "system", "method", "approach", "technique", "strategy" (alone)

**2. Performance Metrics (REMOVE ALL):**
   - "high capacity", "energy density", "power density", "cycling stability"
   - "coulombic efficiency", "capacity retention", "rate capability"
   - "long cycle life", "fast charging", "high reversibility"

**3. Research Activities (REMOVE ALL):**
   - "experimental study", "theoretical analysis", "performance evaluation"
   - "material characterization", "electrochemical testing", "structural analysis"
   - "design optimization", "process improvement", "performance enhancement"

**4. Application Domains (REMOVE ALL):**
   - "lithium-ion batteries", "solid-state batteries", "supercapacitors" (as standalone)
   - "electric vehicles", "energy storage systems", "renewable energy"
   - "medical diagnosis", "drug discovery", "materials science"

**5. Basic Equipment/Instruments (REMOVE unless specific):**
   - "X-ray diffraction", "scanning electron microscopy", "transmission electron microscopy" (too common)
   - Keep only when novel variants are explicit.

Strict evaluation process:

Step 1: Specificity Test (MUST PASS)
- Does it contain specific technical details?
- Can it be distinguished from similar technologies?

Step 2: Innovation Test (MUST PASS)
- Does it represent a novel or breakthrough approach?
- Is it a specific implementation rather than a broad category?

Step 3: Context Relevance (MUST PASS)
- Is it specifically relevant to "{research_topic}" and method-level work in "{research_area}"?

Target retention rate:
- Aim for 15-30% retention.

Input terms:
{terms_list}

Output JSON format:
{
  "important_terms": [...],
  "removed_terms": [...],
  "filtering_summary": {
    "total_input": <number>,
    "retained": <number>,
    "removed": <number>,
    "retention_rate": "<percentage>"
  }
}
```

R2 output contract:
```json
{
  "important_terms": ["..."],
  "removed_terms": ["..."],
  "filtering_summary": {
    "total_input": 0,
    "retained": 0,
    "removed": 0,
    "retention_rate": "0%"
  }
}
```

## 3.6 Round 3 complete prompts (verbatim)
In this implementation, R3 is operationally split into two LLM calls: R3-A strict hotspot-level filtering (batch-wise) and R3-B semantic clustering.

### 3.6.1 R3-A filtering system prompt
```text
You are an expert in identifying breakthrough technologies and technical innovations.
```

### 3.6.2 R3-A filtering user prompt
```text
As an AI expert specializing in breakthrough technology identification, analyze these technical terms from the topic '{topic_name}' and select ONLY the most innovative and specific technical implementations.

Terms to analyze: {batch_terms}

STRICT FILTERING REQUIREMENTS:
1. Keep ONLY terms that meet ALL of these criteria:
   - Represents a SPECIFIC technical implementation (not just a concept)
   - Has UNIQUE methodological characteristics
   - Introduces NOVEL approaches or combinations
   - Can be clearly distinguished from existing standard methods

2. AGGRESSIVELY REMOVE terms that have ANY of these characteristics:
   - Generic or standard terminology (e.g., "deep learning", "machine learning", "neural network")
   - Broad methodological terms (e.g., "optimization", "analysis", "prediction")
   - Basic concepts or common techniques (e.g., "clustering", "regression", "classification")
   - Descriptive or qualifier terms (e.g., "advanced", "improved", "novel")
   - General processes (e.g., "modeling", "processing", "simulation")
   - Common combinations (e.g., "data-driven approach", "hybrid method")
   - Standard architectures (e.g., "CNN", "RNN", "LSTM" without specific modifications)
   - Basic tools or frameworks (e.g., "algorithm", "framework", "system")

3. Examples of Filtering:
   KEEP:
   - "Attention-guided Multi-scale ResNet with Uncertainty Estimation"
   - "Hybrid PSO-LSTM with Adaptive Parameter Tuning"
   - "Cross-modal Transformer-GAN for Seismic Data Enhancement"

   REMOVE:
   - "deep learning approach"
   - "machine learning model"
   - "neural network"
   - "optimization algorithm"
   - "data analysis"
   - "hybrid method"
   - "improved technique"

4. Additional Rules:
   - If a term is a combination of generic methods without specific innovation, REMOVE it
   - If a term could be found in basic textbooks or common practice, REMOVE it
   - If a term doesn't clearly indicate its technical uniqueness, REMOVE it
   - Keep only terms that represent the cutting edge of technology

Return a JSON object with:
{
    "important_terms": ["term1", "term2", ...],
    "reasoning": "Brief explanation of selection criteria"
}

Note: Err on the side of being TOO strict. It's better to keep fewer, truly innovative terms than to include borderline cases.
```

### 3.6.3 R3-B clustering system prompt
```text
You are an expert in technical terminology organization with deep knowledge of specific algorithms, methodologies, and their applications.
```

### 3.6.4 R3-B clustering user prompt
```text
Analyze and cluster these important technical terms from the topic '{topic_name}'.

Terms to cluster: {terms_list}

Requirements for clustering and naming:
1. Group truly related technical concepts that share similar technical foundations or applications
2. Each cluster should represent a specific technical area or methodology
3. Cluster names MUST be highly specific and technical, following these rules:
   - Include the specific type/category of algorithm or method (e.g., "Gradient-Based Optimization Algorithms" instead of just "Machine Learning")
   - Mention the primary technical approach or methodology (e.g., "CNN-Based Image Segmentation Networks" instead of just "Deep Learning")
   - If applicable, include the application domain (e.g., "Transformer-Based Seismic Signal Processing")
   - Never use generic terms like "AI", "Machine Learning", or "Advanced Techniques" alone
4. Provide a technical description that explains the unique characteristics and applications of the cluster

Return a JSON object with:
{
    "clusters": [
        {
            "name": "specific_technical_cluster_name",
            "terms": ["term1", "term2"],
            "description": "Technical description focusing on methodology and applications"
        }
    ]
}

Example cluster names:
- "Nature-Inspired Metaheuristic Optimization Algorithms"
- "Attention-Enhanced CNN Architectures for Geospatial Analysis"
- "Hybrid GAN-Based Data Augmentation Networks"
```

R3 output contract:
```json
{
  "clusters": [
    {
      "name": "specific_technical_cluster_name",
      "terms": [
        {"term": "...", "paper_ids": [123, 456]}
      ],
      "description": "technical description"
    }
  ]
}
```

## 3.7 Round 4 complete prompt (Chinese source translated to English)
R4 in the implementation script uses Chinese prompt text. For international reproducibility and peer review, the full semantic-equivalent English translation is provided below.

### 3.7.1 R4 system prompt (English translation of script prompt)
```text
You are an expert specialized in analyzing and synthesizing commonalities across technologies. Your task is to identify concrete technical implementation methods rather than broad technology categories.
```

### 3.7.2 R4 user prompt (full English-equivalent translation)
```text
Analyze the following topic clusters, extract specific common technologies, and build mapping relationships. Note that common technology names should remain domain-neutral and should not include specific application domains.

Current batch topics and clusters:
{self._format_batch_data(batch_df)}

Existing common-technology list:
{self._format_existing_techs(existing_techs)}

Please perform the following tasks:
1. Analyze clusters in this batch and identify specific common technologies, with requirements:
   - Avoid overly broad technology categories (for example, "machine learning", "deep learning").
   - Extract core technical methods, architectures, and algorithms, removing specific application context.
   - Technology names should be cross-domain portable (for example, "Attention Mechanism" rather than "Attention-based Financial Prediction").
   - Prioritize core implementation components and remove scenario-specific wording.

2. Decision criteria:
   - Too broad (do not use):
     * "Machine Learning Models"
     * "Reinforcement Learning"
     * "Optimization Methods"
     * "Neural Network Models"

   - Too specific (do not use):
     * "Graph-Convolutional LSTM for Time-Series Prediction"
     * "CNN-LSTM with Bidirectional Attention"
     * "Space Vector Modulation for Motor Control"
     * "NSGA-II for Energy Optimization"
     * "Graph-Convolutional LSTM and Transformer Models with Multi-head Attention"
     * "CNN-LSTM with Bidirectional Attention and Layer Normalization"
     * "Space Vector Modulation with Dynamic Hysteresis Current Control and Phase Compensation"
     * "Multi-objective NSGA-II with Adaptive Population Size for Energy Optimization"
     * "Graph-Convolutional LSTM and Transformer Models"
     * "CNN-RNN Hybrid Models for Financial Time-Series Analysis"
     * "Multi-Agent Reinforcement Learning for Industrial Automation"

   - Balanced granularity (recommended):
     * "Graph Neural Networks"
     * "Attention Mechanisms"
     * "Hybrid CNN-RNN Architecture"
     * "Space Vector Modulation"
     * "Multi-objective Optimization"
     * "Ensemble Decomposition"

3. Map each cluster to the most appropriate specific technology:
   - If a cluster contains application-domain description (for example, "for X"), keep only the technical part.
   - If a technology appears in different domains, keep one unified technology name.
   - For hybrid technologies, preserve core elements (for example, "CNN-LSTM Hybrid" rather than scenario-specific composite names).

Return JSON format:
{
    "mappings": [
        {
            "topic_id": "topic ID",
            "cluster_name": "original cluster name",
            "common_tech": "domain-neutral technology name",
            "relationship": "mapping rationale",
            "confidence": "mapping confidence (0-1)"
        },
        ...
    ],
    "new_technologies": [
        {
            "name": "domain-neutral technology name",
            "description": "technology description (domain-neutral)",
            "key_concepts": ["core concept 1", "core concept 2", ...],
            "significance": "cross-domain applicability statement"
        },
        ...
    ]
}
```

R4 output contract:
```json
{
  "mappings": [
    {
      "topic_id": "...",
      "cluster_name": "...",
      "common_tech": "...",
      "relationship": "...",
      "confidence": 0.0
    }
  ],
  "new_technologies": [
    {
      "name": "...",
      "description": "...",
      "key_concepts": ["..."],
      "significance": "..."
    }
  ]
}
```

## 3.8 Deterministic post-processing and quality controls
After LLM responses are received, the pipeline applies deterministic checks to reduce format and semantic instability. JSON parsing is strict, malformed responses are retried, and non-conforming outputs are excluded rather than silently repaired. In R3, term-to-paper provenance is preserved as `paper_ids` at the term level, enabling traceability from local clusters back to source papers. In R4, mapping confidence is interpreted conservatively and only high-confidence mappings are used in the final aggregation table, thereby reducing semantic drift introduced by low-certainty assignments.

This post-processing layer is essential for methodological validity. Without it, small response-format differences can propagate into large structural differences in final CRT counts and coverage statistics.

## 3.9 Known risks and mitigation logic
The funnel uses strict filtering in R2 and R3-A, which can remove borderline but potentially meaningful terms. This is an intentional design choice to prioritize method-level precision over lexical recall in downstream canonicalization. To mitigate over-pruning risk, manuscript-level interpretation is based on aggregate structural signals (coverage, domain diversity, co-occurrence, maturity) rather than any single term decision.

A second risk concerns naming granularity in R4. The prompt explicitly rejects both overly broad and overly specific labels, and the script enforces cross-domain portability naming logic to maintain stable canonical categories.

## 3.10 Linkage to empirical walkthroughs
The practical behavior of this funnel is illustrated with three full examples in `Supplementary_Table_S2_Example_Walkthroughs_of_Four_Round_Funnel.md`, where each case reports R1 raw terms, R2 retained terms, R3 local clusters, R4 canonical CRT mappings, and provenance evidence.

These examples are included to support reviewer audit and to demonstrate that canonicalization decisions are traceable rather than opaque.
