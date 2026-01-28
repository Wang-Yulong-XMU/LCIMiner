# LCIMiner

**LCIMiner** is an automated workflow based on large language models (LLMs) with multimodal capabilities for the retrieval, extraction, and harmonization of life cycle inventory (LCI) data from heterogeneous scientific literature.

This repository accompanies the manuscript:


> **An LLM-Based Workflow with Multimodal Capabilities for Retrieval, Extraction, and Harmonization of Life Cycle Inventory Data from Heterogeneous Literature: Application to Aquaculture**  
> *(under review)*

---

## 🧩 Workflow Structure

The LCIMiner workflow consists of the following core stages:

1. **Literature Retrieval**  
   Identification and collection of relevant LCA-related scientific publications.

2. **Multimodal Content Parsing**  
   Extraction of textual, tabular, and figure-based information from heterogeneous document formats (e.g., PDFs).

3. **LLM-Based Information Extraction**  
   Semantic extraction of candidate LCI items, including inputs, outputs, and contextual metadata.

4. **Data Harmonization and Structuring**  
   Normalization of units, terminologies, and system boundaries into structured LCI representations.

5. **Validation and Error Auditing**  
   Automated and semi-automated checks for numerical, classification, and structural inconsistencies.

---

## 📂 Repository Status

🚧 **Code Availability Notice**

To ensure a clean, well-documented, and reproducible release, the **complete source code, trained prompts, and configuration files will be made publicly available upon acceptance of the associated manuscript**.

At the current stage, this repository serves to:

- Document the overall system design and workflow logic  
- Provide supplementary materials referenced in the manuscript  
- Ensure transparency regarding the implementation roadmap  

Partial examples, schemas, or demonstration materials may be added during the review process.

---

## 📊 Data and Experiments

All experimental evaluations reported in the manuscript are conducted on curated life cycle inventory datasets derived from peer-reviewed aquaculture-related LCA studies.

Due to copyright restrictions, original full-text articles are **not redistributed** in this repository.

---

## 🔁 Reproducibility

Upon publication, this repository will include:

- End-to-end pipeline code  
- Prompt templates used in each workflow stage  
- Configuration files for model invocation  
- Scripts for evaluation and error analysis  

This will enable full reproduction of the results reported in the paper.

---

## 📜 License

The code will be released under an open-source license (to be specified upon publication).

---


## 🔖 Citation

If you find this work useful, please cite the corresponding paper once it is published.  
Citation information will be updated here after acceptance.
