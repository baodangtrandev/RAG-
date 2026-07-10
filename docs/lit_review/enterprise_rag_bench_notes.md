## EnterpriseRAG-Bench — Reading Notes

**Paper:** EnterpriseRAG-Bench: A RAG Benchmark for Company Internal Knowledge
**Authors:** Yuhong Sun, Joachim Rahmfeld, Chris Weaver, Roshan Desai, Wenxi Huang, Mark H. Butler
**Year:** May 2026
**Link:** [arXiv:2605.05253](https://arxiv.org/abs/2605.05253)

### Dataset Characteristics
- **Scale:** 500k+ documents
- **Sources:** 9 enterprise source types including Slack (285k), Gmail (121k), Jira (41k), Confluence (5k), Google Drive, Linear, HubSpot, Fireflies, and GitHub.
- **Nature of Data:** Synthetic but designed to be "messy" and mirror real-world corporate data. It features cross-document coherence (shared projects and initiatives), misfiled documents, near-duplicates, and conflicting information.

### Evaluation Protocol
- **Queries:** 500 questions across ten categories.
- **Capabilities Tested:** Multi-document reasoning, constrained retrieval, and conflict resolution.
- **Resources:** Open-source evaluation harness, leaderboard, and a generation framework for custom datasets.

### Key Findings
1. Standard RAG models struggle with the "messiness" of enterprise data compared to clean, public datasets like Wikipedia.
2. The presence of conflicting information and near-duplicates significantly degrades retrieval precision.
3. Multi-document reasoning requires effective linking of context across different source types (e.g., Slack to Jira).

### Limitations mentioned by authors
1. The dataset is synthetic, which may not capture 100% of the nuances of true proprietary enterprise data.
2. Focuses primarily on text, missing rich multimedia or highly structured database queries.

### Research Gap (our angle for T-RAG)
- **Standard RAG fails because:**
  1. **Temporal Blindness:** It struggles to resolve conflicting information when the truth depends on the recency of the document (e.g., outdated Jira tickets vs. recent Slack updates).
  2. **Vector Density at Scale:** With 500k+ docs, semantic similarities cluster too tightly, causing precision drops that require robust metadata pre-filtering.
  3. **Vocabulary Mismatch:** Natural language queries often fail to match the jargon, IDs, or specific terminology found in Jira or GitHub, necessitating a hybrid retrieval approach.
