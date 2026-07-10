# EvaRAG_Evaluating_Advanced_RAG_Techniques_With_Indexing_and_Distance_Metrics

## Page 1

Received 21 November 2025, accepted 27 November 2025, date of publication 22 December 2025,
date of current version 29 December 2025.
Digital Object Identifier 10.1 109/ACCESS.2025.3646665
EvaRAG: Evaluating Advanced RAG Techniques
With Indexing and Distance Metrics
HARUN ELKIRAN
 1, (Member, IEEE),
AND JAWAD RASHEED
 1,2,3,4, (Member, IEEE)
1Department of Computer Engineering, Istanbul Sabahattin Zaim University, 34303 Istanbul, Türkiye
2Department of Software Engineering, Istanbul Nisantasi University, 34398 Istanbul, Türkiye
3Research Institute, Istanbul Medipol University, 34810 Istanbul, Türkiye
4Applied Science Research Center, Applied Science Private University, Amman, Jordan
Corresponding author: Jawad Rasheed (jawad.rasheed@izu.edu.tr)
ABSTRACT Retrieval Augmented Generation (RAG) has emerged as a powerful paradigm for enhancing
large language models (LLMs) with external knowledge. Y et, the performance of RAG pipelines is
susceptible to design choices across retrieval, similarity metrics, indexing, and reranking. Despite growing
adoption, little systematic work has explored the trade-offs between retrieval quality, semantic accuracy,
computational efficiency, and cost in RAG systems. This study addresses this gap by conducting a
comprehensive evaluation of RAG configurations across multiple dimensions. We propose a benchmarking
framework that systematically varies retrievers (Fusion, HyDe, Hierarchical, SCaNN), indexing methods
(HNSW, IVF, Flat), similarity metrics (Cosine, Inner Product, L2), and rerankers (BGE, minilm) over
datasets of three scales (small, medium, and large). Performance is assessed through coverage, recall, MRR,
and nDCG, while semantic quality is measured using correctness, faithfulness, and relevance. Efficiency is
quantified via latency, throughput, and computational cost. Our experiments reveal that HNSW–IP–Fusion–
minilm achieves the strongest semantic performance, with Coverage Retrieval of 0.942, Correctness of 0.909,
and Faithfulness of 0.970, making it ideal for accuracy-critical tasks. Conversely, IVF–L2–Hierarchical
demonstrates the lowest latency (1.736 ns) and cost, making it suitable for real-time deployments. Reranker
analysis shows modest but consistent gains for minilm over BGE, while HyDe excels in precision at
the expense of efficiency. Notably, no single configuration dominates; optimal designs depend on the
application’s needs, whether it is maximizing semantic accuracy, minimizing latency, or striking a balance
between the two. By demonstrating concrete trade-offs, this work provides a practical foundation for scaling
RAG pipelines across diverse domains, including information retrieval, enterprise search, and knowledge-
intensive reasoning.
INDEX TERMS Data retrieval, large language model, natural language processing, question answering
systems, RAG.
I. INTRODUCTION
Recent breakthroughs in artificial intelligence and natural
language processing have led to the development of powerful
large language models (LLMs), such as the Generative
Pre-trained Transformer (GPT). The rapid progress of
LLMs can be attributed to improvements in deep learning
techniques, the development of large-scale transformers, and
The associate editor coordinating the review of this manuscript and
approving it for publication was Maria Chiara Caschera
.
the availability of massive datasets. Models such as GPT-4 [1]
and Llama 2 [2] excel in various tasks and domains, often
without prompts.
These models have significant potential in various
domains, including coding, medicine, law, agriculture, and
psychology, and are approaching human-level knowledge,
[3], [4], [5], [6]. Although LLMs have extensive pre-
training knowledge, their lack of customized domain-specific
understanding or knowledge of recent events can lead to
outdated or unfounded responses in real-world applications,
215724

 2025 The Authors. This work is licensed under a Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 License.
For more information, see https://creativecommons.org/licenses/by-nc-nd/4.0/
VOLUME 13, 2025

## Page 2

H. Elkiran, J. Rasheed: EvaRAG: Evaluating Advanced RAG Techniques
also known as hallucinations [7], [8], [9], [10]. Unreliability
is a significant barrier to the safe adoption of LLM-based
systems for essential business applications, due to user trust
issues stemming from hallucinations.
Retrieval Augmented Generation (RAG) has recently
gained adoption in question-answering systems because it
enhances query context and improves semantic ground-
ing [11]. Text retrieval is crucial for various information
retrieval applications, including search, (Q&A), and different
types of recommendation systems. RAG systems [12], [13]
retrieve text to provide scope to LLMs. RAG pipelines
use approximate nearest-neighbor (ANN) search methods to
retrieve relevant documents from large databases. The LLM
uses the retrieved text chunk to provide accurate, timely
responses.
The retrieval stage is a critical component of RAG
pipelines, as it relies on similarity measurements and efficient
indexing systems. Semantic closeness between embeddings
is measured through metrics, such as cosine similarity,
Euclidean distance, and inner product. To scale this process,
approximate nearest neighbor search (ANNS) [14] methods
use indexing techniques [13] such as Hierarchical Navigable
Small World (HNSW), ScaNN, and Inverted File Index (IVF).
These indices enable efficient retrieval from large embedding
collections by organizing vectors into graph- or clustering-
based structures. Beyond initial retrieval, rerankers [15] like
BGE and MiniLM enhance candidate results by utilizing
cross-encoder models, thereby improving contextual align-
ment between the query and retrieved passages.
While RAG offers a promising direction for mitigating
hallucinations, existing studies typically evaluate RAG
pipelines under limited conditions. Most prior work focuses
on a single retriever-reranker configuration or on small-scale
datasets, making it challenging to compare design choices
systematically. There remains a lack of comprehensive
frameworks that explore the whole design space of chunking,
embeddings, retrievers, rerankers, and efficiency trade-offs
(accuracy, latency, and cost). To address this issue, this
study conducts a comparative analysis of various indexes,
distance metrics, dataset sizes, retrievers, and rerankers,
providing conclusive results on which configuration performs
better under specific conditions. The following are the key
contributions of this study.
1) Presenting EV ARAG, a reproducible RAG bench-
marking pipeline that integrates dataset preparation,
chunking, embeddings, indexes (HNSW, ScaNN, and
IVF), retrievers (Fusion, Hierarchical, and HyDe), and
rerankers (BGE, MiniLM).
2) Conducted 162 experiments across multiple dataset
scales, enabling controlled comparisons of retrieval
effectiveness, generative quality, and efficiency trade-
offs.
3) Revealing the strengths and limitations of indexes,
distance/similarity metrics, retrievers, and rerankers,
as well as the trade-offs between accuracy, latency, and
cost.
The remainder of this paper is structured as follows. SectionII
reviews related work on RAG and evaluation methodologies.
Section III describes the EV ARAG methodology in detail.
Section IV presents experimental results across all configu-
rations. Section V discusses key findings and implications.
Finally, Section VI concludes the paper with directions for
future research.
II. LITERATURE REVIEW
RAG integrates retrieval and generation techniques to
enhance language modelling tasks. The RAG pipeline has two
major steps: retrieving relevant information and generating
contextually informed text. Table 1 summarizes key RAG
studies, highlighting the retrievers, rerankers, indexing meth-
ods, and distance/similarity metrics employed.
RAG systems use a range of retrieval methods, includ-
ing semantic search and FAISS-based similarity matching,
to efficiently identify relevant information [16]. To retrieve
the most pertinent text, the choice of index plays a central
role. Approaches such as HNSW [17], ScaNN [14], and
IVF [18] enable scalable ANNs through their efficient index-
ing schemes. These approaches strike a balance between
recall and latency. Similarly, FAISS has been shown to
achieve strong performance in high-dimensional vector
search [19]. These methods reduce search complexity but
often trade recall for efficiency, which impacts the quality
of downstream text generation, a key limitation of these
systems.
RAG performance is strongly influenced by how docu-
ments are chunked, retrieved, and filtered before generating
responses. Effective chunking strategies, such as those evalu-
ated by holistic frameworks like HOPE[20], can boost factual
accuracy and coherence. Meanwhile, semantically guided
chunking using LLMs to produce richer, more meaningful
text segments and metadata [21]. Retrieval quality also
remains a core challenge. Techniques like W-RAG [22]
leverage signals from LLM behavior to improve retriever
training, achieving performance close to that of human-
labeled data. For filtering noisy or irrelevant retrievals,
multi-agent approaches like MAIN-RAG [23] tap into LLM
consensus all without extra training. By rearranging candi-
date documents according to semantic relevance, reranking
improves the initial retrieval results. Cross-encoders and
lightweight transformer-based models, such as BGE [24]
and MiniLM [25], are widely adopted as rerankers. Rerank-
ing has been shown to improve the contextual accuracy
of retrieved outputs [19]. Moreover, hybrid systems that
combine multiple retrievers with rerankers often outperform
single-method approaches, particularly for complex or deep
logic queries [24]. Other strategies, such as HyDE, further
enrich retrieval by generating hypothetical documents to
expand the search space.
Evaluating the effectiveness of retrievers relies on similar-
ity measures between query and document embeddings. Stan-
dard metrics include cosine similarity and Euclidean distance,
VOLUME 13, 2025 215725

## Page 3

H. Elkiran, J. Rasheed: EvaRAG: Evaluating Advanced RAG Techniques
with cosine similarity demonstrating superior performance
in most RAG applications [24]. Embedding-based scoring
methods, such as BERT embedding similarity, also provide
strong alignment with semantic meaning. Recently, the
Overall Performance Index (OPI) was proposed, combining
logical correctness and embedding similarity into a unified
measure for holistic RAG evaluation [24].
Several benchmarks have been introduced to assess
retrieval and generation. KILT [12] integrates knowledge-
intensive NLP tasks, whereas domain-specific studies in
medicine [4]. More recent evaluations focus on hallucina-
tion mitigation and factual consistency [8]. Despite these
contributions, most benchmarks evaluate either retrieval or
generation in isolation, with limited insight into trade-offs
across system configurations. This leaves practitioners with
little guidance on the interplay between pipeline com-
ponents, such as retrievers, indexers, and rerankers, and
distance/similarity metrics and dataset size. EV ARAG closes
this gap by providing a unified, reproducible framework for
systematically comparing RAG configurations. By varying
indexes, retrieval and reranking strategies, and measuring
both effectiveness and efficiency metrics, EV ARAG provides
actionable, holistic insights into the trade-offs and best-fit
configurations.
III. METHODOLOGY
The proposed methodology for EV ARAG follows a struc-
tured RAG pipeline, as illustrated in Fig. 1. The work-
flow integrates datasets, embedding modelling, indexing,
retrieval, reranking, and evaluation. The entire process
is designed to be modular, reproducible, and extensi-
ble, enabling comprehensive benchmarking under con-
trolled conditions. The methodology is described in detail
below.
A. DATASETS
The first step in the EV ARAG is data collection. In this
study, we used the Stanford Question Answering Dataset
(SQuAD) [31]. It is widely used as a benchmark for question-
answering tasks. SQuAD consists of more than 100,000
question-answer pairs generated by human annotators from
Wikipedia articles. Each instance in the dataset includes
a context passage, a question, and one or more ground
truth answers. This dataset is particularly well-suited for
evaluating RAG systems where both context retrieval and
answer generation quality are important. In this study,
to capture the impact of dataset scale on RAG performance,
three dataset configurations are used: Small Dataset of
10,000 documents for fast baseline experimentation and
debugging, a Medium Dataset of 30,000 documents to
examine performance trends as the retrieval space grows,
and a Large Dataset of 100,000 documents to approximate
production level scenarios, stressing both indexing efficiency
and retrieval accuracy.
B. CHUNKING
Document chunking is a crucial preprocessing step that
enables efficient and scalable retrieval within the RAG
pipeline. In chunking, instead of treating each document
as a single large unit, the corpus is segmented into
overlapping chunks. Chunking allows fine-grained retrieval.
It preserves contextual continuity. Each chunk is uniquely
identified by the tuple (doc_id, chunk_id), ensuring
precise traceability back to the original document. It is
used in ground truth evaluation. In this study, chunks
were created with a fixed window size of 1,000 characters
and an overlap of 200 characters. This ratio strikes a
balance between granularity and contextual completeness.
The overlap set in this study ensures that information
appearing near chunk boundaries is not lost. It preserves
the narrative flow and semantic continuity across adjacent
chunks. Chunk overlap is significant for tasks involving
question answering or knowledge-intensive reasoning, where
critical context may span multiple segments. This strategy
of carefully selecting an overlap value improves retrieval
precision by narrowing the search to contextually relevant
passages while maintaining high recall through overlap,
ultimately leading to more accurate, contextually grounded
responses. The resulting chunked representation facilitates
efficient embedding generation, as each chunk can be
independently encoded into a vector space for downstream
semantic search.
C. EMBEDDING MODELING
Following chunk creation, each segment is transformed into
a dense vector embedding to enable semantic retrieval. This
vector representation enables semantic search and similarity-
based retrieval. For this purpose, we have used OpenAI’s
text-embedding-3-large model, one of the most
advanced embedding models for capturing rich semantic and
contextual information from natural language text. It pro-
duces dense embeddings with a maximum dimensionality of
d = 3072.
Mathematically, C(D) = {C 1, C2, . . . ,Cn} denotes the set
of chunks generated from a document D. Each chunk Ci is
mapped into a continuous vector space using the embedding
function fθ (·).
ei = fθ (Ci), ei ∈ Rd , d = 3072. (1)
Equation (1) produces a dense embedding vector ei for
each chunk, resulting in an embedding matrix shown by
equation 2
E = [e1, e2, . . . ,en]⊤ ∈ Rn×d , (2)
Together, Equations (1) and (2) define the semantic
representation layer of the RAG pipeline. The resulting
matrix E serves as the foundation for indexing and similarity
search.
215726 VOLUME 13, 2025

## Page 4

H. Elkiran, J. Rasheed: EvaRAG: Evaluating Advanced RAG Techniques
TABLE 1. Summary of recent literature on Retrieval-Augmented Generation (RAG).
FIGURE 1. Illustration of the complete experimental pipeline, including dataset selection (large, medium, small), indexing strategies (HNSW, IVF, ScaNN),
distance metrics (cosine, inner product, L2), retrieval strategies (fusion, hierarchical, HyDE), and reranking (BGE, MiniLM) used to generate and evaluate
results.
D. STORING VECTORS
Once the embeddings are generated, the next step is to
store them in a vector database to enable scalable, low-
latency, similarity-based retrieval. Each embedding ei from
Equation (1) is stored together with its corresponding
doc_id and chunk_id. This storage mechanism ensures
precise traceability back to the original document. These
metadata connections allow downstream evaluation. This
allows retrieved results to be directly compared with
ground-truth answers for recall and relevance analysis.
In this study, we use Milvus, a high-performance, open
source vector database specifically designed for large-scale
similarity search [25]. Milvus supports both dense and hybrid
(dense + sparse) embeddings and is optimized for real-time
ANN queries.
E. INDEXING STRATEGIES
To enable efficient similarity search at scale, we construct
ANN indexes for each dataset configuration (10k, 30k,
100k embeddings). The choice of indexing strategy has a
VOLUME 13, 2025 215727

## Page 5

H. Elkiran, J. Rasheed: EvaRAG: Evaluating Advanced RAG Techniques
significant impact on both latency and recall, especially
in large embedding spaces where exhaustive search is
computationally expensive. Milvus supports several ANN
algorithms, and we have experimented with three widely
adopted methods. Let’s discuss each one of them.
1) HNSW (HIERARCHICAL NAVIGABLE SMALL WORLD)
HNSW is a graph-based ANN algorithm that organizes the
embedding set E = {e 1, e2, . . . ,en} into a hierarchy of
proximity graphs. Each layer ℓ ∈ {1, . . . , L} contains a
subset of nodes, with edges connecting each vector ei to its
M nearest neighbors according to the chosen distance metric
(e.g., cosine or L2). At query time, the search procedure
performs greedy routing through graph G, starting from an
entry point in the top layer and iteratively descending through
layers to reach the base layer, where a local neighborhood
search is conducted. The complexity of HNSW is shown by
equation 3
Tsearch = O(log n) (3)
Equation 3 shows that HNSW is highly efficient for
large-scale, high-dimensional embeddings. The graph-based
structure allows HNSW to avoid exhaustive scanning,
significantly reducing query latency while maintaining near-
exact recall.
2) IVF (INVERTED FILE INDEX)
IVF partitions the embedding space into k coarse clusters by
performing k mean clustering over all embeddings during
the indexing stage. This process assigns each embedding
vector ei to its nearest centroid µc(i), minimizing the within-
cluster variance. Equation (4) ensures that clusters are formed
optimally, supporting efficient coarse-to-fine retrieval.
min
{µj}k
j=1
n∑
i=1
ei − µc(i)

2
2. (4)
At query time, only the nprobe clusters whose centroids are
closest to the query embedding are searched, significantly
reducing the candidate set size. As a result, IVF achieves a
tunable balance between search speed and recall.
3) SCANN (SCALABLE NEAREST NEIGHBORS)
ScaNN is a state-of-the-art ANN method that combines tree
partitioning with vector quantization to accelerate nearest
neighbor search. During indexing, the embedding space
is partitioned into clusters, and each embedding vector
ei is stored as a quantized code ˆei that minimizes the
reconstruction error. In ScaNN, the quantized representation
preserves semantic proximity, enabling efficient approximate
retrieval as shown in Equation (5). ScaNN keeps the recall
close to exact search.
ˆei = arg min
q∈Q
ei − q

2
2, (5)
where Q is the codebook of quantized vectors. At query time,
ScaNN uses a two-stage search: first, a coarse search over
partition centroids, then a re-ranking of the top k candidates
using the exact distance between q and ei.
F. SIMILARITY METRICS
The choice of similarity metric plays a critical role in
determining which vectors are considered nearest neighbors.
To assess the relevance of candidate chunks to a given query,
a similarity function s(q, ei) is used to compare the query
embedding q ∈ Rd with each candidate embedding ei ∈ Rd .
In this work, we explore three widely used formulations for
s(·, ·) cosine similarity, inner product, and L2 distance, each
offering a different notion of closeness in the embedding
space. These three metrics are commonly used and are
supported by Milvus.
1) COSINE SIMILARITY
Cosine similarity measures the angular closeness between
two vectors, capturing the degree to which they point in the
same direction, irrespective of their magnitudes. Formally,
for a query embedding q ∈ Rd and a candidate embedding
ei ∈ Rd , the cosine similarity score is defined by equation 6
scos(q, ei) = q · ei
∥q∥2∥ei∥2
. (6)
Equation (6) produces values in the range [−1, 1], where
1 indicates perfect directional alignment and −1 indi-
cates complete opposition. For most contextual embeddings
derived from modern language models, values typically range
between 0 and 1.
2) INNER PRODUCT (DOT PRODUCT SIMILARITY)
Inner product similarity directly measures the alignment and
magnitude between a query embedding q ∈ Rd and a
candidate embedding ei ∈ Rd . It is shown by equation 7
sip(q, ei) = q · ei. (7)
Equation (7) produces a scalar value that increases with
both directional alignment and vector magnitude, making
it particularly suitable for ranking-based retrieval systems
where larger dot products correspond to higher similarity or
relevance.
3) L2 DISTANCE (EUCLIDEAN DISTANCE)
L2 distance, also known as Euclidean distance, measures the
absolute geometric separation between a query embedding
q ∈ Rd and a candidate embedding ei ∈ Rd using equation 8
dL2(q, ei) = ∥q − ei∥2 =
√
d∑
j=1
(qj − ei,j)2. (8)
Unlike other similarity-based metrics, a lower L2 distance
indicates closer neighbors. This metric is sensitive to vector
magnitude, which can be desirable when absolute position in
the embedding space encodes semantic confidence.
215728 VOLUME 13, 2025

## Page 6

H. Elkiran, J. Rasheed: EvaRAG: Evaluating Advanced RAG Techniques
G. RETRIEVAL STRATEGIES
After storing the embeddings in the vector database, the next
step is to develop retrieval strategies that efficiently fetch
relevant chunks. In this study, we evaluate three comple-
mentary retrieval paradigms: Fusion Retrieval, Hierarchical
Retrieval, and Hypothetical Document Embeddings (HyDE).
Each retrieval strategy is implemented on top of the same
Milvus database, ensuring that performance differences are
attributable to the retrieval approach rather than the database
configuration.
1) FUSION RETRIEVAL
Fusion retrieval is a rank-aggregation strategy that combines
the results of multiple queries or retrieval configurations.
Instead of relying on a single retrieval output, we compute
ranked lists from different similarity metrics (Cosine, Inner
Product, L2) and merge them using rank fusion techniques
such as Reciprocal Rank Fusion (RRF). The RRF score of a
document d across k rank lists is given by equation 9
RRF(d) =
k∑
i=1
1
c + ri(d) (9)
In equation 9 ri(d) is the rank position of document d in the
ith list, and c is a constant (typically c = 60) used to smooth
the contribution of lower-ranked results. This ensures that
documents appearing consistently in the top positions across
retrieval methods receive higher final scores.
2) HIERARCHICAL RETRIEVAL
Hierarchical retrieval decomposes the search process into
multiple stages to improve efficiency and relevance. In the
first stage, we perform a coarse-grained retrieval using a
fast, low-dimensional embedding space to obtain a candidate
set C. In the second stage, a fine-grained reranking step
is applied using the original high-dimensional embeddings
using a scoring function. Mathematically, let q be the query
embedding, E be the set of all document embeddings, and C
be the candidate set retrieved in stage one, it is represented by
equation 10.
C = TopK
(
ANNcoarse(q, E)
)
(10)
Then, the final ranked list R is produced by re-scoring
C using a more accurate similarity function S, which is
represented by equation 11.
R = Sort
(
{(d, S(q, d)) | d ∈ C}
)
(11)
This two-stage approach balances computational cost and
accuracy, as expensive reranking is only applied to a small
candidate set.
3) HYDE RETRIEVAL
Hypothetical Document Embeddings (HyDE) retrieval is a
generative-augmented retrieval technique in which the query
is expanded by generating a synthetic ‘‘hypothetical answer’’
before embedding. The process consists of three steps:
1) Generate a pseudo answer ˆa using a generative model
such as GPT.
2) Compute the embedding eˆa of ˆa using the same
embedding model used for document chunks.
3) Perform retrieval using eˆa instead of eq, thereby pulling
semantically richer documents that align with the
generated context.
HyDE is particularly effective when queries are sparse or
underspecified, as the generative step introduces additional
context that helps match relevant chunks.
H. RERANKING
After initial retrieval, reranking is applied to refine the
candidate list and improve the relevance of retrieved chunks
for downstream generative tasks. Reranking leverages more
computationally expensive, context-aware models that can
consider pairwise interactions between the query and each
candidate document. In this study, we experiment with
two state-of-the-art reranking models: BGE and MiniLM
Cross-Encoder.
1) BGE (BI-ENCODER) RERANKING
BGE is a bi-encoder architecture that maps both the query and
candidate documents independently into a shared embedding
space, enabling efficient similarity computation. Let q be the
query and di a candidate chunk, with embeddings eq = fθq(q)
and edi = fθd (di). The similarity score is computed using
equation 12
sBGE(q, di) = cos(eq, edi) = eq · edi
∥eq∥2∥edi∥2
. (12)
Because embeddings are precomputed for the candidate
documents, scoring can be performed efficiently using
vector operations over the top k retrieved candidates. BGE
excels at capturing semantic similarity while remaining
computationally feasible for large candidate sets.
2) MINILM CROSS-ENCODER
MiniLM Cross Encoder is a transformer-based model that
jointly encodes the query and candidate document, allowing
full attention across both sequences. Unlike bi-encoders, this
approach models fine-grained interactions between query
tokens and document tokens. For query q and document di,
the cross encoder produces a scalar relevance score using
equation 13
sCE(q, di) = CrossEncoder(q, di; θCE), (13)
In equation 13 θCE are the parameters of the MiniLM
model. The cross encoder directly outputs a relevance prob-
ability or score, which is used to rerank the top k candidates
retrieved by the ANN index. Although Cross-Encoders pro-
vide superior accuracy by modeling token-level interactions,
they are computationally more expensive than bi-encoder
methods, making them suitable primarily for reranking a
small subset of candidate documents rather than the entire
corpus.
VOLUME 13, 2025 215729

## Page 7

H. Elkiran, J. Rasheed: EvaRAG: Evaluating Advanced RAG Techniques
I. QUESTION ANSWERING PIPELINE
After the final reranking step, the top k most relevant
document chunks are selected to serve as contextual evidence
for the language model. Let Ctop = {C 1, C2, . . . ,Ck } denote
the set of these top-ranked chunks returned by the retrieval
and reranking stages. The user query Q is then combined
with these chunks to form a structured input prompt for the
language model using equation 14.
Input = FormatPrompt(Q, Ctop) (14)
In equation 14, where FormatPrompt ensures clear
separation of the context from the question and incorpo-
rates instructions for generating grounded responses. The
model used in this study is gpt-4o-mini. In the end,
LLM integrates information from all selected chunks while
maintaining semantic coherence and factual consistency.
Since the scope of this study is to evaluate the impact
of indexing, reranking, and similarity metrics on the RAG
system, we have, for consistency, used only a single LLM.
J. EVALUATION
The performance of the proposed RAG pipeline is assessed
through a comprehensive set of evaluation metrics, spanning
the retrieval, reranking, and generation stages. This multilevel
evaluation provides a detailed understanding of both the
quality of information retrieval and the efficiency and
effectiveness of the final generated answers.
1) RETRIEVAL AND RERANKING METRICS
The retrieval and reranking stages are evaluated using stan-
dard information retrieval metrics, capturing both accuracy
and ranking quality:
• Recall@k (R@k): Recall@k measures the fraction of
queries for which at least one relevant document appears
in the top k retrieved results. Formally, for a set of
queries Q and corresponding relevant documents Rq for
query q, R@k is defined by equation 15
R@k = 1
|Q|
∑
q∈Q
1
(
|Topk (q) ∩ Rq| ≥ 1
)
, (15)
Topk (q) denotes the set of top-k retrieved documents for
query q, and 1(·) is the indicator function.
• Mean Reciprocal Rank (MRR): MRR evaluates the
average rank position of the first relevant document in
the retrieved list. It is computed using equation 16.
MRR = 1
|Q|
∑
q∈Q
1
rankq
, (16)
rankq is the rank of the first relevant document for query
q. Higher MRR indicates that relevant documents are
ranked closer to the top.
• Normalized Discounted Cumulative Gain
(NDCG@k): NDCG@k accounts for both relevance
and position, assigning higher weight to relevant
documents appearing earlier in the ranked list. It is
calculated using equation 17.
NDCG@k = DCG@k
IDCG@k , DCG@k =
k∑
i=1
2reli − 1
log2(i + 1) ,
(17)
In equation 17 rel i is the graded relevance of the
document at rank i, and IDCG@k is the ideal DCG
obtained by perfect ranking.
• Coverage: Coverage indicates the proportion of queries
for which at least one relevant document is retrieved,
providing a measure of the retrieval system’s complete-
ness across all queries.
2) COMPLETION METRICS
Once the top-ranked document chunks are identified and
provided as context to the generative language model, the
evaluation focuses on efficiency and cost-related metrics:
• Latency: The time required by the model to generate
an answer for a given query, measured from prompt
submission to response completion. Lower latency
indicates better realtime performance.
• Token Usage: The total number of tokens consumed,
including both prompt and completion tokens. Token
usage reflects both computational efficiency and poten-
tial API costs when using commercial LLMs.
• Cost: Monetary cost per query, computed based on
token usage and model pricing. This metric allows eval-
uation of economic efficiency in addition to technical
performance.
IV. EVARAG EXPERIMENTAL DESIGN
To systematically evaluate the RAG pipeline, we designed
experiments by combining multiple factors: dataset size,
indexing method, similarity metric, retrieval strategy, and
reranking approach. The parameters used are summarized in
Algorithm 1.
All possible combinations of these parameters were
generated to create the full set of experiments. The total
number of experiments is calculated using the Cartesian
product and are represented by 18
Nexperiments = |D | × |I | × |S | × |Rret| × |Rrerank|
= 3 × 3 × 3 × 3 × 2 = 162. (18)
To illustrate the experimental design example, a subset
of the generated RAG parameter combinations is shown
in Table 2. Each row represents a unique experiment
defined by the combination of dataset size, indexing method,
similarity metric, retrieval strategy, and reranking approach.
The experiments are numbered sequentially for clarity. Only
a few examples are presented here to illustrate the structure;
the complete set comprises 162 distinct experiments that
cover all possible parameter combinations. These parameters
were chosen because they reflect the most widely used
and practically relevant design decisions in today’s RAG
215730 VOLUME 13, 2025

## Page 8

H. Elkiran, J. Rasheed: EvaRAG: Evaluating Advanced RAG Techniques
Algorithm 1 Generate All RAG Parameter Combinations
Require: Lists of parameters: DatasetList, IndexingList,
SimilarityList, RetrievalList, RerankingList
Ensure: List of all possible RAG parameter combinations.
DatasetList ← {Small, Medium, Large}
IndexingList ← {HNSW, ScaNN, IVF}
SimilarityList ← {Cosine, Inner Product, L2}
RetrievalList ← {Fusion, Hierarchical, HyDE}
RerankingList ← {BGE, MiniLM, –}
Combinations ← empty list
for dataset in DatasetList do
for indexing in IndexingList do
for similarity in SimilarityList do
for retrieval in RetrievalList do
for reranking in RerankingList do
Add (dataset, indexing similarity, retrieval,
reranking) to Combinations.
end for
end for
end for
end for
end for
return Combinations
TABLE 2. Example combinations of experiments (Subset).
pipelines. Dataset size, indexing structure, and similarity
metric form the foundation of any retrieval system, while
retrieval strategies such as Fusion, Hierarchical Search, and
HyDE represent the prevailing paradigms found in the current
literature. Likewise, BGE and MiniLM are lightweight
rerankers commonly selected for scalable RAG solutions.
Together, these parameters cover the full range of typical
RAG configurations, ensuring that the experimental design is
comprehensive, practical, and aligned with real-world system
requirements.
Table 3 shows the implementation details for the EvaRAG
pipeline, focusing on libraries, models, and APIs for
reproducibility.
V. RESULTS
The experiments were implemented using Python and
rely on several key libraries langchain, langchain-openai,
langchain_community, pymilvus, and langchain-milvus.
Milvus Lite was used for vector storage and retrieval. All
experiments were performed on a MacBook Pro (Late 2019)
with Intel Core i9 8-Core CPU (2.3 GHz), 32 GB DDR4
TABLE 3. Key implementation details of EvaRAG pipeline focusing on
libraries, models, and APIs for reproducibility.
RAM (2667 MHz), macOS 15.6.1. Let’s discuss the empirical
experiments in detail.
A. RETRIEVAL RESUL TS
Table 4 reports the retrieval performance of HNSW using
the Cosine metric across retrievers, rerankers, and dataset
sizes. Overall, the Fusion retriever achieves the best results,
with BGE reranking yielding R1 = 0.733, R3 = 0.870,
MRR = 0.807, and nDCG@10 = 0.839 on the Medium
dataset, and slightly lower but comparable performance on
Large datasets. The Hierarchical retriever performs poorly on
Large datasets (R1 = 0.448, MRR = 0.499), but improves
substantially on Small datasets (R1 = 0.655, MRR = 0.727).
The HyDe retriever provides a middle ground, e.g., the
Medium dataset with BGE reranker achieves R1 = 0.612,
R3 = 0.786, and nDCG@10 = 0.751. These results indicate
that Fusion + BGE is the most effective configuration,
Hierarchical benefits from smaller corpora, and HyDe offers
balanced performance.
Table 5 presents the retrieval performance of the HNSW
index using the Inner Product (IP) metric across different
retrievers, rerankers, and dataset sizes. The results show that
the Fusion retriever consistently outperforms the Hierarchical
and HyDe retrievers across all dataset sizes, with R@1
(R1) ranging from 0.720 to 0.750 for Fusion, compared to
0.454 to 0.569 for Hierarchical and 0.578 to 0.626 for HyDe.
Similarly, R@3 (R3) and R@10 (R10) follow the same trend,
with Fusion achieving up to 0.887 and 0.942, respectively.
The choice of reranker also impacts performance: BGE and
minilm show nearly identical results for Fusion, while minilm
slightly improves R@1–R@10 for Hierarchical and HyDe.
MRR values mirror these trends, with Fusion reaching a
maximum of 0.823, compared to 0.627 for Hierarchical
and 0.719 for HyDe. The nDCG metrics (nDCG1, nDCG3,
nDCG10) indicate that Fusion retrieves more relevant items
in top positions, achieving an nDCG10 of up to 0.852,
while Hierarchical and HyDe lag behind at 0.650 and
0.758, respectively. Overall, the table demonstrates that both
retriever choice and reranker selection have a substantial
effect on retrieval quality, and that the Fusion retriever with
either BGE or minilm reranker provides the most effective
combination for HNSW with the IP metric.
Table 6 presents the retrieval performance of the HNSW
index using the Euclidean (L2) distance across different
retrievers, rerankers, and dataset sizes. The Fusion retriever
VOLUME 13, 2025 215731

## Page 9

H. Elkiran, J. Rasheed: EvaRAG: Evaluating Advanced RAG Techniques
TABLE 4. Retrieval performance of HNSW with Cosine Metrics across different retrievers, rerankers, and dataset sizes. Metrics include Recall@1, @3, @10
(R1, R3 and R10, fraction of queries with correct item in top-K), MRR (average reciprocal rank of first correct result), and nDCG@1, @3, @10 (nDCG1,
nDCG3, nDCG10 account for relevance and position of retrieved items).
TABLE 5. Retrieval performance of HNSW with Inner Product (IP) across different retrievers, rerankers, and dataset sizes. Metrics include Recall@1, @3,
@10 (R1, R3 and R10, fraction of queries with correct item in top-K), MRR (average reciprocal rank of first correct result), and nDCG@1, @3, @10 (nDCG1,
nDCG3, nDCG10 account for relevance and position of retrieved items).
performs poorly with L2, achieving very low R1 (0.030–
0.046) and R10 (0.271–0.373), indicating that top-ranked
results rarely contain the correct item. Hierarchical retrievers
improve substantially with R1 ranging 0.481–0.572 and R10
0.631–0.731, while HyDe achieves the highest performance
among all retrievers (R1 0.580–0.625, R10 0.856–0.888).
MRR and nDCG metrics follow similar trends, showing
that Fusion fails to rank relevant items effectively with L2,
whereas HyDe maintains relatively high reciprocal rank and
relevance-aware ranking (nDCG10 up to 0.763). Overall,
these results highlight that the choice of distance metric has a
drastic impact on retrieval quality, with L2 being unsuitable
for Fusion retrievers but effective with HyDe and Hierarchical
configurations.
Table 7 reports the retrieval performance of the IVF index
using the Cosine metric across various retrievers, rerankers,
and dataset sizes. Fusion retrievers achieve the highest
performance, with R1 ranging from 0.683 to 0.736 and R10
ranging from 0.870 to 0.927, indicating that relevant items
are frequently ranked within the top results. HyDe retrievers
provide moderate performance (R1 0.565–0.618, R10 0.828–
0.870), while Hierarchical retrievers perform the lowest (R1
0.409–0.540, R10 0.535–0.688). MRR and nDCG metrics
reflect similar trends, indicating that Fusion consistently
ranks correct results higher and more accurately account
for relevance in the top positions. Overall, the choice of
retriever significantly impacts retrieval quality, with Fusion
outperforming both Hierarchical and HyDe across all dataset
sizes.
Table 8 presents the retrieval performance of the IVF
index with the Inner Product (IP) metric across different
retrievers, rerankers, and dataset sizes. Fusion retrievers
achieve the highest performance, with R1 ranging from
0.684 to 0.752 and R10 from 0.910 to 0.937, indicating
215732 VOLUME 13, 2025

## Page 10

H. Elkiran, J. Rasheed: EvaRAG: Evaluating Advanced RAG Techniques
TABLE 6. Retrieval performance of HNSW with Euclidean Distance (L2) across different retrievers, rerankers, and dataset sizes. Metrics include Recall@1,
@3, @10 (R1, R3 and R10, fraction of queries with correct item in top-K), MRR (average reciprocal rank of first correct result), and nDCG@1, @3, @10
(nDCG1, nDCG3, nDCG10 account for relevance and position of retrieved items).
TABLE 7. Retrieval performance of IVF with Cosine Metrics across different retrievers, rerankers, and dataset sizes. Metrics include Recall@1, @3, @10
(R1, R3 and R10, fraction of queries with correct item in top-K), MRR (average reciprocal rank of first correct result), and nDCG@1, @3, @10 (nDCG1,
nDCG3, nDCG10 account for relevance and position of retrieved items).
that relevant items are consistently retrieved within the
top ranks. HyDe retrievers show moderate performance
(R1 0.566–0.625, R10 0.832–0.880), while Hierarchical
retrievers generally perform lower (R1 0.434–0.643, R10
0.567–0.838). MRR and nDCG metrics align with this trend,
demonstrating that Fusion retrievers rank the correct results
higher and more accurately account for relevance in top
positions. Overall, the results highlight that Fusion-based
retrieval with IP outperforms both Hierarchical and HyDe
across all dataset sizes.
Table 9 presents the retrieval performance of the IVF index
with Euclidean Distance (L2) across different retrievers,
rerankers, and dataset sizes. Fusion retrievers exhibit very low
performance, with R1 ranging from 0.023 to 0.037 and R10
ranging from 0.271 to 0.370, indicating poor retrieval for top-
ranked items. Hierarchical retrievers improve considerably
(R1 0.431–0.553, R10 0.556–0.698), while HyDe retrievers
achieve the highest L2-based performance (R1 0.574–0.610,
R10 0.822–0.857). MRR and nDCG metrics follow the
same trend, reflecting that HyDe retrievers rank relevant
results higher and more accurately consider relevance at top
positions. Overall, L2 distance performs worse for Fusion but
remains effective for HyDe and Hierarchical retrievers across
all dataset sizes.
Table 10 reports the retrieval performance of SCANN with
Cosine similarity across retrievers, rerankers, and dataset
sizes. Fusion retrievers consistently perform best, achieving
R1 values between 0.690 and 0.733 and R10 values of up to
0.923, indicating highly accurate top-ranked retrieval. HyDe
retrievers follow with moderate performance (R1 0.569–
0.622, R10 0.830–0.877), while Hierarchical retrievers
remain the weakest (R1 0.440–0.589, R10 0.578–0.759).
MRR and nDCG values show the same ranking pattern,
confirming that SCANN with Cosine is most effective
VOLUME 13, 2025 215733

## Page 11

H. Elkiran, J. Rasheed: EvaRAG: Evaluating Advanced RAG Techniques
TABLE 8. Retrieval performance of IVF with Inner Product (IP) across different retrievers, rerankers, and dataset sizes. Metrics include Recall@1, @3, @10
(R1, R3 and R10, fraction of queries with correct item in top-K), MRR (average reciprocal rank of first correct result), and nDCG@1, @3, @10 (nDCG1,
nDCG3, nDCG10 account for relevance and position of retrieved items).
TABLE 9. Retrieval performance of IVF with Euclidean Distance (L2) across different retrievers, rerankers, and dataset sizes. Metrics include Recall@1,
@3, @10 (R1, R3 and R10, fraction of queries with correct item in top-K), MRR (average reciprocal rank of first correct result), and nDCG@1, @3, @10
(nDCG1, nDCG3, nDCG10 account for relevance and position of retrieved items).
with Fusion retrievers, especially on smaller datasets, where
performance peaks.
Table 11 reports the retrieval performance of SCANN with
Inner Product (IP) across retrievers, rerankers, and dataset
sizes. Fusion retrievers consistently achieve the highest
performance, with R1 ranging from 0.696 to 0.752 and R10
reaching 0.911 to 0.937, demonstrating strong top-ranked
retrieval accuracy. HyDe retrievers perform moderately well
(R1 0.572–0.627, R10 0.833–0.884), while Hierarchical
retrievers lag, especially on large datasets (R1 as low as
0.422, R10 0.559). MRR and nDCG values follow the same
trend, confirming that SCANN with IP combined with Fusion
retrievers provides the most effective retrieval across dataset
sizes.
Table 12 presents the retrieval performance of SCANN
with Euclidean Distance (L2). Fusion retrievers perform
poorly, with R1 as low as 0.022–0.038 and R10 below 0.36,
indicating weak top-ranked retrieval quality. Hierarchical
retrievers perform moderately, with improvements observed
in smaller datasets (R1: 0.398–0.558, R10: 0.506–0.704).
HyDe retrievers consistently achieve the best results for
L2, with R1 between 0.573 to 0.613 and R10 reaching
up to 0.865, clearly outperforming other retrievers under
this metric. MRR and nDCG trends mirror these results,
confirming that SCANN with L2 is most effective when
paired with HyDe retrievers.
B. RERANKING RESUL TS
The reranking results of HNSW with Cosine Similarity
are summarized in Table 13. Overall, performance varies
across retrievers, rerankers, and dataset sizes, with clear
trends indicating that larger datasets generally improve
retrieval effectiveness. Among retrievers, Fusion combined
215734 VOLUME 13, 2025

## Page 12

H. Elkiran, J. Rasheed: EvaRAG: Evaluating Advanced RAG Techniques
TABLE 10. Retrieval performance of SCANN with Cosine across different retrievers, rerankers, and dataset sizes. Metrics include Recall@1, @3, @10 (R1,
R3 and R10, fraction of queries with correct item in top-K), MRR (average reciprocal rank of first correct result), and nDCG@1, @3, @10 (nDCG1, nDCG3,
nDCG10 account for relevance and position of retrieved items).
TABLE 11. Retrieval performance of SCANN with Inner Product (IP) across different retrievers, rerankers, and dataset sizes. Metrics include Recall@1, @3,
@10 (R1, R3 and R10, fraction of queries with correct item in top-K), MRR (average reciprocal rank of first correct result), and nDCG@1, @3, @10 (nDCG1,
nDCG3, nDCG10 account for relevance and position of retrieved items).
with minilm reranker consistently yields the strongest
performance, achieving R1 scores above 0.80 for all dataset
sizes and reaching a peak of 0.82 on the medium dataset,
alongside high nDCG values (nDCG1 = 0.872). This
demonstrates that minilm reranking is highly effective when
paired with Fusion retrieval. In contrast, Hierarchical retrieval
shows relatively weaker performance, with R1 scores ranging
from 0.33 to 0.75, depending on the dataset size; however,
its performance improves markedly on the smaller dataset.
HyDe retrievers perform moderately, with minilm reranking
again outperforming BGE reranking, achieving R1 scores up
to 0.77. Across all configurations, MRR trends closely mirror
R1, confirming that systems retrieving correct items earlier
also deliver higher overall ranking quality. The consistently
strong nDCG3 and nDCG10 scores for Fusion + minilm
configurations indicate that not only are correct results found
early, but other relevant items are also ranked appropriately.
These findings highlight that pairing HNSW with cosine
similarity, Fusion retrieval, and minilm reranking is the most
effective configuration for achieving high top-K recall and
ranking quality across dataset scales.
Table 14 presents the reranking performance of HNSW
with Inner Product (IP). Overall, performance trends are
highly consistent with those observed for cosine similarity,
though IP yields slightly higher scores in several con-
figurations. Fusion retrieval with minilm reranking again
stands out as the strongest configuration, achieving the
highest R1 (0.825) and MRR (0.872) on the small dataset,
with similarly strong results across medium and large
datasets. This indicates that IP based HNSW retrieval benefits
from the dense representation power of Fusion combined
with minilm’s fine-grained reranking. Fusion + BGE also
performs well, with R1 values around 0.58–0.58 and steadily
improving nDCG scores as dataset size grows. Hierarchical
VOLUME 13, 2025 215735

## Page 13

H. Elkiran, J. Rasheed: EvaRAG: Evaluating Advanced RAG Techniques
TABLE 12. Retrieval performance of SCANN with Euclidean Distance (L2) across different retrievers, rerankers, and dataset sizes. Metrics include
Recall@1, @3, @10 (R1, R3 and R10, fraction of queries with correct item in top-K), MRR (average reciprocal rank of first correct result), and nDCG@1, @3,
@10 (nDCG1, nDCG3, nDCG10 account for relevance and position of retrieved items).
TABLE 13. Reranking performance of HNSW with Cosine Similarity across different retrievers, rerankers, and dataset sizes. Metrics include Recall@1, @3,
@10 (R1, R3 and R10, fraction of queries with correct item in top-K), MRR (average reciprocal rank of first correct result), and nDCG@1, @3, @10 (nDCG1,
nDCG3, nDCG10 account for relevance and position of retrieved items).
retrieval remains weaker overall but improves steadily with
dataset size, especially when paired with minilm reranking
(R1 = 0.638 on small dataset). HyDe retrievers perform
competitively in combination with minilm, achieving R1 up
to 0.775 and MRR above 0.81. The close alignment between
R1, MRR, and nDCG metrics confirms that systems not
only retrieve the correct items early but also effectively rank
multiple relevant items. These results suggest that HNSW
with IP is a strong alternative to cosine similarity, with
slightly better retrieval quality in many cases, particularly for
Fusion + minilm configurations.
Table 15 reports the reranking performance of HNSW
with Euclidean Distance (L2). Compared to cosine and inner-
product similarity, L2 generally produces lower R1 and MRR
scores for Fusion based retrieval, with Fusion + minilm
achieving only R1 = 0.335 on the large dataset. However,
Hierarchical retrieval benefits more under L2, especially with
minilm reranking, where performance improves significantly
as dataset size decreases, reaching R1 = 0.648 and MRR =
0.680 on the small dataset. HyDe retrievers continue to
perform competitively, particularly when combined with
minilm reranking, consistently achieving R1 above 0.74 and
strong nDCG scores across all dataset sizes, with the best
result on the small dataset (R1 = 0.783, nDCG1 = 0.830).
The results indicate that while L2 distance is less effective
for dense retrievers like Fusion, it still performs well with
Hierarchical and HyDe retrieval, especially when reranking
with minilm, which consistently yields the highest ranking
quality (MRR and nDCG values).
Table 16 presents the reranking performance of IVF
with Cosine similarity. Fusion-based retrieval with minilm
reranking consistently achieves the strongest results across
all dataset sizes, with R1 increasing from 0.760 (large)
to 0.818 (small) and MRR reaching 0.862, highlighting
215736 VOLUME 13, 2025

## Page 14

H. Elkiran, J. Rasheed: EvaRAG: Evaluating Advanced RAG Techniques
TABLE 14. Reranking performance of HNSW with Inner Product (IP) across different retrievers, rerankers, and dataset sizes. Metrics include Recall@1,
@3, @10 (R1, R3 and R10, fraction of queries with correct item in top-K), MRR (average reciprocal rank of first correct result), and nDCG@1, @3, @10
(nDCG1, nDCG3, nDCG10 account for relevance and position of retrieved items).
TABLE 15. Reranking performance of HNSW with Euclidean Distance (L2) across different retrievers, rerankers, and dataset sizes. Metrics include
Recall@1, @3, @10 (R1, R3 and R10, fraction of queries with correct item in top-K), MRR (average reciprocal rank of first correct result), and nDCG@1, @3,
@10 (nDCG1, nDCG3, nDCG10 account for relevance and position of retrieved items).
that IVF benefits from reranking more on smaller datasets.
BGE reranking with Fusion also performs competitively,
showing a steady gain in recall and nDCG as the dataset size
decreases, peaking at R1 = 0.548 and nDCG10 = 0.740 for
the small dataset. Hierarchical retrievers show moderate
improvements, particularly with minilm reranking, achieving
R1 = 0.608 and MRR = 0.639 on the small dataset. HyDe
retrievers also demonstrate strong performance, especially
when paired with minilm reranking, consistently achieving
R1 above 0.72 across dataset sizes. Overall, IVF with Cosine
shows a clear benefit from reranking, and minilm emerges as
the most effective reranker for improving top-rank accuracy
and overall ranking quality (MRR, nDCG).
Table 16 presents the reranking performance of IVF
with Cosine similarity. Fusion-based retrieval with minilm
reranking consistently achieves the strongest results across
all dataset sizes, with R1 increasing from 0.760 (large)
to 0.818 (small) and MRR reaching 0.862, highlighting
that IVF benefits from reranking more on smaller datasets.
BGE reranking with Fusion also performs competitively,
showing a steady gain in recall and nDCG as the dataset size
decreases, peaking at R1 = 0.548 and nDCG10 = 0.740 for
the small dataset. Hierarchical retrievers show moderate
improvements, particularly with minilm reranking, achieving
R1 = 0.608 and MRR = 0.639 on the small dataset. HyDe
retrievers also demonstrate strong performance, especially
when paired with minilm reranking, consistently achieving
R1 above 0.72 across dataset sizes. Overall, IVF with Cosine
shows a clear benefit from reranking, and minilm emerges as
the most effective reranker for improving top-rank accuracy
and overall ranking quality (MRR, nDCG).
Table 18 presents the performance of IVF with Euclidean
distance (L2) under various retrievers, rerankers, and dataset
sizes. Overall, L2-based reranking yields noticeably weaker
VOLUME 13, 2025 215737

## Page 15

H. Elkiran, J. Rasheed: EvaRAG: Evaluating Advanced RAG Techniques
TABLE 16. Reranking performance of IVF with Cosine across different retrievers, rerankers, and dataset sizes. Metrics include Recall@1, @3, @10 (R1, R3
and R10, fraction of queries with correct item in top-K), MRR (average reciprocal rank of first correct result), and nDCG@1, @3, @10 (nDCG1, nDCG3,
nDCG10 account for relevance and position of retrieved items).
TABLE 17. Reranking performance of IVF with Inner Product (IP) across different retrievers, rerankers, and dataset sizes. Metrics include Recall@1, @3,
@10 (R1, R3 and R10, fraction of queries with correct item in top-K), MRR (average reciprocal rank of first correct result), and nDCG@1, @3, @10 (nDCG1,
nDCG3, nDCG10 account for relevance and position of retrieved items).
results compared to Inner Product (IP) and Cosine similarity,
especially for Fusion with BGE, where R1 drops below
0.26 even on the largest dataset. MiniLM reranking improves
performance slightly but remains below 0.35 in R1 for
Fusion, indicating that L2 is suboptimal for dense embedding
similarity in this setting. In contrast, Hierarchical retrievers
achieve relatively better results under L2, particularly with
MiniLM, where R1 increases from 0.485 (large) to 0.618
(small), suggesting that L2 may complement hierarchical
clustering–based retrieval to some extent. HyDe retrievers
paired with MiniLM consistently deliver the strongest results
in this configuration, achieving R1 of up to 0.753 and MRR
of nearly 0.794 on the small dataset, which narrows the
gap with IP and Cosine. nDCG metrics follow the same
pattern, confirming that improvements occur not just in
recall but also in ranking quality. These findings suggest
that while L2 is generally less effective for embedding-based
retrieval, combining it with HyDe and MiniLM reranking
can partially mitigate performance loss, making it viable in
specific scenarios where L2 is required due to computational
or hardware constraints.
Table 19 reports the reranking performance of SCaNN with
Cosine similarity across retrievers, rerankers, and dataset
scales. Similar to IVF, Fusion with MiniLM consistently
yields the strongest outcomes, achieving R1 = 0.810 and
MRR = 0.855 on the small dataset, with only marginal drops
on larger datasets. Fusion with BGE performs moderately
well, reaching R1 around 0.534–0.542, while still benefitting
from larger datasets and showing a steady increase in nDCG
values. Hierarchical retrievers are weaker overall, especially
with BGE (R1 = 0.317 on the large dataset), though
MiniLM reranking provides a substantial boost, raising R1 to
0.673 on the small dataset. HyDe retrievers again demonstrate
competitive performance, with MiniLM pushing results to
215738 VOLUME 13, 2025

## Page 16

H. Elkiran, J. Rasheed: EvaRAG: Evaluating Advanced RAG Techniques
TABLE 18. Reranking performance of IVF with Euclidean distance (L2) across different retrievers, rerankers, and dataset sizes. Metrics include Recall@1,
@3, @10 (R1, R3 and R10, fraction of queries with correct item in top-K), MRR (average reciprocal rank of first correct result), and nDCG@1, @3, @10
(nDCG1, nDCG3, nDCG10 account for relevance and position of retrieved items).
R1 = 0.776 and MRR = 0.815 on small datasets, rivalling
Fusion. Across all settings, nDCG scores closely follow
recall trends, reinforcing that improvements reflect not only
higher retrieval coverage but also better placement of relevant
documents in the top ranks. These results indicate that
SCaNN with Cosine, particularly when paired with MiniLM
reranking, is highly effective, providing robust performance
across retrievers and dataset sizes.
Table 20 presents the reranking performance of SCaNN
with Inner Product across retrievers, rerankers, and dataset
scales. Fusion with MiniLM emerges as the most effective
combination, achieving the strongest results across all
dataset sizes, with R1 values reaching 0.828 and MRR =
0.872 on the small dataset, and maintaining consistently
high performance even on larger sets. Fusion with BGE
performs moderately well (R1 = 0.573–0.578), showing
incremental gains in R3 and R10 as the dataset size decreases.
Hierarchical retrievers underperform with BGE, particularly
on large datasets (R1 = 0.309). However, reranking with
MiniLM substantially improves the results, boosting R1
to 0.738 on small datasets. HyDe retrievers demonstrate
balanced performance, with MiniLM reranking producing
competitive scores (R1 = 0.781 and MRR = 0.821 on small
datasets), closely approaching the performance of Fusion.
Across all settings, nDCG trends mirror recall improvements,
indicating that gains are not only due to higher coverage
but also better ranking of relevant items. Overall, SCaNN
with IP , coupled with MiniLM, provides a robust and reliable
reranking pipeline that outperforms other retriever–reranker
combinations, especially on smaller datasets.
Table 21 reports the reranking performance of SCaNN with
Euclidean Distance (L2) across different retrievers, rerankers,
and dataset sizes. The results show a clear performance
gap depending on the retriever–reranker combination. Fusion
retrievers with either BGE or MiniLM yield comparatively
low R1 values (≤0.32) and shallow improvements in R3
and R10, suggesting weak alignment between L2 similarity
and semantic relevance. Hierarchical retrievers perform
moderately better, with gains from small dataset sizes (e.g.,
R1 = 0.620 for MiniLM) showing more substantial precision
at top ranks, though performance remains limited at larger
scales. In contrast, HyDe retrievers paired with MiniLM
consistently achieve the best results across dataset sizes,
with R1 above 0.72 and stable nDCG values, highlighting
their effectiveness in leveraging L2 for semantic match-
ing. Notably, HyDe + MiniLM small-scale configurations
achieve the highest overall performance (R1 = 0.762, R10 =
0.865, MRR = 0.801). These findings indicate that while
SCaNN with L2 underperforms in Fusion setups, its synergy
with HyDe–MiniLM offers competitive retrieval quality,
especially in smaller-scale collections.
C. COMPARATIVE ANAL YSIS
Fig. 2 shows the comparative analysis of HNSW, IVF, and
SCaNN across performance, latency, cost, and token metrics.
It reveals that all three indices achieve nearly identical
outcomes in terms of coverage, correctness, faithfulness,
relevance, token usage, and cost, indicating no significant
trade-offs in retrieval quality or resource consumption. The
main point of divergence lies in latency: SCaNN achieves
the lowest mean latency (3.05 ns) and p95 latency (4.44 ns),
followed closely by IVF, while HNSW incurs higher delays
(3.50 ns mean, 5.03 ns p95). This demonstrates that although
the indices are equivalent in reliability and efficiency from
a quality and cost standpoint, SCaNN provides superior
retrieval speed, making it the most suitable choice for latency-
sensitive applications.
As shown in Fig. 3, the analysis of dataset size (Large,
Medium, Small) indicates that retrieval quality metrics such
as coverage, correctness, faithfulness, and relevance remain
stable across different sizes, with only marginal improve-
ments for smaller datasets. For example, correctness rises
VOLUME 13, 2025 215739

## Page 17

H. Elkiran, J. Rasheed: EvaRAG: Evaluating Advanced RAG Techniques
TABLE 19. Reranking performance of SCaNN with Cosine across different retrievers, rerankers, and dataset sizes. Metrics include Recall@1, @3, @10 (R1,
R3 and R10, fraction of queries with correct item in top-K), MRR (average reciprocal rank of first correct result), and nDCG@1, @3, @10 (nDCG1, nDCG3,
nDCG10 account for relevance and position of retrieved items).
TABLE 20. Reranking performance of SCaNN with Inner Product (IP) different retrievers, rerankers, and dataset sizes. Metrics include Recall@1, @3, @10
(R1, R3 and R10, fraction of queries with correct item in top-K), MRR (average reciprocal rank of first correct result), and nDCG@1, @3, @10 (nDCG1,
nDCG3, nDCG10 account for relevance and position of retrieved items).
slightly from 0.73 (Large) to 0.78 (Small), and faithfulness
reaches its peak at 0.93 for Small. Latency shows minimal
variation across sizes, with a mean latency of 3.2ns and p95
latency between 4.65–4.69ns, indicating consistent retrieval
speed regardless of scale. Cost metrics are virtually identical,
with mean cost fixed at 1.2e −4 and slight variations in
standard deviation. Token usage also remains stable, though
smaller datasets exhibit slightly lower prompt token means
(546 vs. 558) and total token counts (612 vs. 625).
As illustrated in Fig. 4, comparison across similarity
metrics (Cosine, Inner Product, and Euclidean L2) reveals
more noticeable differences. Cosine and IP consistently
deliver higher coverage (0.81–0.82) and correctness ( 0.79–
0.80) compared to L2, which lags with values around 0.60 for
coverage and 0.67 for correctness. Faithfulness remains
strong and uniform ( 0.91–0.93), while relevance is highest
for Cosine and IP (0.87) compared to L2 (0.78). Latency
again shows only minor variation, with L2 achieving the
lowest mean latency (3.07 ns) and p95 latency (4.51 ns),
followed by Cosine and IP . Costs and token usage are nearly
indistinguishable across metrics, reinforcing that the primary
differences lie in retrieval quality rather than computational
efficiency. Thus, Cosine and IP emerge as stronger options
for accuracy oriented applications, while L2 offers slightly
faster retrieval at the cost of reduced quality.
As shown in Fig.5, the comparison of Fusion, Hierarchical,
and HyDe retrievers reveals clear trade-offs across perfor-
mance, latency, cost, and token usage. HyDe consistently
delivers the best retrieval quality, achieving the highest
coverage (0.86), correctness, faithfulness, and relevance
(0.89), but this comes at the expense of efficiency, with the
slowest mean and p95 latencies (4.84ns and 6.93ns), the
highest cost (1.7e-04, nearly double Fusion and Hierarchical),
and the most significant token consumption (710 total tokens
215740 VOLUME 13, 2025

## Page 18

H. Elkiran, J. Rasheed: EvaRAG: Evaluating Advanced RAG Techniques
FIGURE 2. Comparative performance of HNSW, IVF, and SCaNN across key metrics including retrieval quality, latency, cost, and token efficiency.
FIGURE 3. Comparative performance on Small, Medium, and Large size datasets across key metrics including retrieval quality, latency, cost, and token
efficiency.
on average). In contrast, Hierarchical is the fastest and
most resource-efficient, with a mean latency of 2.21 ns,
a p95 latency of 3.11 ns, and lower cost and token usage
(571 total tokens). However, its accuracy and coverage lag
(0.65–0.78). Fusion occupies a middle ground, balancing
accuracy, speed, and efficiency with moderate values across
all metrics. Retriever choice depends on application needs:
HyDe for accuracy-critical tasks, Hierarchical for latency-
sensitive scenarios, and Fusion for balanced performance.
Fig. 6 compares the rerankers BGE and minilm, showing
only minor differences in performance, latency, cost, and
token usage. Both models achieve identical scores in
VOLUME 13, 2025 215741

## Page 19

H. Elkiran, J. Rasheed: EvaRAG: Evaluating Advanced RAG Techniques
FIGURE 4. Comparative performance of Cosine, Inner Product and L2 metrics across key metrics including retrieval quality, latency, cost, and token
efficiency.
FIGURE 5. Comparative performance of Fusion, Hierarchical, and HyDe metrics across key metrics including retrieval quality, latency, cost, and token
efficiency.
Coverage Retrieval and Coverage Reranked (0.74). At the
same time, minilm performs slightly better in Correctness
(0.78 vs. 0.73) and Faithfulness (0.93 vs. 0.92), and BGE
shows a marginally higher Relevance score (0.86 vs. 0.82),
suggesting near-equivalent retrieval quality. Latency provides
the most evident distinction, with minilm being faster (mean
3.07s, p95 4.61 ns) than BGE (mean 3.35 ns, p95 4.73 ns),
making minilm more suitable for speed-sensitive applica-
tions; however, the gap is modest compared to retrievers.
Cost and token metrics reveal almost no difference, with both
rerankers averaging a mean cost of 1.2e-04 and similar token
usage (616 for BGE vs. 622 for minilm). Overall, reranker
215742 VOLUME 13, 2025

## Page 20

H. Elkiran, J. Rasheed: EvaRAG: Evaluating Advanced RAG Techniques
TABLE 21. Reranking performance of SCaNN with Euclidean Distance (L2) different retrievers, rerankers, and dataset sizes. Metrics include Recall@1, @3,
@10 (R1, R3 and R10, fraction of queries with correct item in top-K), MRR (average reciprocal rank of first correct result), and nDCG@1, @3, @10 (nDCG1,
nDCG3, nDCG10 account for relevance and position of retrieved items).
FIGURE 6. Comparative performance of BGE and minilm metrics across key metrics, including retrieval quality, latency, cost, and token efficiency.
selection has minimal impact on performance or resource
efficiency, with only slight advantages depending on whether
accuracy (minilm) or relevance (BGE) is prioritized.
Table 22 summarizes the performance, efficiency, and
cost-related metrics across all retrieval and reranking config-
urations. Coverage metrics show a mean of 0.743 with high
variability, with maximum coverage achieved using HNSW,
IP , Fusion, and BGE on the Small dataset, and minimum
coverage with SCANN, l2, Fusion, and BGE on the Small
dataset. Correctness, Faithfulness, and Relevance exhibit
high mean values (0.752–0.924), with Faithfulness being
the most consistent. Maximum performance is generally
observed with HNSW and minilm-based rerankers. Latency
varies notably, with a mean latency of 3.212 ns and a P95 of
4.669 ns, peaking for HNSW, IP , Fusion, minilm, and Small
combinations, reflecting the trade-off between accuracy and
computational cost. Cost and token usage remain generally
low and efficient, although the total number of tokens ranges
from 553 to 725, depending on the configuration. Overall,
the Table highlights the trade-offs between retrieval accuracy,
consistency, latency, and computational efficiency across
different system settings.
The Table 22 shows that as retrieval datasets grow from
small to medium, retrieval quality, measured by coverage,
VOLUME 13, 2025 215743

## Page 21

H. Elkiran, J. Rasheed: EvaRAG: Evaluating Advanced RAG Techniques
TABLE 22. Summary statistics for all metrics, including mean, std, max/min values and their corresponding setting combinations.
correctness, and faithfulness, improves due to greater doc-
ument diversity and a higher likelihood of finding relevant
information. However, these benefits largely plateau from
medium to large datasets, suggesting a saturation point where
adding more documents introduces more noise than signal,
especially in SCANN and L2-based settings, which show the
weakest scalability.
As shown in Table 23, the best-performing parameter
combinations vary across retrieval, reranking, correctness,
efficiency, and token-related metrics. The table systemati-
cally compares different retrievers (HNSW, IVF, SCANN),
similarity functions (IP , Cosine, L2), reranking strategies
(Fusion, Hierarchical, HyDe), and rerankers (BGE, minilm).
These results highlight clear performance trade-offs, where
specific configurations excel in retrieval precision, while
others optimize computational efficiency or semantic quality.
D. GUIDING PRINCIPLES
For retrieval-level metrics, HNSW combined with IP simi-
larity and Fusion reranking shows strong performance, par-
ticularly with the BGE reranker. It achieves the highest R@3
(0.887), R@10 (0.942), and Coverage (0.942), demonstrating
its effectiveness in deeper recall. Meanwhile, IVF with IP–
Fusion–BGE yields the strongest R@1 and nDCG@1 scores
(both 0.752), suggesting that IVF may be more precise in
top-ranked retrieval compared to HNSW. This distinction
underlines the importance of retriever choice depending
on whether the task prioritizes breadth (coverage) or top
precision.
At the reranked stage, the influence of the reranker
becomes more pronounced. SCANN with IP–Fusion–minilm
delivers the best R@1 reranked (0.828) and nDCG@1
reranked (0.828), while HNSW with minilm achieves the
strongest R@3 reranked (0.913), MRR reranked (0.872),
and nDCG@10 reranked (0.889). Notably, the correctness
(0.909), faithfulness (0.970), and relevance (0.959) metrics
peak under HNSW–IP–Fusion with minilm, indicating that
reranking with minilm substantially enhances semantic
accuracy and trustworthiness beyond raw retrieval.
For efficiency related metrics, IVF in combination with
hierarchical reranking and minilm provides the lowest
latencies, reaching a mean of 1.736s and a p95 latency
of 2.574s. Cost metrics also favor lightweight config-
urations: IVF–Cosine–Hierarchical–BGE minimizes aver-
age cost (0.000096), while HNSW–Cosine–Fusion–minilm
reduces cost variance (0.000023). These results suggest that
IVF configurations are preferable in latency-sensitive or
resource-constrained scenarios.
Finally, token-level metrics in Table 23 show dis-
tinct patterns: SCANN–Cosine–Hierarchical–BGE maxi-
mizes prompt (523.263) and total tokens (552.645), whereas
SCANN–l2–Fusion–BGE yields the highest completion
token mean (26.977). V ariance measures indicate stabil-
ity differences, with SCANN–IP–Fusion–minilm minimiz-
ing completion token variability (18.263). These results
215744 VOLUME 13, 2025

## Page 22

H. Elkiran, J. Rasheed: EvaRAG: Evaluating Advanced RAG Techniques
TABLE 23. Best performing parameter combinations for each metric.
demonstrate that token usage is significantly influenced by
the retriever–reranker choice and may reflect the linguistic
richness of the retrieved content.
VI. LIMITATION
Although the EV ARAG framework provides a systematic and
comprehensive framework for benchmarking RAG systems,
several limitations remain. The experimental evaluation was
conducted on a restricted set of QA dataset (SquAD), which
doesn’t fully capture the diversity, particularly in specialized
domains such as medicine, law, or finance. Similarly, while
our metrics focused on retrieval accuracy and generative
quality, broader aspects such as factual consistency, bias,
and robustness were not extensively measured. Another
important consideration is that although chunking strategies
were considered, adaptive or dynamic chunking approaches
that could significantly affect retrieval quality were not fully
incorporated. Also, our system does not account for the case
where feature boundaries are blurred, which typically occurs
under high semantic density. This overlap makes it difficult
for the model to clearly separate features, leading to reduced
precision in downstream tasks.
VII. FUTURE WORK
Future research will extend EV ARAG in several promising
directions. One natural step is to evaluate RAG pipelines
in domain-specific contexts, such as healthcare or legal
applications, to assess the robustness of retriever-similarity-
metric combinations under domain constraints. Another
avenue lies in the design of hybrid similarity functions
that adaptively combine cosine, inner product, and L2
distances, thereby balancing semantic precision with recall.
Beyond performance, future extensions of EV ARAG could
incorporate metrics that explicitly measure factual grounding
and hallucination reduction, providing a more holistic assess-
ment of RAG systems. Additionally, a potential direction
for future improvement is to integrate techniques such as
context-aware feature disentanglement and density-adaptive
representations, which can be adapted to handle overlapping
feature boundaries. While our study primarily focuses on
retrieval metrics, it does not explicitly examine how factors
such as semantic sparsity and the distribution of embedding
vectors affect retrieval efficiency. Future research will also
incorporate causal analysis to investigate the impact of these
factors on metric outcomes, thus improving theoretical and
practical understanding of RAG systems.
VIII. CONCLUSION AND DISCUSSION
This study presented a comprehensive evaluation of RAG
pipelines, examining the interplay between retrievers, sim-
ilarity metrics, indexing strategies, rerankers, and dataset
sizes across a wide range of performance and efficiency
metrics. Our results demonstrate that component choice
has a substantial impact on retrieval quality, semantic
accuracy, and computational efficiency, often revealing
trade-offs among these dimensions. In terms of retrieval
performance, the combination of HNSW indexing with inner
product (IP) similarity, Fusion reranking, and the BGE
VOLUME 13, 2025 215745

## Page 23

H. Elkiran, J. Rasheed: EvaRAG: Evaluating Advanced RAG Techniques
reranker consistently delivered top-tier results on the Small
dataset, achieving a maximum Coverage Retrieval of 0.942,
R@10 of 0.942, and MRR of 0.823. When reranking is
applied, the HNSW–IP–Fusion–minilm configuration further
improved semantic quality, reaching the highest Correctness
(0.909), Faithfulness (0.970), and Relevance (0.959) scores,
underscoring the effectiveness of cross-encoder rerankers
for semantic refinement. The effectiveness of HNSW–IP–
Fusion–MiniLM stems from combining HNSW’s efficient
nearest neighbor search, Inner Product similarity’s alignment
with embedding semantics, Fusion’s robust evidence aggre-
gation, and MiniLM’s lightweight, context-aware reranking.
HyDe consistently outperforms Fusion when paired with L2,
as its generative query expansion yields embeddings with
more stable magnitudes, allowing L2 to serve as a reliable
discriminator. In contrast, Fusion’s raw retrieval signals lead
to greater variance in embedding norms, undermining L2
performance. From an efficiency standpoint, configurations
built on IVF indexing with L2 similarity and Hierarchical
retrievers proved to be the most resource efficient, with
a mean latency as low as 1.736 ns and a p95 latency of
2.574 ns, making them ideal for latency-sensitive applica-
tions. Cost analysis showed that IVF–Cosine–Hierarchical–
BGE minimized computational expense (9.6 × 10−5) while
HNSW–Cosine–Fusion–minilm minimized cost variability
(2.3 × 10−5). Analysis of similarity metrics revealed that
the Cosine and Inner Product functions outperform L2
in retrieval quality, with coverage ranging from 0.81 to
0.82, compared to 0.60 for L2, although L2 retained a
slight latency advantage. Furthermore, the HyDe retriever
demonstrated superior retrieval quality (Coverage 0.86,
Relevance 0.89) at the cost of higher latency (4.84 ns)
and token usage (710 tokens). In comparison, Hierarchical
retrievers achieved the lowest latency (2.21 ns) and cost,
highlighting the fundamental trade-off between precision
and efficiency. Finally, reranker analysis showed marginal
differences: minilm slightly outperformed BGE in correct-
ness (0.78 vs. 0.73) and faithfulness (0.93 vs. 0.92), while
maintaining lower latency (3.07 ns vs. 3.35 ns). This suggests
that reranker choice has a modest but measurable impact.
To summarize, our findings show that no single configuration
universally dominates; instead, optimal design depends on
task requirements. HNSW–IP–Fusion–minilm is ideal for
accuracy-critical applications, IVF–L2–Hierarchical–minilm
excels in latency-sensitive environments, and SCaNN offers
a balanced trade-off with the lowest average latency
(3.05 ns) and competitive performance. Beyond empirical
performance, EV ARAG reveals a deeper guiding principle
for RAG design. Retrieval effectiveness emerges not from
any single component but from the interaction between
retriever similarity geometry, indexing structure, and reranker
semantic refinement. The study shows that retrieval pipelines
function as tightly coupled systems in which early-stage
vector search determines the coarse semantic neighbourhood,
while rerankers selectively sharpen this signal by resolving
fine-grained meaning. This layered interaction explains why
certain combinations, such as HNSW–IP–Fusion–MiniLM,
consistently outperform alternatives: they align vector-space
geometry with evidence aggregation and lightweight seman-
tic filtering. Similarly, the trade-offs observed across IVF,
L2, and hierarchical configurations illustrate that efficiency
and precision are governed by the structural properties
of the embedding space rather than by model size alone.
Taken together, EV ARAG demonstrates that optimal RAG
performance depends on achieving alignment across these
layers: indexing, similarity, retriever type, and reranking
rather than merely improving any single module in isolation.
ACKNOWLEDGMENT
Computing resources used in this work were provided by
the National Center for High Performance Computing of
Türkiye (UHeM). This study was part of a thesis titled
‘‘Performance Analysis of Advanced Retrieval-Augmented
Generation Applications using V ector Databases, Indexing
Algorithms and Distance Metrics’’ under the supervision of
Jawad Rasheed (Cevat Resit).
CONFLICT OF INTEREST/COMPETING INTERESTS
The authors declare that they have no known competing
financial interests or personal relationships that could have
appeared to influence the work reported in this article.
ETHICS DECLARATION
Not applicable.
CLINICAL TRIAL NUMBER
Not applicable.
CONSENT TO PARTICIPATE DECLARATION
Not applicable.
CONSENT FOR PUBLICATION
Not applicable.
DATA AVAILABILITY
The data supporting the findings of this study can
be obtained from the author (Harun Elkiran, email:
harun.elkiran@izu.edu.tr) upon reasonable request.
REFERENCES
[1] J. Achiam et al., ‘‘GPT-4 technical report,’’ 2023, arXiv:2303.08774.
[2] H. Touvron, T. Lavril, G. Izacard, X. Martinet, M.-A. Lachaux, T. Lacroix,
B. Rozière, N. Goyal, E. Hambro, F. Azhar, A. Rodriguez, A. Joulin,
E. Grave, and G. Lample, ‘‘LLaMA: Open and efficient foundation
language models,’’ 2023, arXiv:2302.13971.
[3] D. Demszky, D. Y ang, D. S. Y eager, C. J. Bryan, M. Clapper, S. Chandhok,
J. C. Eichstaedt, C. A. Hecht, J. P . Jamieson, M. Johnson, M. Jones,
D. Krettek-Cobb, L. C. Lai, N. JonesMitchell, D. C. Ong, C. S. Dweck,
J. J. Gross, and J. W. Pennebaker, ‘‘Using large language models in
psychology,’’Nature Rev. Psychol., vol. 2, no. 11, pp. 688–701, 2023.
[4] H. Nori, N. King, S. M. McKinney, D. Carignan, and E. Horvitz, ‘‘Capabil-
ities of GPT-4 on medical challenge problems,’’ 2023, arXiv:2303.13375.
[5] M. Arslan, L. Mahdjoubi, and S. Munawar, ‘‘Driving sustainable energy
transitions with a multi-source RAG-LLM system,’’ Energy Buildings,
vol. 324, Dec. 2024, Art. no. 114827.
215746 VOLUME 13, 2025

## Page 24

H. Elkiran, J. Rasheed: EvaRAG: Evaluating Advanced RAG Techniques
[6] M. Arslan, H. Ghanem, S. Munawar, and C. Cruz, ‘‘A survey on RAG with
LLMs,’’ Proc. Comput. Sci., vol. 246, pp. 3781–3790, Mar. 2024.
[7] N. Kandpal, H. Deng, A. Roberts, E. Wallace, and C. Raffel, ‘‘Large
language models struggle to learn long-tail knowledge,’’ inProc. Int. Conf.
Mach. Learn., 2022, pp. 15696–15707.
[8] K. Sun, Y . E. Xu, H. Zha, Y . Liu, and X. L. Dong, ‘‘Head-to-tail: How
knowledgeable are large language models (LLMs)? A.K.A. will LLMs
replace knowledge graphs?’’ 2023, arXiv:2308.10168.
[9] J. Li, X. Cheng, W. X. Zhao, J.-Y . Nie, and J.-R. Wen, ‘‘HaluEval: A large-
scale hallucination evaluation benchmark for large language models,’’
2023, arXiv:2305.11747.
[10] V . Rawte, S. Chakraborty, A. Pathak, A. Sarkar, S. M. T. I. Tonmoy,
A. Chadha, A. Sheth, and A. Das, ‘‘The troubling emergence of
hallucination in large language models-an extensive definition, quantifi-
cation, and prescriptive remediations,’’ in Proc. 2023 Conf. Empirical
Methods Natural Lang. Process., H. Bouamor, J. Pino, and K. Bali,
Eds., Singapore: Association for Computational Linguistics, Dec. 2023,
pp. 2541–2573, doi: 10.18653/v1/2023.emnlp-main.155.
[11] A. Asai, Z. Wu, Y . Wang, A. Sil, and H. Hajishirzi, ‘‘Self-RAG:
Learning to retrieve, generate, and critique through self-reflection,’’ 2023,
arXiv:2310.11511.
[12] P . Lewis, E. Perez, A. Piktus, F. Petroni, V . Karpukhin, N. Goyal, H. Küttler,
M. Lewis, W.-T. Yih, T. Rocktäschel, S. Riedel, and D. Kiela, ‘‘Retrieval-
augmented generation for knowledge-intensive NLP tasks,’’ in Proc. Adv.
Neural Inf. Process. Syst., vol. 33, 2020, pp. 9459–9474.
[13] O. Ram, Y . Levine, I. Dalmedigos, D. Muhlgay, A. Shashua, K. Leyton-
Brown, and Y . Shoham, ‘‘In-context retrieval-augmented language
models,’’ Trans. Assoc. Comput. Linguistics, vol. 11, pp. 1316–1331,
May 2023.
[14] W. Sarah, ‘‘Boosting rag performance: A comparative study of scann
and traditional vector search in large language model pipelines,’’
Apr. 2025. [Online]. Available: https://www.researchgate.net/publication/
391645756_Boosting_RAG_Performance_A_Comparative_Study_of_
ScaNN_and_Traditional_V ector_Search_in_Large_Language_Model_
Pipelines
[15] A. Abdallah, B. Piryani, J. Mozafari, M. Ali, and A. Jatowt, ‘‘Rankify:
A comprehensive Python toolkit for retrieval, re-ranking, and retrieval-
augmented generation,’’ 2025, arXiv:2502.02464.
[16] S. Deshmukh and A. Bajaj, ‘‘CareerBoost: A hybrid RAG-NLP job
recommendation framework,’’ in Proc. 8th Int. Conf. I-SMAC (IoT Social,
Mobile, Anal. Cloud) (I-SMAC), Oct. 2024, pp. 853–858.
[17] D. Mozolevskyi and W. AlShikh, ‘‘Comparative analysis of retrieval
systems in the real world,’’ 2024, arXiv:2405.02048.
[18] J. Kim and D. Mahajan, ‘‘V ectorLiteRAG: Latency-aware and fine-grained
resource partitioning for efficient RAG,’’ 2025, arXiv:2504.08930.
[19] D. Tanyildiz, S. Ayvaz, and M. F. Amasyali, ‘‘Enhancing retrieval-
augmented generation accuracy with dynamic chunking and optimized
vector search,’’ Orclever Proc. Res. Develop., vol. 5, no. 1, pp. 215–225,
Dec. 2024.
[20] H. Brådland, M. Goodwin, P .-A. Andersen, A. S. Nossum, and A. Gupta,
‘‘A new HOPE: Domain-agnostic automatic evaluation of text chunking,’’
in Proc. 48th Int. ACM SIGIR Conf. Res. Develop. Inf. Retr ., Jul. 2025,
pp. 170–179.
[21] Y . Ateş, A. Sayar, İ. U. Bozlar, S. Ertuğrul, and S. S. Arslan, ‘‘Semantic
chunking and chain-of-thought reasoning for rag-based document process-
ing,’’ in Proc. IEEE 35th Int. Workshop Mach. Learn. Signal Process.
(MLSP), May 2025, pp. 1–6.
[22] C.-Y . Chang, Z. Jiang, V . Rakesh, M. Pan, C.-C.-M. Y eh, G. Wang,
M. Hu, Z. Xu, Y . Zheng, M. Das, and N. Zou, ‘‘MAIN-RAG: Multi-agent
filtering retrieval-augmented generation,’’ in Proc. 63rd Annu. Meeting
Assoc. Comput. Linguistics, 2025, pp. 2607–2622.
[23] J. Nian, Z. Peng, Q. Wang, and Y . Fang, ‘‘W-RAG: Weakly supervised
dense retrieval in RAG for open-domain question answering,’’ inProc. Int.
ACM SIGIR Conf. Innov. Concepts Theories Inf. Retr . (ICTIR), Jul. 2025,
pp. 136–146.
[24] J. Hu, Y . Zhou, and J. Wang, ‘‘Intrinsic evaluation of RAG systems for
deep-logic questions,’’ 2024, arXiv:2410.02932.
[25] W. Wang, J. Ma, P . Zhang, Z. Hu, Q. Jiang, and Y . Liu, ‘‘Application of
multi-way recall fusion reranking based on tensor and ColBERT in RAG,’’
in Proc. IEEE 7th Int. Conf. Inf. Syst. Comput. Aided Educ. (ICISCAE),
Sep. 2024, pp. 138–141.
[26] A. K. Shahade and P . V . Deshmukh, ‘‘Enhancing natural language
processing: A comprehensive review of retrieval augmented generation,’’
in Proc. 4th Int. Conf. Sustain. Expert Syst. (ICSES), Oct. 2024,
pp. 609–611.
[27] A. Leto, C. Aguerrebere, I. Bhati, T. Willke, M. Tepper, and V . A. V o,
‘‘Toward optimal search and retrieval for RAG,’’ 2024,arXiv:2411.07396.
[28] H. Sun, Y . Wang, and S. Zhang, ‘‘Retrieval-augmented generation for
domain-specific question answering: A case study on Pittsburgh and
CMU,’’ 2024, arXiv:2411.13691.
[29] J. Dong, B. Fatemi, B. Perozzi, L. F. Y ang, and A. Tsitsulin, ‘‘Don’t
forget to connect! Improving RAG with graph-based reranking,’’ 2024,
arXiv:2405.18414.
[30] I. Papadimitriou, I. Gialampoukidis, S. Vrochidis, Ioannis, and Kom-
patsiaris, ‘‘RAG playground: A framework for systematic evaluation of
retrieval strategies and prompt engineering in RAG systems,’’ 2024,
arXiv:2412.12322.
[31] P . Rajpurkar, J. Zhang, K. Lopyrev, and P . Liang, ‘‘SQuAD: 100,000+
questions for machine comprehension of text,’’ 2016, arXiv:1606.05250.
HARUN ELKIRAN (Member, IEEE) received the
M.S. degree in computer science and engineering
from Istanbul Sabahattin Zaim University, Istan-
bul, Türkiye, where he is currently pursuing the
Ph.D. degree in computer science and engineering,
under the supervision of Dr. Jawad Rasheed.
His research interests include RAG, LLM, deep
learning, and database systems and management.
JAWAD RASHEED (Member, IEEE) received
the B.S. degree in telecommunication engineering
from the National University of Computer and
Emerging Sciences, Pakistan, and the M.S. degree
in electrical and electronics engineering and the
Ph.D. degree in computer science and engineering
from Türkiye.
He is currently an Associate Professor with the
Department of Computer Engineering, Istanbul
Sabahattin Zaim University, Türkiye. He is also
a Senior Researcher with Istanbul Medipol University, Türkiye; and a
Research Fellow with Applied Science Private University, Jordan. He is
the author/co-author of over 80 articles published in well-reputed journals
and highly-ranked conferences. His research interests include artificial
intelligence and image processing, pattern recognition, the IoT, blockchain,
and data analytics. He was a Gold Medalist. He received the Academic
Excellence Award for securing straight A ’s in O’ Level exams held by
Cambridge University. Later, he also received a prestigious Doctorate and
Research Scholarship for his Ph.D. studies (for three years). He serves as
an editor/guest-editor at several reputed journals, including BMC Infectious
Disease, PLOS One, International Journal of Computational Intelligence
Systems, Discover Artificial Intelligence, and International Journal of
Intelligent Transportation Systems Research.
VOLUME 13, 2025 215747
