# Database Agent — Architecture

A simple view of how a question flows through the system.

## High level

```mermaid
flowchart LR
    U[User] -->|POST /ask| API[FastAPI api.py]
    API --> MEM[Relevant Memory<br/>keeps only useful Q&A]
    API --> MAIN[Main Agent]
    MAIN --> SQL[SQL Agent]
    SQL --> DB[(Database<br/>pooled, read-only)]
    MAIN -.uses.-> LLM[Mistral LLM]
    SQL -.uses.-> LLM
    API --> API
```

## Main agent (orchestrator)

```mermaid
flowchart TD
    START([start]) --> DEC[question_decomposer<br/>split into sub-questions]
    DEC -->|one Send per sub-question| SQLA[sql_agent<br/>SQL subgraph, run per question]
    SQLA --> MERGE[merger<br/>summarize all answers]
    MERGE --> CRITIC{critic_route<br/>answer complete?}
    CRITIC -->|yes / revision cap hit| FINAL[final_result]
    CRITIC -->|no, needs more| DEC
    FINAL --> END([end])
```

## SQL agent (subgraph, runs for each sub-question)

```mermaid
flowchart TD
    S([start]) --> SCHEMA[database_details<br/>list tables]
    SCHEMA --> REQ[required_tables<br/>pick relevant tables]
    REQ --> DET[table_details<br/>describe columns]
    DET --> GEN[query_generator<br/>write SQL]
    GEN --> CR{critic<br/>SQL valid & safe?}
    CR -->|approved| EXE[excute_query<br/>read-only, validated]
    CR -->|rejected, retries left| GEN
    CR -->|retries exhausted| FMT[format_result]
    EXE -->|success| FMT
    EXE -->|failed, retries left| GEN
    EXE -->|retries exhausted| FMT
    FMT --> E([end])
```

## Key guarantees

- **Read-only:** every query is validated before it runs; the DB session is read-only and never committed.
- **Bounded:** SQL retries capped by `SQL_MAX_RETRIES`; the main critic loop capped by `MAX_REVISIONS`.
- **Relevant memory:** only meaningful answers are stored, and only overlapping ones are recalled.
- **Pooled + timed out:** connections come from a pool; LLM and SQL calls have timeouts; results are row-capped.
