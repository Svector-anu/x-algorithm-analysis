# STEP 1: Repository Mapping

## Overview

The X Algorithm repository contains the core recommendation system powering the "For You" feed on X (formerly Twitter). It combines in-network content (from accounts you follow) with out-of-network content (discovered through ML-based retrieval) and ranks everything using a Grok-based transformer model.

**Repository:** https://github.com/xai-org/x-algorithm

---

## Directory Structure

```
x-algorithm/
├── candidate-pipeline/      # Core: Candidate sourcing framework
├── home-mixer/             # Core: Feed orchestration service
├── phoenix/                # Core: ML models (ranking + retrieval)
├── thunder/                # Support: Real-time data ingestion
├── README.md
├── LICENSE
└── CODE_OF_CONDUCT.md
```

---

## Component Classification

### Core Components

| Component | Type | Role |
|-----------|------|------|
| **`phoenix/`** | **CORE (ML)** | ML brain - transformer models for ranking and retrieval |
| **`home-mixer/`** | **CORE (Service)** | Orchestration engine - assembles final feed |
| **`candidate-pipeline/`** | **CORE (Framework)** | Pipeline framework - defines sourcing/scoring logic |

### Support Components

| Component | Type | Role |
|-----------|------|------|
| **`thunder/`** | **SUPPORT (Infrastructure)** | Real-time event streaming and in-memory post store |

---

## File Breakdown by Component

### 1. Phoenix (Python - ML Models)

**Location:** `phoenix/`

**Key Files:**
- `grok.py` - Grok transformer implementation (ported from Grok-1)
- `recsys_model.py` - Main ranking model architecture
- `recsys_retrieval_model.py` - Two-tower retrieval model
- `run_ranker.py` - Ranking inference script
- `run_retrieval.py` - Retrieval inference script
- `runners.py` - Training/inference orchestration
- `README.md` - Detailed architecture documentation

**Role:** The "brain" of the system. Implements transformer-based ranking and retrieval models that decide what content matches a user.

---

### 2. Home Mixer (Rust - Service Layer)

**Location:** `home-mixer/`

**Key Subdirectories:**
- `candidate_hydrators/` - Enrich candidate metadata
- `filters/` - Remove unwanted posts
- `scorers/` - Apply ML predictions + weighting
- `selectors/` - Pick top-K posts
- `sources/` - Thunder + Phoenix integration
- `query_hydrators/` - Process user context

**Key Files:**
- `main.rs` - Service entry point
- `server.rs` - gRPC API server

**Role:** The "engine room." Orchestrates the entire process from fetching candidates to producing the final ranked list for the user's UI.

---

### 3. Candidate Pipeline (Rust - Framework)

**Location:** `candidate-pipeline/`

**Key Files:**
- `candidate_pipeline.rs` - Pipeline orchestrator
- `source.rs` - Candidate source trait
- `hydrator.rs` - Data enrichment trait
- `filter.rs` - Filtering trait
- `scorer.rs` - Scoring trait
- `selector.rs` - Selection trait
- `side_effect.rs` - Side effects (logging, tracking)

**Role:** The "sourcing layer." Defines how different types of content (following vs. out-of-network) are initially gathered and processed.

---

### 4. Thunder (Rust - Streaming)

**Location:** `thunder/`

**Key Subdirectories:**
- `kafka/` - Kafka consumer integration
- `posts/` - Post data structures

**Key Files:**
- `main.rs` - Service entry point
- `thunder_service.rs` - In-memory post store
- `kafka_utils.rs` - Kafka connection management
- `deserializer.rs` - Parsing incoming data formats

**Role:** The "nervous system." Handles high-speed data streams (engagement signals like likes/retweets) used to update recommendations in real-time.

---

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         FOR YOU FEED REQUEST                        │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                           HOME MIXER                                │
│                      (Orchestration Layer)                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ QUERY HYDRATION                                              │  │
│  │ • User Action Sequence (engagement history)                  │  │
│  │ • User Features (following list, preferences)                │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ CANDIDATE SOURCES                                            │  │
│  │ ┌─────────────────────┐  ┌──────────────────────────────┐   │  │
│  │ │ THUNDER             │  │ PHOENIX RETRIEVAL            │   │  │
│  │ │ (In-Network Posts)  │  │ (Out-of-Network Posts)       │   │  │
│  │ │                     │  │                              │   │  │
│  │ │ Posts from accounts │  │ ML-based similarity search   │   │  │
│  │ │ you follow          │  │ across global corpus         │   │  │
│  │ └─────────────────────┘  └──────────────────────────────┘   │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ HYDRATION                                                    │  │
│  │ Fetch: post metadata, author info, media entities            │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ FILTERING                                                    │  │
│  │ Remove: duplicates, old posts, blocked authors, etc.         │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ SCORING                                                      │  │
│  │ • Phoenix Scorer (ML predictions)                            │  │
│  │ • Weighted Scorer (combine predictions)                      │  │
│  │ • Author Diversity Scorer (ensure variety)                   │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ SELECTION                                                    │  │
│  │ Sort by score, select top K candidates                       │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       RANKED FEED RESPONSE                          │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Key Technologies

- **ML Framework:** JAX + Haiku (Python)
- **Service Layer:** Rust
- **Streaming:** Kafka
- **Model Architecture:** Grok-based Transformer (ported from Grok-1)
- **API:** gRPC

---

## Quick Reference

### What Each Component Does

1. **Thunder** → Streams real-time posts from Kafka → In-memory store
2. **Phoenix Retrieval** → Finds similar posts from global corpus → Top-K candidates
3. **Home Mixer** → Combines sources → Filters → Scores → Ranks → Returns feed
4. **Phoenix Ranking** → Predicts engagement probabilities → Used by scorers

### Data Flow

```
User Request → Home Mixer → Query Hydration → Candidate Sourcing 
→ (Thunder + Phoenix Retrieval) → Filtering → Scoring → Selection → Response
```

---

## Next Steps

- **STEP 2:** Analyze core algorithm files (Phoenix ML models)
- **STEP 3:** Understand engagement logic and feedback loops
- **STEP 4:** Examine growth and distribution mechanisms
- **STEP 5:** Build mental model of the entire system
