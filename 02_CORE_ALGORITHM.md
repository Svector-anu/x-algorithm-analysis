# STEP 2: Core Algorithm Files

## Overview

This document analyzes the core Phoenix ML model files that implement the ranking and retrieval algorithms. The Phoenix system uses a Grok-based transformer architecture to predict user engagement with posts.

---

## Core Files Analyzed

1. **`phoenix/recsys_model.py`** - Main ranking transformer
2. **`phoenix/recsys_retrieval_model.py`** - Two-tower retrieval model  
3. **`phoenix/grok.py`** - Transformer architecture
4. **`phoenix/run_ranker.py`** - Ranking execution script
5. **`phoenix/runners.py`** - Inference orchestration

---

## Input Data Structures

### RecsysBatch (Feature Data)

```python
RecsysBatch:
    user_hashes: [B, num_user_hashes]              # User identity hashes
    history_post_hashes: [B, S, num_item_hashes]   # Posts user engaged with
    history_author_hashes: [B, S, num_author_hashes] # Authors of history posts
    history_actions: [B, S, num_actions]           # Multi-hot engagement vectors
    history_product_surface: [B, S]                # Where engagement happened
    candidate_post_hashes: [B, C, num_item_hashes] # Posts to rank
    candidate_author_hashes: [B, C, num_author_hashes] # Candidate authors
    candidate_product_surface: [B, C]              # Candidate surfaces
```

**Dimensions:**
- `B` = Batch size (typically 1)
- `S` = History sequence length (32 in demo, 128 max)
- `C` = Candidates to rank (8 in demo, 32 max)
- `num_actions` = 19 engagement types
- `emb_size` = 128 (embedding dimension)

### RecsysEmbeddings (Pre-looked-up Embeddings)

```python
RecsysEmbeddings:
    user_embeddings: [B, num_user_hashes, D]
    history_post_embeddings: [B, S, num_item_hashes, D]
    candidate_post_embeddings: [B, C, num_item_hashes, D]
    history_author_embeddings: [B, S, num_author_hashes, D]
    candidate_author_embeddings: [B, C, num_author_hashes, D]
```

**Hash-Based Embeddings:**
- Uses **2 hash functions** per entity (user, post, author)
- Hash value `0` = padding/invalid
- Multiple hashes are concatenated then projected to single embedding

---

## Output Data Structures

### RankingOutput

```python
RankingOutput:
    scores: [B, C, 19]           # Probability for each of 19 actions
    ranked_indices: [B, C]       # Candidates sorted by favorite_score
    p_favorite_score: [B, C]     # Individual action probabilities
    p_reply_score: [B, C]
    p_repost_score: [B, C]
    # ... (19 total action probabilities)
```

**Primary Ranking Metric:**
```python
primary_scores = probs[:, :, 0]  # favorite_score
ranked_indices = argsort(-primary_scores, axis=-1)
```

Posts are **ranked by predicted "favorite" (like) probability**.

---

## The 19 Engagement Actions

```python
ACTIONS = [
    "favorite_score",           # 0 - PRIMARY RANKING SIGNAL ⭐
    "reply_score",              # 1
    "repost_score",             # 2
    "photo_expand_score",       # 3
    "click_score",              # 4
    "profile_click_score",      # 5
    "vqv_score",                # 6 - Video quality view
    "share_score",              # 7
    "share_via_dm_score",       # 8
    "share_via_copy_link_score",# 9
    "dwell_score",              # 10
    "quote_score",              # 11
    "quoted_click_score",       # 12
    "follow_author_score",      # 13
    "not_interested_score",     # 14 - NEGATIVE ⚠️
    "block_author_score",       # 15 - NEGATIVE ⚠️
    "mute_author_score",        # 16 - NEGATIVE ⚠️
    "report_score",             # 17 - NEGATIVE ⚠️
    "dwell_time",               # 18
]
```

---

## Model Architecture Flow

### Step 1: Embedding Combination

```python
# User embedding
user_embedding = project(concat(user_hashes_embeddings))  
# Shape: [B, 1, D]

# History embedding
history_embedding = project(concat([
    history_post_embeddings,
    history_author_embeddings,
    history_actions_embeddings,      # ← Learned from past actions
    history_product_surface_embeddings
]))  
# Shape: [B, S, D]

# Candidate embedding
candidate_embedding = project(concat([
    candidate_post_embeddings,
    candidate_author_embeddings,
    candidate_product_surface_embeddings
]))  
# Shape: [B, C, D]
```

### Step 2: Sequence Construction

```python
embeddings = concat([user_embedding, history_embedding, candidate_embedding])
# Shape: [B, 1 + S + C, D]
```

### Step 3: Transformer with Candidate Isolation

```python
# Special attention mask ensures:
# - User+history: causal attention
# - Candidates: can attend to user+history and SELF, but NOT other candidates
attn_mask = make_recsys_attn_mask(seq_len, candidate_start_offset)

model_output = Transformer(embeddings, attn_mask)
```

**Candidate Isolation Mechanism:**
```python
def make_recsys_attn_mask(seq_len, candidate_start_offset):
    # Candidates CANNOT attend to each other
    # This ensures score(post_i) is independent of other posts in batch
    # Makes scores CACHEABLE and CONSISTENT
    
    causal_mask = tril(ones(seq_len, seq_len))
    attn_mask = causal_mask
    attn_mask[candidate_start:, candidate_start:] = 0  # Block candidate-to-candidate
    attn_mask[diag(candidate_indices)] = 1  # Allow self-attention
    return attn_mask
```

### Step 4: Prediction Head

```python
candidate_embeddings = model_output[:, candidate_start_offset:, :]
logits = dot(layer_norm(candidate_embeddings), unembedding_matrix)
# Shape: [B, C, 19]
```

### Step 5: Probabilities

```python
probs = sigmoid(logits)  # [B, C, 19]
```

---

## Critical Code Snippets

### Action Embedding (How Past Engagement Affects Future)

```python
def _get_action_embeddings(self, actions: jax.Array) -> jax.Array:
    """Convert multi-hot action vectors to embeddings."""
    _, _, num_actions = actions.shape
    D = config.emb_size
    
    action_projection = get_parameter(
        "action_projection",
        [num_actions, D],  # 19 x 128
    )
    
    # Convert {0,1} to {-1,+1} for signed actions
    actions_signed = (2 * actions - 1).astype(jnp.float32)
    action_emb = dot(actions_signed, action_projection)
    
    return action_emb
```

**Key Insight:** Negative actions get **negative embeddings** (`-1`), creating repulsion in the embedding space.

### Ranking by Favorite Score

```python
def hk_rank_candidates(batch, recsys_embeddings):
    output = hk_forward(batch, recsys_embeddings)
    logits = output.logits
    probs = jax.nn.sigmoid(logits)
    
    # PRIMARY RANKING: Use favorite_score (index 0)
    primary_scores = probs[:, :, 0]
    ranked_indices = jnp.argsort(-primary_scores, axis=-1)
    
    return RankingOutput(scores=probs, ranked_indices=ranked_indices, ...)
```

### Weighted Scoring (from README)

```python
Final_Score = Σ (weight_i × P(action_i))

# Example weights (conceptual):
Final_Score = 
    2.0 × P(favorite) +
    1.5 × P(reply) +
    1.0 × P(repost) +
    0.5 × P(click) +
    (-3.0) × P(block) +      # Negative weight
    (-2.0) × P(mute) +       # Negative weight
    (-5.0) × P(report)       # Negative weight
```

---

## Retrieval Model (Two-Tower Architecture)

### User Tower

```python
def build_user_representation(batch, recsys_embeddings):
    # Encode user + history through transformer
    embeddings = concat([user_embeddings, history_embeddings])
    model_output = transformer(embeddings)
    
    # Mean pool over valid positions
    user_representation = mean_pool(model_output, padding_mask)
    
    # L2 normalize for dot product similarity
    user_representation = user_representation / norm(user_representation)
    
    return user_representation  # [B, D]
```

### Candidate Tower

```python
def build_candidate_representation(batch, recsys_embeddings):
    # Project post + author embeddings
    post_author_embedding = concat([post_embeddings, author_embeddings])
    
    # Two-layer MLP
    hidden = silu(dot(post_author_embedding, proj_1))
    candidate_representation = dot(hidden, proj_2)
    
    # L2 normalize
    candidate_representation = candidate_representation / norm(candidate_representation)
    
    return candidate_representation  # [N, D]
```

### Similarity Search

```python
def retrieve_top_k(user_representation, corpus_embeddings, top_k):
    # Dot product similarity (both are L2-normalized)
    scores = dot(user_representation, corpus_embeddings.T)  # [B, N]
    
    # Get top-k candidates
    top_k_scores, top_k_indices = top_k(scores, k=1000)
    
    return top_k_indices, top_k_scores
```

---

## Key Parameters

### Model Configuration

```python
emb_size = 128                # Embedding dimension
num_actions = 19              # Number of engagement actions
history_seq_len = 32          # Max history length (128 in production)
candidate_seq_len = 8         # Max candidates to rank (32 in production)

# Hash configuration
num_user_hashes = 2
num_item_hashes = 2
num_author_hashes = 2

# Transformer configuration
num_layers = 2                # Transformer depth
num_q_heads = 2               # Query attention heads
num_kv_heads = 2              # Key-value attention heads
key_size = 64                 # Attention key dimension
widening_factor = 2           # FFN expansion factor
```

---

## Engagement-Related Variables

### History Actions Encoding

```python
# Multi-hot action vector converted to embedding
actions_signed = (2 * actions - 1)  # {0,1} -> {-1,+1}
action_emb = dot(actions_signed, action_projection_matrix)
```

**This is how the model learns from past behavior:**
- If you liked similar posts → positive signal
- If you blocked similar authors → negative signal
- Actions are **signed** (-1 for negative, +1 for positive)

### Product Surface

```python
product_surface_vocab_size = 16
# Encodes WHERE the engagement happened:
# - Home timeline
# - Search results
# - Notifications
# - Profile view
# etc.
```

---

## Summary

The algorithm directly implements engagement prediction through:

1. **Multi-action prediction** (19 engagement types)
2. **Signed action embeddings** (negative actions create negative signals)
3. **Transformer-based context understanding** (user + history → candidate relevance)
4. **Candidate isolation** (scores are independent and cacheable)
5. **Two-tower retrieval** (efficient similarity search at scale)

**Core Insight:** Past engagement creates signed embeddings that flow through the transformer to amplify or suppress similar future content.
