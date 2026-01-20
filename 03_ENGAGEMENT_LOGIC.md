# STEP 3: Engagement Logic

## Definition of Engagement

In the X algorithm, **engagement** is defined as **any user interaction with a post**, tracked across **19 distinct action types**.

---

## The 19 Engagement Actions

### Positive Engagement Actions (increase post score)

1. `favorite_score` - Like/heart the post ❤️
2. `reply_score` - Reply to the post 💬
3. `repost_score` - Retweet/share the post 🔄
4. `quote_score` - Quote tweet 💭
5. `click_score` - Click on the post 👆
6. `profile_click_score` - Click on author's profile 👤
7. `photo_expand_score` - Expand images 🖼️
8. `vqv_score` - Video quality view (watch video) 📹
9. `share_score` - Share the post 📤
10. `share_via_dm_score` - Share via DM 💌
11. `share_via_copy_link_score` - Copy link 🔗
12. `dwell_score` - Dwell on post (time spent) ⏱️
13. `quoted_click_score` - Click on quoted tweet 🔍
14. `follow_author_score` - Follow the author ➕
15. `dwell_time` - Actual time spent (continuous value) ⏲️

### Negative Engagement Actions (decrease post score)

16. `not_interested_score` - Mark "not interested" 👎
17. `block_author_score` - Block the author 🚫
18. `mute_author_score` - Mute the author 🔇
19. `report_score` - Report the post ⚠️

---

## How Engagement is Created

### 1. Real-Time Event Capture (Thunder Service)

```
User Action → Kafka Event Stream → Thunder In-Memory Store
```

**Thunder's Role:**
- Consumes post create/delete events from Kafka
- Maintains per-user stores for recent posts
- Tracks engagement events as they happen
- Provides sub-millisecond lookups for in-network content

### 2. Engagement History Storage

```python
history_actions: [B, S, 19]  # Multi-hot vector per history item
```

**Example engagement vector:**
```python
# User liked and clicked a post, but didn't reply
history_actions[0, 5, :] = [
    1.0,  # favorite_score ✓
    0.0,  # reply_score
    0.0,  # repost_score
    0.0,  # photo_expand_score
    1.0,  # click_score ✓
    0.0,  # profile_click_score
    0.0,  # vqv_score
    0.0,  # share_score
    0.0,  # share_via_dm_score
    0.0,  # share_via_copy_link_score
    0.0,  # dwell_score
    0.0,  # quote_score
    0.0,  # quoted_click_score
    0.0,  # follow_author_score
    0.0,  # not_interested_score
    0.0,  # block_author_score
    0.0,  # mute_author_score
    0.0,  # report_score
    0.0,  # dwell_time
]
```

### 3. Engagement Encoding

```python
def _get_action_embeddings(actions):
    # Convert {0,1} multi-hot to {-1,+1} signed vector
    actions_signed = (2 * actions - 1)
    
    # Project to embedding space
    action_emb = dot(actions_signed, action_projection_matrix)
    
    return action_emb
```

**Critical Insight:** Negative actions get **negative embeddings** (`-1`), creating repulsion in the embedding space.

---

## How Engagement is Measured

### 1. Prediction Stage

The transformer outputs **probabilities for all 19 actions**:

```python
logits = transformer(user + history + candidates)  # [B, C, 19]
probs = sigmoid(logits)  # Convert to probabilities [0, 1]
```

### 2. Primary Ranking Metric

```python
# Posts are ranked by FAVORITE_SCORE (index 0)
primary_scores = probs[:, :, 0]
ranked_indices = argsort(-primary_scores)
```

**Why favorite_score?**
- Most common positive engagement
- Strong signal of content quality
- Correlates with other positive engagements

### 3. Weighted Combination (Home Mixer)

```
Final Score = Σ (weight_i × P(action_i))

Where:
  weight_favorite > 0      (e.g., +2.0)
  weight_reply > 0         (e.g., +1.5)
  weight_repost > 0        (e.g., +1.0)
  weight_block < 0         (e.g., -3.0)
  weight_mute < 0          (e.g., -2.0)
  weight_report < 0        (e.g., -5.0)
```

---

## How Engagement is Updated

### 1. Continuous Learning Loop

```
User sees post → User engages → Event logged → Model retraining → Updated predictions
```

### 2. Embedding Table Updates

```python
# Hash-based embeddings are updated during training
user_embeddings[user_hash] ← gradient_update
post_embeddings[post_hash] ← gradient_update
author_embeddings[author_hash] ← gradient_update
```

### 3. History Sequence Updates

```python
# As user engages, history grows
history_post_hashes = [post_1, post_2, ..., post_32]  # Last 32 interactions
history_actions = [actions_1, actions_2, ..., actions_32]
```

**Retention:**
- Thunder trims posts older than retention period
- History sequence has max length (32 in demo, 128 max)
- Older interactions are dropped (FIFO)

---

## Feedback Loops

### 1. Positive Feedback Loop (Engagement Amplification)

```
User likes post A
  ↓
Model learns: User → Post A embedding similarity ↑
  ↓
Similar posts (B, C, D) get higher scores
  ↓
User sees more similar content
  ↓
User engages more with similar content
  ↓
Similarity strengthens further
```

**Code Implementation:**
```python
# History embedding includes past actions
history_embedding = project(concat([
    post_embeddings,
    author_embeddings,
    action_embeddings,  # ← Past engagement creates bias
    product_surface_embeddings
]))

# Transformer learns: if user liked similar posts → boost candidate
candidate_score = transformer(user + history + candidate)
```

### 2. Negative Feedback Loop (Engagement Suppression)

```
User blocks author X
  ↓
action_embedding = -1 × action_projection[block_author_score]
  ↓
Negative signal flows through transformer
  ↓
Posts from similar authors get lower scores
  ↓
User sees less content from similar authors
```

**Code Implementation:**
```python
# Signed action embeddings create repulsion
actions_signed = (2 * actions - 1)  # {0,1} → {-1,+1}
action_emb = dot(actions_signed, action_projection)

# Block action (index 15) creates negative embedding
# This suppresses similar candidates in transformer attention
```

### 3. Diversity Feedback Loop (Author Diversity Scorer)

```
Author appears in feed
  ↓
Author Diversity Scorer attenuates repeated author scores
  ↓
Same author's next post gets lower score
  ↓
Feed shows different authors
```

**Purpose:** Prevent feed from being dominated by single author.

### 4. Recency Feedback Loop (Thunder Retention)

```
Post created → Thunder stores → Time passes → Post ages → Thunder trims
```

**Effect:** Old posts naturally drop out of candidate pool, ensuring freshness.

---

## How Past Engagement Affects Future Outcomes

### Mechanism 1: Embedding Space Clustering

```python
# User who liked posts [A, B, C] gets user_embedding positioned near them
user_representation = mean_pool(transformer(user + history))

# Candidates similar in embedding space get higher scores
similarity = dot(user_representation, candidate_embedding)
```

### Mechanism 2: Attention Patterns

```python
# Transformer learns attention patterns like:
# "If user liked tech posts + followed authors → attend to tech candidates"

attention_weights = softmax(
    dot(query, key) / sqrt(d_k)
)

# History with tech engagement → high attention to tech candidates
```

### Mechanism 3: Action-Specific Predictions

```python
# Model learns conditional probabilities:
# P(like | user liked similar posts) > P(like | user ignored similar posts)

# Training objective (implicit):
# Maximize: log P(observed_actions | user_history, candidate)
```

---

## Engagement Feedback Loop Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    ENGAGEMENT FEEDBACK LOOP                     │
└─────────────────────────────────────────────────────────────────┘

    User Sees Feed
         │
         ▼
    ┌─────────────────┐
    │ User Engages    │ ← Positive: like, reply, share
    │ with Post       │ ← Negative: block, mute, report
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ Event Logged    │
    │ (Kafka Stream)  │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ Thunder Updates │ ← In-memory post store
    │ History Store   │ ← User action sequence
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ Model Training  │ ← Embeddings updated
    │ (Offline)       │ ← Transformer weights updated
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ Updated Model   │
    │ Deployed        │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ Next Feed       │ ← Predictions reflect past engagement
    │ Request         │ ← Similar content boosted/suppressed
    └────────┬────────┘
             │
             └──────────────┐
                            │
                            ▼
                    User Sees Feed (loop continues)
```

---

## Engagement Persistence

### Short-Term (Seconds to Minutes)
- Thunder in-memory store
- Recent posts from followed accounts
- Real-time candidate availability

### Medium-Term (Hours to Days)
- User action sequence (last 32-128 interactions)
- Embedding table lookups
- Model predictions based on recent history

### Long-Term (Weeks to Months)
- Trained embedding tables
- Transformer weights
- Learned user preferences encoded in parameters

---

## Key Takeaways

1. **Engagement is a closed feedback loop** where past actions influence future content
2. **Signed embeddings** create attraction (positive actions) and repulsion (negative actions)
3. **Transformer learns patterns** from engagement history to predict future engagement
4. **Similar content gets amplified or suppressed** based on past behavior
5. **The algorithm wants users to engage more**, so it continuously learns what drives engagement

**The Core Truth:** Every engagement you make teaches the algorithm what to show you next. Like creates more similar content. Block creates less similar content. The system is always learning and adapting.
