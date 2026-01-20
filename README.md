# X Algorithm Analysis

> A comprehensive, step-by-step analysis of the X (Twitter) "For You" feed recommendation algorithm

## Overview

This repository contains a detailed analysis of the open-source X algorithm (https://github.com/xai-org/x-algorithm), broken down into 5 digestible documents. Each document can be read independently or as part of a complete understanding of how the recommendation system works.

**Analysis Date:** January 2026  
**Source Repository:** https://github.com/xai-org/x-algorithm  
**Model Architecture:** Grok-based Transformer (ported from Grok-1)

---

## 📚 Document Structure

### [01_REPO_MAPPING.md](./01_REPO_MAPPING.md)
**What it covers:** Repository structure, component classification, and system architecture overview

**Read this if you want to:**
- Understand the codebase organization
- Know which components are core vs support
- Get a high-level system architecture view
- Identify key files and their roles

**Key Takeaway:** The system has 3 core components (Phoenix ML, Home Mixer orchestration, Candidate Pipeline framework) and 1 support component (Thunder streaming).

---

### [02_CORE_ALGORITHM.md](./02_CORE_ALGORITHM.md)
**What it covers:** Phoenix ML model implementation, inputs, outputs, and scoring logic

**Read this if you want to:**
- Understand the 19 engagement actions tracked
- See how the transformer model works
- Learn about hash-based embeddings
- Understand candidate isolation mechanism
- Know how ranking predictions are made

**Key Takeaway:** The algorithm predicts 19 engagement probabilities using a Grok-based transformer with candidate isolation, ranking posts primarily by predicted "favorite" (like) probability.

---

### [03_ENGAGEMENT_LOGIC.md](./03_ENGAGEMENT_LOGIC.md)
**What it covers:** How engagement is created, measured, updated, and how feedback loops work

**Read this if you want to:**
- Understand what counts as "engagement"
- Learn how past actions influence future content
- See the positive and negative feedback loops
- Understand signed action embeddings
- Know how the system learns from your behavior

**Key Takeaway:** Every engagement creates signed embeddings (positive/negative) that flow through the transformer to amplify or suppress similar future content in a continuous feedback loop.

---

### [04_GROWTH_DISTRIBUTION.md](./04_GROWTH_DISTRIBUTION.md)
**What it covers:** What the algorithm favors, growth mechanisms, and anti-gaming logic

**Read this if you want to:**
- Understand the two distribution pathways (in-network vs out-of-network)
- Learn what factors drive viral growth
- See the dampening mechanisms (diversity scorer, filters)
- Know the optimal growth strategy
- Understand growth limiters

**Key Takeaway:** The algorithm is velocity-driven (fast engagement > total engagement), quality-gated (negative actions suppress), and diversity-constrained (can't spam to success).

---

### [05_MENTAL_MODEL.md](./05_MENTAL_MODEL.md)
**What it covers:** Simple mental model, core equation, flow diagram, and practical implications

**Read this if you want to:**
- Get a simple mental model of the entire system
- See the one equation that summarizes everything
- Understand the "three laws" of the algorithm
- Learn practical growth strategies
- Get a quick reference card

**Key Takeaway:** The algorithm is a "physics simulation" where posts and users are particles in 128D space, connected by engagement gravity, subject to velocity decay, and constrained by quality/diversity forces.

---

## 🎯 Quick Start

### If you have 5 minutes:
Read [05_MENTAL_MODEL.md](./05_MENTAL_MODEL.md) - Get the core equation and mental model

### If you have 15 minutes:
Read [05_MENTAL_MODEL.md](./05_MENTAL_MODEL.md) + [04_GROWTH_DISTRIBUTION.md](./04_GROWTH_DISTRIBUTION.md) - Understand growth mechanics

### If you have 30 minutes:
Read all 5 documents in order - Get complete understanding

### If you're a developer:
Start with [01_REPO_MAPPING.md](./01_REPO_MAPPING.md) and [02_CORE_ALGORITHM.md](./02_CORE_ALGORITHM.md)

### If you're a content creator:
Focus on [03_ENGAGEMENT_LOGIC.md](./03_ENGAGEMENT_LOGIC.md) and [04_GROWTH_DISTRIBUTION.md](./04_GROWTH_DISTRIBUTION.md)

---

## 🔑 Key Insights

### The One Equation
```
Score = Similarity × Freshness × Quality × Diversity × Velocity
```

### The Three Laws
1. **Embedding Gravity** - Engagement attracts content and users together
2. **Velocity Decay** - Time kills momentum; only fast engagement overcomes decay
3. **Quality Equilibrium** - System self-corrects toward quality through negative actions

### What the Algorithm Favors
- ✅ **Early Traction** (first 1-2 hours critical)
- ✅ **Velocity** (fast engagement > total engagement)
- ✅ **Network Effects** (viral cascades through similarity)
- ⚠️ **Consistency** (limited by diversity scorer)
- ❌ **Spam/Low Quality** (blocked by filters and negative predictions)

### What the Algorithm Wants
> Maximize engagement velocity while maintaining feed quality and diversity

---

## 📊 System Architecture Summary

```
User Request
    ↓
Home Mixer (Orchestration)
    ↓
Query Hydration (User history + features)
    ↓
Candidate Sourcing
    ├── Thunder (In-network posts from followers)
    └── Phoenix Retrieval (Out-of-network via similarity search)
    ↓
Filtering (Remove duplicates, old posts, blocked authors)
    ↓
Scoring
    ├── Phoenix Scorer (ML predictions for 19 actions)
    ├── Weighted Scorer (Combine predictions)
    └── Author Diversity Scorer (Ensure variety)
    ↓
Selection (Top K posts)
    ↓
Feed Response
    ↓
User Engagement
    ↓
Feedback Loop (Update embeddings, retrain model)
```

---

## 🛠️ Technical Stack

- **ML Framework:** JAX + Haiku (Python)
- **Service Layer:** Rust
- **Streaming:** Kafka
- **Model:** Grok-based Transformer (128D embeddings)
- **API:** gRPC

---

## 📈 The 19 Engagement Actions

### Positive (increase score)
1. favorite_score (like) ⭐ PRIMARY RANKING SIGNAL
2. reply_score
3. repost_score
4. quote_score
5. click_score
6. profile_click_score
7. photo_expand_score
8. vqv_score (video view)
9. share_score
10. share_via_dm_score
11. share_via_copy_link_score
12. dwell_score
13. quoted_click_score
14. follow_author_score
15. dwell_time

### Negative (decrease score)
16. not_interested_score
17. block_author_score
18. mute_author_score
19. report_score

---

## 💡 Practical Takeaways

### For Content Creators
1. **Optimize for early velocity** - First 2 hours determine viral potential
2. **Create quality content** - High P(like, reply) and low P(block, mute)
3. **Post when audience is active** - Maximize early engagement
4. **Don't spam** - Diversity scorer penalizes frequent posting
5. **Engage with your niche** - Strengthen embedding similarity

### For Developers
1. **Candidate isolation** prevents gaming through batch manipulation
2. **Hash-based embeddings** enable efficient lookup and updates
3. **Two-tower retrieval** scales to millions of candidates
4. **Signed action embeddings** create attraction/repulsion in embedding space
5. **Transformer with special masking** ensures consistent, cacheable scores

### For Researchers
1. **No hand-engineered features** - Pure transformer-based learning
2. **Multi-action prediction** - 19 simultaneous engagement predictions
3. **Grok-1 architecture** - Adapted for recommendation use case
4. **Embedding-based retrieval** - L2-normalized dot product similarity
5. **Quality self-correction** - Negative actions create repulsion

---

## 🎓 Learning Path

### Beginner
1. Read [05_MENTAL_MODEL.md](./05_MENTAL_MODEL.md)
2. Understand the one equation
3. Learn the three laws
4. Review the quick reference card

### Intermediate
1. Read [03_ENGAGEMENT_LOGIC.md](./03_ENGAGEMENT_LOGIC.md)
2. Understand feedback loops
3. Read [04_GROWTH_DISTRIBUTION.md](./04_GROWTH_DISTRIBUTION.md)
4. Learn growth mechanics

### Advanced
1. Read [01_REPO_MAPPING.md](./01_REPO_MAPPING.md)
2. Read [02_CORE_ALGORITHM.md](./02_CORE_ALGORITHM.md)
3. Study the source code at https://github.com/xai-org/x-algorithm
4. Experiment with the Phoenix models

---

## 🔗 Resources

- **Source Repository:** https://github.com/xai-org/x-algorithm
- **Grok-1 Model:** https://github.com/xai-org/grok-1
- **Phoenix README:** https://github.com/xai-org/x-algorithm/blob/main/phoenix/README.md

---

## 📝 Notes

- This analysis is based on the open-source code released by X.AI
- The production system may have additional optimizations not reflected in the open-source release
- Model parameters and weights are representative, not production values
- The analysis focuses on the algorithmic logic, not infrastructure/scaling details

---

## 🤝 Contributing

This is an analysis repository. For questions or corrections, please open an issue.

---

## 📄 License

This analysis is provided for educational purposes. The original X algorithm code is licensed under Apache 2.0.

---

## 🙏 Acknowledgments

- X.AI team for open-sourcing the algorithm
- Grok-1 team for the transformer implementation
- The open-source community for making this analysis possible

---

**Last Updated:** January 20, 2026
