# STEP 5: Mental Model

## The Simple Model

The X algorithm is an **"Engagement-Weighted Embedding Similarity Engine with Velocity Decay"**.

Think of it as:

> Every post and user exists in a 128-dimensional "engagement space"  
> Posts that get engaged with move closer to users who engaged  
> Users move toward posts they engage with  
> The algorithm shows you posts that are CLOSE to you in this space  
> But only if they're RECENT and FRESH

---

## The One Equation

```
Score(post, user) = 
    
    Similarity(user_embedding, post_embedding)
    × Freshness(post_age)
    × Quality(Σ positive_engagements - Σ negative_engagements)
    × Diversity(1 - author_saturation)
    × Network(viral_momentum)

Where:
    user_embedding = Transformer(user_history + past_actions)
    post_embedding = Hash(post_content + author)
    
    Freshness = exp(-λ × post_age)  // Exponential decay
    
    Quality = Σ(w_i × P(action_i))
            = w_like × P(like) 
            + w_reply × P(reply)
            + w_repost × P(repost)
            - w_block × P(block)
            - w_mute × P(mute)
            - w_report × P(report)
    
    Diversity = max(0, 1 - (author_posts_shown / diversity_threshold))
    
    Network = sqrt(total_engagements / time_since_post)  // Velocity
```

**Simplified:**
```
Score = How_Similar × How_Fresh × How_Good × How_Diverse × How_Viral
```

---

## The Flow Diagram

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    X ALGORITHM MENTAL MODEL                              │
│                  "Engagement Gravity in 128D Space"                      │
└──────────────────────────────────────────────────────────────────────────┘

                        THE EMBEDDING SPACE
    
    User A ●────────────────────────────────────● Post X
           │                                    │
           │  Engagement                        │  Similar
           │  pulls closer                      │  content
           │                                    │
           ▼                                    ▼
    User A'●────────────────────────────────────● Post Y
           │                                    │
           │  High                              │  Viral
           │  similarity                        │  momentum
           │                                    │
           └────────────────────────────────────┘
                        FEED SHOWS Y


                    THE COMPLETE FLOW

┌─────────────┐
│ POST CREATED│
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│ EMBEDDING ASSIGNMENT                                        │
│ post_embedding = Hash(content + author)                     │
│ Initial position in 128D space                              │
└──────┬──────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│ DISTRIBUTION PHASE 1: IN-NETWORK (Thunder)                  │
│ • Immediate delivery to followers                           │
│ • Sub-millisecond latency                                   │
│ • Guaranteed visibility (no ML filtering)                   │
└──────┬──────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│ EARLY ENGAGEMENT WINDOW (0-2 hours)                         │
│ • Followers engage (like, reply, repost)                    │
│ • Post embedding MOVES toward engaged users                 │
│ • Velocity metric calculated: engagements/time              │
└──────┬──────────────────────────────────────────────────────┘
       │
       ├──── High Velocity ────────────────────────────────────┐
       │                                                       │
       ▼                                                       ▼
┌─────────────────────────────────────────────────────────────┐
│ DISTRIBUTION PHASE 2: OUT-OF-NETWORK (Phoenix Retrieval)    │
│                                                             │
│ For each user:                                              │
│   1. Encode user: user_emb = Transformer(history)           │
│   2. Similarity search: scores = dot(user_emb, all_posts)   │
│   3. Retrieve top 1000 similar posts                        │
│                                                             │
│ Post with strong embedding gets retrieved MORE              │
└──────┬──────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│ FILTERING STAGE                                             │
│ ❌ Too old (AgeFilter)                                      │
│ ❌ Already seen (PreviouslySeenFilter)                      │
│ ❌ Blocked/muted author (AuthorSocialgraphFilter)           │
│ ❌ Duplicate content (DropDuplicatesFilter)                 │
│ ✅ Fresh, unseen, quality content passes                    │
└──────┬──────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│ RANKING STAGE (Phoenix Transformer)                         │
│                                                             │
│ Input: [User + History + Candidates]                        │
│ Output: P(like), P(reply), P(repost), P(block), ...         │
│                                                             │
│ Final_Score = Σ(weight_i × P(action_i))                     │
│             = 2.0×P(like) + 1.5×P(reply) + 1.0×P(repost)    │
│               - 3.0×P(block) - 2.0×P(mute) - 5.0×P(report)  │
└──────┬──────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│ DIVERSITY ADJUSTMENT                                        │
│ • Author Diversity Scorer attenuates repeated authors       │
│ • OON Scorer adjusts out-of-network content                 │
│ • Ensures feed variety                                      │
└──────┬──────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│ FINAL SELECTION                                             │
│ • Sort by final score                                       │
│ • Select top K (e.g., 50 posts)                             │
│ • VF Filter (safety check)                                  │
└──────┬──────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│ USER SEES FEED                                              │
└──────┬──────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│ USER ENGAGES                                                │
│ • Likes → Positive signal                                   │
│ • Replies → Positive signal                                 │
│ • Blocks → Negative signal                                  │
└──────┬──────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│ FEEDBACK LOOP                                               │
│ • User embedding updates (moves toward liked posts)         │
│ • Post embedding updates (moves toward engaged users)       │
│ • Action embeddings update (learns patterns)                │
│ • Transformer weights update (offline training)             │
└──────┬──────────────────────────────────────────────────────┘
       │
       └──────────────────────────────────────────────────────┐
                                                              │
                                                              ▼
                                                    NEXT FEED REQUEST
                                                    (Loop continues)
```

---

## What This Algorithm REALLY Wants

The algorithm has **one core objective**:

> **Maximize engagement velocity while maintaining feed quality and diversity**

Breaking this down:

### 1. Maximize Engagement
```
More engagement = Better embeddings = Better predictions = More engagement
```
The algorithm is optimizing for a **self-reinforcing engagement loop**.

### 2. Velocity Over Volume
```
10 likes in 1 hour > 100 likes in 1 week
```
The algorithm prefers **fast engagement** (viral content) over slow accumulation.

### 3. Quality Over Quantity
```
1 great post > 10 mediocre posts
```
Author Diversity Scorer ensures you can't spam your way to success.

### 4. Freshness Over History
```
Recent post with 10 likes > Old post with 1000 likes
```
Recency filters and retention windows enforce temporal decay.

### 5. Similarity Over Serendipity
```
Show users MORE of what they engage with
```
The embedding space naturally clusters similar content and users.

---

## The Core Insight

**The algorithm is a MOMENTUM DETECTOR in engagement space.**

It's asking:
1. **Is this post gaining momentum?** (velocity)
2. **Is it similar to what this user likes?** (embedding similarity)
3. **Is it fresh?** (recency)
4. **Is it quality?** (positive vs negative engagements)
5. **Is it diverse?** (not too much from same author)

If **YES** to all → **AMPLIFY**  
If **NO** to any → **SUPPRESS**

---

## The Three Laws of X Algorithm

### Law 1: The Law of Embedding Gravity
```
Content and users are attracted to each other through engagement.
Every engagement pulls them closer in embedding space.
Proximity = Probability of future connection.
```

### Law 2: The Law of Velocity Decay
```
All content decays exponentially with time.
Only velocity (engagement rate) can overcome decay.
Momentum is temporary; freshness is mandatory.
```

### Law 3: The Law of Quality Equilibrium
```
The system self-corrects toward quality.
Low-quality content triggers negative actions.
Negative actions create repulsion in embedding space.
Only sustainable quality achieves sustained reach.
```

---

## The Mental Model in One Sentence

**"The X algorithm is a physics simulation where posts and users are particles in 128-dimensional space, connected by engagement gravity, subject to exponential time decay, and constrained by quality and diversity forces."**

---

## Practical Implications

If you want to **grow on X**, you must:

### 1. Optimize for Early Velocity
- First 2 hours are critical
- Seed with engaged followers
- Post when audience is active

### 2. Create Quality Content
- High P(like, reply, repost)
- Low P(block, mute, report)
- Engagement quality > engagement quantity

### 3. Leverage Embedding Similarity
- Understand your niche's embedding cluster
- Create content similar to what works
- Engage with similar creators to strengthen embeddings

### 4. Respect the Constraints
- Don't spam (diversity penalty)
- Don't repost (filtered out)
- Don't engagement bait (negative predictions)

### 5. Ride the Momentum
- When a post gains traction, engage with replies
- Momentum compounds through network effects
- Viral cascades happen through embedding similarity

---

## What the Algorithm Doesn't Want

❌ **Stale content** (recency decay kills it)  
❌ **Spam** (diversity scorer suppresses it)  
❌ **Low quality** (negative actions repel it)  
❌ **Manipulation** (candidate isolation prevents gaming)  
❌ **Repetition** (seen/served filters block it)

---

## The Ultimate Truth

**The algorithm is not trying to show you "the best content."**

**It's trying to show you "content most likely to make you engage, that is fresh, diverse, and won't make you leave the platform."**

It's an **engagement maximization engine** with **quality guardrails** and **diversity constraints**.

The "For You" feed is really the **"Most Likely To Make You Engage Right Now"** feed.

---

## Quick Reference Card

```
┌─────────────────────────────────────────────────────────────┐
│              X ALGORITHM QUICK REFERENCE                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│ ONE EQUATION:                                               │
│   Score = Similarity × Freshness × Quality ×               │
│           Diversity × Velocity                              │
│                                                             │
│ ONE DIAGRAM:                                                │
│   Post → Embedding → Early Engagement → Velocity →         │
│   Retrieval → Ranking → Feed → Engagement →                │
│   Embedding Update → Loop                                   │
│                                                             │
│ ONE TRUTH:                                                  │
│   The algorithm wants you to engage, fast, with             │
│   quality content, repeatedly.                              │
│                                                             │
│ THREE LAWS:                                                 │
│   1. Embedding Gravity (engagement attracts)                │
│   2. Velocity Decay (time kills momentum)                   │
│   3. Quality Equilibrium (system self-corrects)             │
│                                                             │
│ FIVE FACTORS:                                               │
│   ✅ Early Traction (strong)                                │
│   ✅ Velocity (strongest)                                   │
│   ✅ Network Effects (medium)                               │
│   ⚠️  Consistency (weak, limited by diversity)              │
│   ❌ Spam/Low Quality (blocked)                             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Summary

The X algorithm is:
- **A 128D embedding similarity engine** (core mechanism)
- **Velocity-driven** (favors fast engagement)
- **Quality-gated** (negative actions suppress)
- **Diversity-constrained** (prevents spam)
- **Recency-biased** (fresh content prioritized)
- **Network-amplified** (viral cascades through similarity)

**The system rewards:** Early traction + High velocity + Quality engagement  
**The system limits:** Spam + Low quality + Excessive posting

**The core loop:** Engagement → Embedding update → Similar content → More engagement
