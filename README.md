# GNN-based player valuation for football transfer market
# Football Player Evaluation Using Graph Neural Networks

## Master's Thesis

This GitHub repository contains the code, data-processing workflow, experiments, and documentation for my Master's thesis in Data Science.

The thesis investigates whether Graph Neural Networks (GNNs), particularly Graph Attention Networks (GATs), can be used to evaluate football actions and player performance by considering the spatial relationships between players.

---

## 1. Thesis Overview

Football performance is often evaluated using traditional statistics such as:

- Goals
- Assists
- Passes
- Shots
- Tackles
- Minutes played

These statistics are useful, but they do not always capture the context in which an action happened.

For example, two players may complete the same pass, but the difficulty and value of the two passes may be very different depending on:

- The player's position
- Teammate positions
- Opponent positions
- Distance to the goal
- Pressure from opponents
- Available passing options
- Tactical situation

This thesis investigates whether this spatial and relational information can be represented using graphs and processed using Graph Neural Networks.

The central idea is:

> **Instead of evaluating a football action only by what happened, evaluate the action together with the spatial context in which it happened.**

---

# 2. Research Questions

## RQ1 — Football Action Evaluation

### Research Question

**Does a Graph Attention Network (GAT) achieve better predictive performance than an established football action-value approach such as VAEP?**

The main evaluation metric is:

- ROC-AUC

Additional metrics include:

- PR-AUC
- Precision
- Recall
- F1-score
- Confusion Matrix

The objective is to determine whether incorporating spatial relationships between players improves football action prediction.

---

## RQ2 — Player Market Value

### Research Question

**Does a GNN-based player performance score predict football player market value better than conventional football statistics?**

The GNN-based player score will be compared with traditional performance indicators such as:

- Goals
- Assists
- Pass-related statistics
- Other conventional football statistics

The main statistical comparison will use:

- R² (Coefficient of Determination)

The objective is to investigate whether the proposed player representation contains information that is useful for explaining differences in player market value.

---

## RQ3 — Identification of Undervalued Players

### Research Question

**Can the proposed GNN-based player representation identify players who have strong performance profiles but relatively low market values?**

The research will investigate whether player representations can be used to find players who are:

- High-performing according to the model
- Similar to high-performing players
- Relatively low in market value

Similarity-search methods such as FAISS may be investigated for this purpose.

---

# 3. Research Framework

The overall research framework is:

```text
Football Event Data
        |
        v
Spatial / 360 Data
        |
        v
Player Interaction Representation
        |
        v
Graph Construction
        |
        v
Graph Neural Network
        |
        v
Action Evaluation
        |
        v
Player Performance Representation
        |
        +----------------------+
        |                      |
        v                      v
Market Value Analysis     Undervalued Players
        |
        v
       RQ2                    RQ3
