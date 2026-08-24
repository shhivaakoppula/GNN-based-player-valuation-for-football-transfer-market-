Yes. If you mean **“write the actual `README.md` file that I can put directly into my GitHub repository”**, use this version:

````markdown
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
````

RQ1 focuses on whether the graph-based model can improve action-level prediction.

RQ2 investigates whether the resulting player-level score is related to market value.

RQ3 investigates whether the representation can support player scouting and identification of potentially undervalued players.

---

# 4. Dataset

The main football data used in this research comes from **StatsBomb Open Data**.

The dataset provides detailed football event information and, for selected matches, additional StatsBomb 360 positional information.

The data contains football events such as:

* Pass
* Carry
* Shot
* Pressure
* Dribble
* Duel
* Interception
* Clearance
* Ball Recovery
* Foul
* Goalkeeper actions
* Substitution
* Other match events

The extracted event dataset contains approximately:

**1.58 million football events**

from the selected match collection.

---

# 5. Event Data

The event data describes what happened during a football match.

Important variables include:

| Variable              | Description                           |
| --------------------- | ------------------------------------- |
| `match_id`            | Unique match identifier               |
| `id`                  | Unique event identifier               |
| `period`              | Match period                          |
| `timestamp`           | Event timestamp                       |
| `minute`              | Match minute                          |
| `second`              | Match second                          |
| `type.name`           | Type of football event                |
| `player.id`           | Player identifier                     |
| `player.name`         | Player name                           |
| `team.id`             | Team identifier                       |
| `team.name`           | Team name                             |
| `location`            | Location of the event                 |
| `pass.end_location`   | End location of a pass                |
| `pass.length`         | Pass length                           |
| `pass.angle`          | Pass angle                            |
| `pass.recipient.name` | Pass recipient                        |
| `shot.statsbomb_xg`   | StatsBomb expected goals value        |
| `shot.outcome.name`   | Shot outcome                          |
| `carry.end_location`  | End location of a carry               |
| `under_pressure`      | Whether the player was under pressure |

The complete event dataset contains many additional event-specific variables.

---

# 6. StatsBomb 360 Data

StatsBomb 360 data provides additional spatial information around football events.

This allows the research to investigate not only:

> "What action did the player perform?"

but also:

> "What was the spatial situation when the action was performed?"

Important variables include:

| Variable       | Description                                    |
| -------------- | ---------------------------------------------- |
| `match_id`     | Match identifier                               |
| `event_uuid`   | Event identifier associated with the 360 frame |
| `visible_area` | Visible part of the pitch                      |
| `x`            | Player x-coordinate                            |
| `y`            | Player y-coordinate                            |
| `teammate`     | Whether the player is a teammate               |
| `keeper`       | Whether the player is a goalkeeper             |
| `actor`        | Whether the player performed the event         |

This information provides the basis for constructing player interaction graphs.

---

# 7. Graph Representation

Football situations can be represented as graphs.

## Nodes

Each visible player can be represented as a node.

Node features can include:

* X-coordinate
* Y-coordinate
* Teammate/opponent indicator
* Goalkeeper indicator
* Actor indicator
* Distance to goal
* Angle to goal

## Edges

Edges represent relationships between players.

For the prototype, spatial proximity is used to determine neighbouring players.

This allows the model to represent relationships such as:

```text
Player A
 /     \
Player B Player C
 \       /
  Player D
```

The graph therefore represents the spatial structure of the football situation.

---

# 8. Graph Neural Network

A Graph Neural Network (GNN) is a neural-network architecture designed for graph-structured data.

A football situation is suitable for graph modelling because players interact with each other.

The specific architecture investigated in this research is the:

## Graph Attention Network (GAT)

GAT uses an attention mechanism to learn the relative importance of neighbouring nodes.

In the context of football, this allows the model to learn which surrounding players and spatial relationships are more relevant to the action being evaluated.

---

# 9. Baseline: VAEP

One of the existing football action-value approaches considered in this research is:

**VAEP — Valuing Actions by Estimating Probabilities**

VAEP evaluates football actions by estimating how an action changes the probability of desirable outcomes.

The thesis investigates whether a graph-based representation containing spatial player relationships can provide additional predictive information compared with an established football action-value methodology.

The comparison must use a consistent experimental setup, including:

* Comparable prediction target
* Comparable dataset
* Appropriate training/test split
* Same evaluation metric
* No information leakage

---

# 10. Training and Validation

A key part of the research methodology is separating data used to develop the model from data used to evaluate it.

## Training Data

Training data is used to:

* Learn model parameters
* Develop the model
* Test modelling decisions

## Validation/Test Data

Previously unseen data is used to:

* Evaluate predictive performance
* Measure generalisation
* Compare models

The research considers football-specific splitting strategies.

### Temporal Split

Earlier matches can be used for model development and later matches for evaluation.

```text
Earlier matches
      |
      v
Training
      |
      v
Model
      |
      v
Later matches
      |
      v
Validation/Test
```

This approach reflects a realistic football analytics scenario in which historical data is used to make predictions about future matches.

Other split strategies, such as team or league-based splits, may also be considered depending on the final experimental design.

---

# 11. Data Processing

The research follows a structured data-processing pipeline:

```text
Raw Data
   |
   v
Data Extraction
   |
   v
Data Selection
   |
   v
Data Cleaning
   |
   v
Event / 360 Matching
   |
   v
Feature Engineering
   |
   v
Graph Construction
   |
   v
Model Training
   |
   v
Model Evaluation
```

The processing steps are documented so that the research can be reproduced and the methodological decisions can be inspected.

---

# 12. Feature Engineering

The graph model uses spatial and contextual information.

Example node features include:

### Spatial features

* `x`
* `y`

### Player relationship features

* `teammate`
* `keeper`
* `actor`

### Derived spatial features

* Distance to goal
* Angle to goal

These features are used to represent the spatial context of the football action.

The feature-selection process will be documented and justified as part of the thesis methodology.

---

# 13. Class Imbalance

Football actions are not evenly distributed.

For example, there are many more:

* Passes
* Carries
* Ball receipts

than:

* Shots
* Goals
* Other rare events

Therefore, the classification problem can contain a substantial class imbalance.

This is important when evaluating the model.

For this reason, the research does not rely only on accuracy.

Additional metrics such as:

* ROC-AUC
* PR-AUC
* Precision
* Recall
* F1-score

are considered.

---

# 14. Evaluation Metrics

## ROC-AUC

**ROC-AUC = Area Under the Receiver Operating Characteristic Curve**

ROC-AUC measures how well a model can distinguish between different classes across classification thresholds.

A value closer to:

**1.0**

indicates stronger discrimination.

---

## PR-AUC

**PR-AUC = Area Under the Precision-Recall Curve**

PR-AUC is particularly useful when the positive class is relatively rare.

---

## Precision

Precision measures:

> Of the events predicted as positive, how many were actually positive?

---

## Recall

Recall measures:

> Of all actual positive events, how many did the model identify?

---

## F1-score

F1-score combines precision and recall into a single metric.

---

## R²

**R² = Coefficient of Determination**

R² is used for the player market-value analysis.

It measures how much of the variation in the target variable can be explained by the model.

---

# 15. Research Transparency

An important principle of this thesis is transparency.

The research will document:

* Data sources
* Data selection criteria
* Data-cleaning procedures
* Feature definitions
* Graph-construction decisions
* Model architecture
* Hyperparameters
* Training procedure
* Validation strategy
* Evaluation metrics
* Baseline methodology
* Experimental results

All important methodological decisions should have a clear reason.

---

# 16. Literature Review

The literature review covers several areas:

### Football Analytics

Research concerning:

* Football event analysis
* Player evaluation
* Tactical analysis
* Action valuation

### Action Valuation

Research concerning:

* VAEP
* Expected goals
* Expected assists
* Possession value
* Action-value models

### Graph Neural Networks

Research concerning:

* GNN
* GCN
* GAT
* Graph representation learning

### Sports Analytics

Research concerning:

* Player performance
* Player recruitment
* Player valuation
* Scouting

### Machine Learning

Research concerning:

* Classification
* Regression
* Imbalanced datasets
* Model evaluation

The literature search methodology will be documented separately, including:

* Databases searched
* Search terms
* Boolean operators
* Inclusion criteria
* Exclusion criteria
* Screening process
* Relevant papers
* Relationship of papers to the research questions

---

# 17. Reproducibility

The goal of this repository is to make the research process understandable and reproducible.

The repository will contain:

```text
football-player-evaluation/
│
├── README.md
│
├── data/
│
├── scripts/
│
├── src/
│
├── notebooks/
│
├── results/
│
├── models/
│
├── literature/
│
├── thesis/
│
└── requirements.txt
```

The exact directory structure may change as the research develops.

---

# 18. Software and Technologies

The research uses Python-based data-science and machine-learning tools.

Main technologies include:

* Python
* Pandas
* NumPy
* Scikit-learn
* PyTorch
* PyTorch Geometric
* Matplotlib
* StatsBomb data
* Jupyter Notebook

Additional libraries may be added as the research develops.

---

# 19. Abbreviations

| Abbreviation | Full Form                                              |
| ------------ | ------------------------------------------------------ |
| AI           | Artificial Intelligence                                |
| ML           | Machine Learning                                       |
| DL           | Deep Learning                                          |
| GNN          | Graph Neural Network                                   |
| GAT          | Graph Attention Network                                |
| GCN          | Graph Convolutional Network                            |
| VAEP         | Valuing Actions by Estimating Probabilities            |
| ROC          | Receiver Operating Characteristic                      |
| AUC          | Area Under the Curve                                   |
| ROC-AUC      | Area Under the Receiver Operating Characteristic Curve |
| PR-AUC       | Area Under the Precision-Recall Curve                  |
| xG           | Expected Goals                                         |
| R²           | Coefficient of Determination                           |
| KNN          | K-Nearest Neighbours                                   |
| FAISS        | Facebook AI Similarity Search                          |
| API          | Application Programming Interface                      |
| CSV          | Comma-Separated Values                                 |
| GATConv      | Graph Attention Convolution                            |

---

# 20. Current Research Status

The research is currently focused on developing and validating the methodology for **RQ1**.

The current work includes:

* Football data extraction
* Event-data exploration
* StatsBomb 360 data processing
* Event and 360-data matching
* Data cleaning
* Feature engineering
* Graph construction
* GAT model development
* Initial model training
* Initial evaluation

The prototype is being used to identify methodological problems before applying the approach to a larger and more rigorous experimental setup.

The results from the prototype should therefore be interpreted as **preliminary results**, not as the final thesis findings.

---

# 21. Future Work

The next stages of the research include:

### RQ1

* Finalise the experimental design
* Establish a properly justified baseline
* Ensure a fair comparison with VAEP
* Evaluate different model configurations
* Perform robustness checks
* Investigate generalisation

### RQ2

* Develop player-level aggregation
* Obtain and integrate market-value data
* Compare GNN-based scores with conventional statistics
* Perform regression analysis
* Compare R² and other relevant metrics

### RQ3

* Create player representations
* Apply similarity search
* Compare player performance with market value
* Identify potential undervalued players
* Perform qualitative analysis of candidate players

---

# 22. Research Goal

The overall goal of the thesis is to investigate whether **spatial and relational information between football players can improve football performance evaluation using Graph Neural Networks**.

The research ultimately aims to connect:

```text
Football Events
       +
Spatial Context
       +
Player Relationships
       ↓
Graph Neural Network
       ↓
Action Evaluation
       ↓
Player Performance
       ↓
Market Value
       ↓
Scouting / Recruitment
```

---

## Author

**Master's Thesis — Data Science**

This repository is developed as part of my Master's thesis research.

---

## Disclaimer

The results presented in this repository are part of an ongoing academic research project.

Prototype results should not be interpreted as final conclusions until the complete experimental design, validation procedure, baseline comparison, and statistical analysis have been completed.

```
```
