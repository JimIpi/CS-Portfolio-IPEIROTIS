# Power-Network Disruption: Predicting Interruptions using Network Features

## 1. Project Description
This is the final portfolio project for the "Complex Systems and Applications" course (Option 2). It investigates whether a substation's local position in an electrical grid can help predict its likelihood of failing after an initial disruption in the network.

## 2. Problem Statement
The goal is to predict which currently functioning substations will experience a simulated interruption over the following 24 hours. Specifically, the central question is whether adding network topology features (direct connectivity and local cross-links) improves the predictive power of a logistic regression model compared to using only basic substation attributes (load and equipment age). 

## 3. Data Set
The project uses two prepared synthetic datasets provided by the course:
* **power_substations.csv**: Contains node information including `substation_id`, `load_ratio` (pre-disruption load), `equipment_age_years`, initial status, and the target variable `interrupted_later`.
* **power_links.csv**: Contains the undirected physical connections between substations, used to build the network graph.

*Note: The data represents a synthetic teaching scenario, not a physical cascade simulation or operational power-system data.*

## 4. Method
* **Network Construction:** Used `NetworkX` to build an undirected graph of the substations and calculate two network features for each node: `degree` (number of direct links) and `local_clustering` coefficient.
* **Modeling:** Trained two Logistic Regression models using `scikit-learn`.
* **Preprocessing:** Applied `StandardScaler` within a pipeline. Data was split into a 70% training and 30% stratified test set.
* **Comparison:** Evaluated a baseline model (using only load and age) against a network-aware model (adding degree and clustering) on exactly the same test observations.

## 5. Results
The model with network features significantly outperformed the baseline model across all key metrics on the test set:
* **Baseline Model:** Accuracy: 0.767 | F1 Score: 0.444 | ROC AUC: 0.828
* **Network-Aware Model:** Accuracy: 0.860 | F1 Score: 0.750 | ROC AUC: 0.949

Looking at the confusion matrices (out of 12 actual later interruptions):
* False Negatives (missed interruptions) dropped from 8 to 3.
* False Positives (false alarms) slightly increased from 2 to 3.

## 6. Interpretation
Adding the network features clearly helped predict later interruptions in this synthetic test set. Degree measures how many direct neighbours a substation has, while local clustering measures how densely interconnected those neighbours are. 

**Limitations & Ethics:** Because these are synthetic interruptions and not electrical flow simulations, a better score does not establish a causal failure process. If this model were applied to a real grid to prioritise repairs, ethical issues arise: false negatives could lead to unprotected communities suffering power outages, while false positives waste valuable repair resources. Real-world implementation would strictly require human engineering review.

## 7. Reflection
The project went well, and using `NetworkX` to extract features from tabular link data was very effective. The most challenging part was ensuring the data leakage was prevented (calculating network features without using the target variable and fitting the scaler only on the training set). With more time, I would like to explore testing non-linear models like Random Forests to see if they can capture more complex interactions between the load, age, and network position.
