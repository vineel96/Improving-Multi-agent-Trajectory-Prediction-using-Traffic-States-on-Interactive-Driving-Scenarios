# Improving Multi agent Trajectory Prediction using Traffic States on Highly Interactive Driving Scenarios
## Abstract
Predicting trajectories of multiple agents in interactive driving scenarios such as
intersections, and roundabouts are challenging due to the high density of agents, varying
speeds, and environmental obstacles. Existing approaches use relative distance and
semantic maps of intersections to improve trajectory prediction. However, drivers base
their driving decision on the overall traffic state of the intersection and the surrounding
vehicles.So, we propose to use traffic states that denote changing spatio-temporal
interaction between neighboring vehicles, to improve trajectory prediction. An example
of a traffic state is a clump state which denotes that the vehicles are moving close to
each other, i.e., congestion is forming. We develop three prediction models with
different architectures, namely, Transformer-based (TS-Transformer), Generative
Adversarial Network-based (TS-GAN), and Conditional Variational Autoencoder-based
(TS-CVAE). We show that traffic state-based models consistently predict better future
trajectories than the vanilla models.TS-Transformer produces state-of-the-art results on
two challenging interactive trajectory prediction datasets, namely, Eye-on-Traffic
(EOT), and INTERACTION.Our qualitative analysis shows that traffic state-based
models have better aligned trajectories to the ground truth

## Datasets
### EyeonTraffic(EoT) Dataset
Preprocessed EoT datasets are available in this repository, under EOT_split.
Link to Dataset: [EoT Dataset](https://github.com/NaveenKumar-1311/EoT-EyeonTraffic)
### Interaction Dataset
Preprocessed Interaction datasets are available in this repository, under interaction-dataset.
Link to Dataset: [Interaction Dataset](https://github.com/interaction-dataset/interaction-dataset)

## Model Training And Evaluation
This repo consists of Three different base architectures:
- Conditional Variational Autoencoder (CVAE) based model
  - TS-CVAE folder includes required files to train and evaluate traffic state incorporated CVAE
- Generative Adverserial Networks (GAN) based model
  - TS-GAN folder includes required files to train and evaluate traffic state incorporated GAN
- Transformer based model
  - TS-TRANSFORMER folder includes required files to train and evaluate traffic state incorporated Transformer
## Acknowledgment
This work uses datassets
