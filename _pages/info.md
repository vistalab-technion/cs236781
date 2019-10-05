---
permalink: /info/
title: Course Info
classes: wide text-justify
---

Deep learning is a branch of machine learning which aims to learn meaningful
representations directly from raw data, without the need for traditional feature
engineering.
In recent years it has been successfully applied to some of the most challenging
problems in the broad field of AI, such as recognizing objects in an image,
converting speech to text or playing games. In many such tasks,
the state of the art performance today is attained by deep-learning algorithms,
in some cases surpassing human-level performance.

This course will focus on the theory and algorithms behind deep learning,
as well as on hardware and software interfaces that allow efficient training of
deep learning algorithms. It will provide both the necessary theoretical
background and the hands-on experience required to be an effective deep learning
practitioner, or to start on the path towards deep learning research.

## Learning Outcomes

At the end of the course, the student will:

1.	Understand the key notions of deep learning, such as learning regimes, model
    types, optimization and training methodologies.
1.  Be able to apply deep learning algorithms to real work data and problems.
1.	Know how to effectively use leading python machine-learning and deep
    learning frameworks such as PyTorch.
1.	Know how to leverage GPUs and write custom computational kernels to
    accelerate both training and inference.
1.	Perform a small research project using the studied notions and techniques.


## Administration

**Evaluation**: 40% Homework assignments, 60% final project.

**Language**: The course will be taught in English.

**Credits**: 3.0.

**Prerequisites**:
- A good background of linear algebra, probability and calculus. See the
  [supplemental material]({{ site.baseurl }}{% link _pages/supplements.md %})
  page if you need a refresher on one of these.
- Programming competency. The course will be very hands-on; much programming
  will be required.  We will use Python exclusively, so it's recommended to have
  experience with it.
- An introductory course about machine learning and/or signal/image processing.

## Course Staff

{% include course_staff.html %}

## Literature

The course does not follow any specific book. For your own reference, the
following material may be useful.

{% include literature.html %}

## Detailed Syllabus

This semester, the course will be presented using a flipped-classroom approach.

Students are expected to watch and read the pre-requisite material, available
from the couse [Lectures page]({{ site.baseurl }}/lectures) before each class.
The in-class lectures will then be divided into a *supplementary* part, relating
to the pre-requisite material and an *introductory* part presenting new material
relating to the next lecture.


| Date         | #    | Pre requisite | Lecture                                                                                                                                 | Tutorial                                                                            | Homework   |
| -----------  | ---- | ----          | --------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- | ---------- |
| `17/03/2019` | 1    |               | Introduction to machine learning                                                                                                        | Python, numpy and friends                                                           |            |
| `24/03/2019` | 2    | Lecture 1b    | **Introductory**: Supervised learning, probability and statistics                                                                       | Logistic regression                                                                 | HW1        |
| `31/03/2019` | 3    | Lecture 2     | **Supplementary**:  performance evaluation, ROC, confusion matrix;<br>**Introductory**: neural networks                                 | MLP                                                                                 |            |
| `07/04/2019` | 4    | Lecture 3     | **Supplementary**: CNNs architectures<br>**Introductory**: training, calculus, optimization                                             | CNNs                                                                                |            |
| `14/04/2019` | 5    | Lecture 4     | Training deep networks: Optimization, generalization and regularization                                                                 |                                                                                     | HW2        |
| `21/04/2019` | 6    |               | *No class*                                                                                                                              |                                                                                     |            |
| `28/04/2019` | 7    | Lecture 5     | **Supplementary**: Word embeddings, attention<br>**Introductory**: Unsupervised learning                                                | RNNs                                                                                |            |
| `05/05/2019` | 8    | Lecture 6     | **Supplementary**: GANs, image generation, domain adaptation<br>**Introductory**: Reinforcement learning                                | Domain adaptation                                                                   |            |
| `12/05/2019` | 9    | Lecture 7     | **Supplementary**: Actor-critic, AutoML, NAS<br>**Introductory**: Non-euclidean domains, harmonic analysis                              |                                                                                     | HW3        |
| `19/05/2019` | 10   | Lecture 11    | **Supplementary**: Applications of CNNs on graphs<br>**Introductory**: Hardware accelerators                                            | Deep reinforcement learning                                                         |            |
| `26/05/2019` | 11   | Lecture 8     | Neural network compression and pruning                                                                                                  |                                                                                     |            |
| `02/06/2019` | 12   | Lecture 9     | **Supplementary**: GPU architectures                                                                                                    | Geometric deep learning                                                             |            |
| `09/06/2019` | 13   |               | *No class*                                                                                                                              |                                                                                     | HW4        |
| `16/06/2019` | 14   | Lecture 10    | **Supplementary**: Hardware for inference                                                                                               | CUDA                                                                                |            |
| `23/06/2019` | 15   |               | *Project Presentations*                                                                                                                 |                                                                                     |            |
| -----------  | ---- | ----          | --------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- |            |

