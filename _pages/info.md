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


| #    | Date         | Pre-requisite<br>(video) | Lecture<br>(in-class)                           | Tutorial<br>(in-class)                      | Homework   |
| ---- | -----------  | ----                     | ----------------------------------------------- | ------------------------------------------- | ---------- |
| 1    | `24/10/2019` | -                        | Course Introduction                             | Python, numpy and friends                   |            |
| 2    | `31/10/2019` | Lecture 2                | Supervised learning                             | Logistic regression                         | HW1        |
| 3    | `07/11/2019` | Lecture 3                | CNN applications and architectures              | MLP                                         |            |
| 4    | `14/11/2019` | Lecture 4                | Training techniques                             | CNNs                                        |            |
| 5    | `21/11/2019` | Lecture 5                | Attention and Transformers                      | Sequence models                             | HW2        |
| 6    | `28/11/2019` | Lecture 6                | Losses for generative models                    |                                             |            |
| 7    | `05/12/2019` | -                        | Object detection                                | Transfer learning and domain adaptation     |            |
| 8    | `12/12/2019` | Lecture 7                | AutoML                                          | Deep reinforcement learning                 | HW3        |
| 9    | `19/12/2019` |                          | *NO CLASS*                                      |                                             |            |
| 10   | `26/12/2019` |                          | *NO CLASS*                                      |                                             |            |
| 11   | `02/01/2020` | Lecture 11               | Applications of graph neural networks           | Geometric deep learning                     |            |
| 12   | `09/01/2020` | Lecture 12               | DNN compression practices                       | TBD                                         | HW4        |
| 13   | `16/01/2020` | Lecture 9                | Training hardware                               | TBD                                         |            |
| 14   | `23/01/2020` | Lecture 10               | Inference hardware                              | CUDA                                        |            |
| ---- | -----------  | ----                     | ----------------------------------------------- | ------------------------------------------- | ---------- |

