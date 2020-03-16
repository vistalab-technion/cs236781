---
permalink: /info/
title: Course Info
classes: wide text-justify
---

Deep learning is a powerful and relatively-new branch of machine learning.
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
1.  Be able to apply deep learning algorithms to real-world data and problems.
1.	Know how to effectively use python and deep-learning frameworks to implement
    models and algorithms from the recent literature.
1.	Know how to leverage GPUs and write custom computational kernels to
    accelerate both training and inference.
1.	Perform a small research project using the studied notions and techniques.


## Administration

**Evaluation**: 40% Homework assignments, 60% final project.

**Language**: The course will be taught in English.

**Credits**: 3.0.

### Prerequisites

- A good background of linear algebra, probability and calculus. See the
  [supplemental material]({{ site.baseurl }}{% link _pages/supplements.md %})
  page if you need a refresher on one of these.
- Programming competency. The course will be very hands-on; much programming
  will be required.  We will use Python exclusively, so it's recommended to have
  experience with it.
- An introductory course about machine learning and/or signal/image processing.

### Collaboration Policy and Honor Code

By enrolling in this course, you agree that you will strictly follow our
collaboration policy as specified below. Any violation of this policy will
result in an immediate failure in the course, and treatment by the Technion
regulations committee.

0. Submission of assignments is in singles or pairs.
   You are free to form study groups and discuss homeworks with other students.
   However, you must implement all required code independently of other groups
   (only with your submission partner).
1. Submitted work must only be your own. You must do your own thinking,
   coding, debugging and write all answers yourself. We will run automatic
   plagiarism-detection software on your submissions to enforce this policy.
3. You may not use any solutions from previous semesters' homeworks.
4. You may not share your solutions with other students.
5. You may not upload your homework solutions to *any* public website, such as
   github. Private repos are OK, but they must remain so even after course completion.
   As an exception, for the course final project only, you may use a public github repo.

## Course Staff

{% include course_staff.html %}

## Literature

The course does not follow any specific book. For your own reference, the
following material may be useful.

{% include literature.html %}

## Detailed Syllabus

The course will be presented using a mixed approach of offline content
(videos and lecture notes), in-class frontal learning, and hands-on homework
assignments. The frontal lectures
are meant to deepen understanding of the topics in the videos and provide useful
context, techniques and applicative examples. The in-class tutorials and
homework assignments are meant to teach you the technical aspects of
implementing deep learning systems.

Students are expected to watch and read the pre-requisite material, available
from the couse [Lectures page]({{ site.baseurl }}/lectures) before each class.
Viewing and/or reading the pre-requisite material is **mandatory**.


| #    | Date                       | Pre-requisite<br>(video) | Lecture<br>(in-class; Alex, Avi, Chaim)         | Tutorial<br>(in-class; Aviv)                | Homework   |
| ---- | -----------                | ----                     | ----------------------------------------------- | ------------------------------------------- | ---------- |
| 1    | `24/10/2019`               | ---                      | Course Introduction                             | Python, numpy, environment setup            |            |
| 2    | `31/10/2019`               | Lecture 2                | Introduction to hardware for Deep Learning      | Supervised learning, PyTorch basics I       | HW1        |
| 3    | `07/11/2019`               | Lecture 3                | CNN applications and architectures              | MLP, PyTorch basics II                      |            |
| 4    | `14/11/2019`               | Lecture 4                | Training techniques                             | CNNs, ResNets                               |            |
| 5    | `21/11/2019`               | Lecture 5                |                                                 | Sequence modeling, RNNs, TCNs               | HW2        |
| 6    | `28/11/2019`               | Lecture 6                | Attention and Transformers                      |                                             |            |
| 7    | `05/12/2019`               | ---                      | Object detection I                              | Transfer learning and domain adaptation     |            |
| 8    | `12/12/2019`               | Lecture 7                | Object detection II                             |                                             |            |
| 9    | **Sunday<br>`22/12/2019`** | ---                      |                                                 | Attention mechanisms                        | HW3        |
| 10   | `26/12/2019`               |                          | **NO CLASS**                                    |                                             |            |
| 11   | `02/01/2020`               | Lecture 11               | Applications of graph NNs                       | Deep reinforcement learning                 |            |
| 12   | `09/01/2020`               | ---                      | DNN compression                                 |                                             | HW4        |
| 13   | `16/01/2020`               | Lecture 9                | Training hardware                               | Geometric deep learning                     |            |
| 14   | `23/01/2020`               | Lecture 10               | Inference hardware                              | CUDA                                        |            |
| ---- | -----------                | ----                     | ----------------------------------------------- | ------------------------------------------- | ---------- |

