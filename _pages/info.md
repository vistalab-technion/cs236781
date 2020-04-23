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

Due to the COVID-19 pandemic, this semester the course will be given using a remote-learning approach.

We provide lecture videos, notes, and slides to facilitate self-learning of the
core topics.  The tutorials are based on detailed and self-contained Jupyter
notebooks, which guide you through a full implementation of one or more models
and techniques for solving a specific task. They are meant to teach you the
technical aspects of implementing deep learning systems.  The course also
includes hands-on homework assignments where you will implement working
real-world models and run them on GPUs in the course servers.  Finally, you will be
required to perform a small research project instead of a final exam.

Students are expected to watch and read course material, available from the
course [Lectures]({{ site.baseurl }}/lectures) and [Tutorials]({{ site.baseurl
}}/tutorials) pages.  Viewing and/or reading the lecture and tutorial material is
**mandatory** and crucial for success in the homework and project.


| #    | Date                       | Lecture materials                                                        | Tutorial materials                                                       | Homework   |
| ---- | -----------------------    | ------------------------------------------------------------------------ | -------------------------------------------                              | ---------- |
| 1    | `19/03/2020`               | Lecture 1: Course Introduction (Zoom, video lecture)                     | Tutorial 1: Python, numpy, env setup (Zoom, jupyter notebook)            |            |
| 2    | `26/03/2020`               | Lecture 2: Supervised learning (motivation and main lecture videos)      | Tutorial 2: Supervised learning, PyTorch basics I (video and notebook)   | HW1        |
| 3    | `02/04/2020`               | Lecture 3: Neural networks and CNNs (video)                              | Tutorial 3: MLP, PyTorch basics II (video and notebook)                  |            |
| 4    | `09/04/2020`               | **NO CLASS**                                                             |                                                                          |            |
| 5    | `16/04/2020`               | **NO CLASS**                                                             |                                                                          | HW2        |
| 6    | `23/04/2020`               | Lecture 4: Training (motivation and main lecture videos)                 | Tutorial 4: CNNs, ResNets (video and notebook)                           |            |
| 7    | **Monday**<br>`27/04/2020` | Advanced training techniques                                             |                                                                          |            |
| 8    | `07/05/2020`               | Lecture 5: Sequence models (part 1 and 2 videos)                         | Tutorial 5: Sequence modeling, RNNs, TCNs (video and notebook)           | HW3        |
| 9    | `14/05/2020`               | **NO CLASS**                                                             |                                                                          |            |
| 10   | `21/05/2020`               | Lecture 5: Attention and Transformers (slides)                           | Tutorial 7: Attention (video and notebook)                               |            |
| 11   | **Sunday**<br>`24/05/2020` | Lecture 6: Unsupervised learning and supplementary (videos)              | Tutorial 6: Transfer learning and domain adaptation (video and notebook) |            |
| 12   | `04/06/2020`               | Lecture 7: Deep Reinforcement learning (video)                           | Tutorial 8: Deep reinforcement learning (video and notebook)             | HW4        |
| 13   | `11/06/2020`               | Lecture 11: Non-euclidean domains (video)                                | Tutorial 9: Geometric deep learning (video and notebook)                 |            |
| 14   | `18/06/2020`               | Lecture 8:  Intro to parallel architectures (video)                      | Tutorial 10: CUDA                                                        |            |
| 15   | `25/06/2020`               | Lecture 9:  Training hardware (video)                                    |                                                                          |            |
| 16   | `02/07/2020`               | **NO CLASS**                                                             |                                                                          |            |
| ---- | -----------------------    | ------------------------------------------------------------------------ | -------------------------------------------                              | ---------- |

