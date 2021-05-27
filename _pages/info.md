---
permalink: /info/
title: Course Info
classes: text-justify 
---

Deep learning is a powerful and relatively-new branch of machine learning. In
recent years it has been successfully applied to some of the most challenging
problems in the broad field of AI, such as recognizing objects in an image,
converting speech to text or playing games. In many such tasks, the state of
the art performance today is attained by deep-learning algorithms, in some
cases surpassing human-level performance.

This course will focus on the theory and algorithms behind deep learning, as
well as on hardware and software techniques that allow efficient training of
deep learning algorithms. It is a graduate-level course which provides both the
necessary theoretical background and the hands-on experience required to be an
effective deep learning practitioner, or to start on the path towards deep
learning research.

## Learning Outcomes

At the end of the course, the student will:

1.	Understand the key notions of deep learning, such as learning regimes, model
    types, optimization and training methodologies.
1.  Be able to apply deep learning algorithms to real-world data and problems.
1.	Know how to effectively use python and pytorch to implement models and
    algorithms from the recent literature.
1.	Know how to leverage GPUs and write custom computational kernels to
    accelerate both training and inference.
1.	Perform a small research project using the studied notions and techniques.


## Administration

**Evaluation**: 100% Homework assignments.

**Language**: The course will be taught in English only.

**Credits**: 3.0.

### Prerequisites

This is an advanced course. Without **both** mathematical maturity and
programming competency it will be very challenging to complete.
The recommended pre-requisites are as follows:

- A good background of linear algebra, probability and calculus. See the
  [supplemental material]({{ site.baseurl }}{% link _pages/supplements.md %})
  page if you need a refresher on one of these.
- Programming competency. The course will be very hands-on; much programming
  will be required.  We will use Python exclusively, so it's crucial to have
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
   coding, debugging and write all answers yourself. We **will** run automatic
   plagiarism-detection software on your submissions to enforce this policy.
3. You may not use any solutions from previous semesters' homeworks.
4. You may not share your solutions with other students.
5. You may not upload your homework solutions to *any* public website, such as
   github. Private repos are OK, but they must remain so even after course completion.

## Course Staff

{% include course_staff.html %}

## Literature

The course does not follow any specific book. For your own reference, the
following material may be useful.

{% include literature.html %}

## Detailed Syllabus


The lectures will follow a flipped-classroom approach: Students will be
requested to watch recorded video lectures as a **mandatory** course
requirement. We provide videos and written material, on the course
[Lectures]({{ site.baseurl }}/lectures) page, to facilitate self-learning of
the core topics. The in-class lectures will be short (1h), optional, and
cover more advanced material, such as state of the art approaches from the
latest research.

The [Tutorials]({{ site.baseurl }}/tutorials) are based on detailed and
self-contained Jupyter notebooks, which guide you through a full implementation
of one or more models and techniques for solving a specific task. They are
meant to teach you the technical aspects of implementing deep learning systems.
The in-class tutorials will cover all this material - no pre-requisite viewing
required.

The course also includes hands-on homework assignments in which you'll
implement working real-world models and run them on GPUs on the course servers.
Performing the assignments in full is a crucial aspect of the course, which
will provide you with many of the technical skills required to be effective
with Deep Learning.

This semester's syllabus is provided below. Please watch the linked (🔗) video
lecture before each respective class.


| Week | Date          | Lecture (video, mandatory)                                                | Supplemental (class, optional) | Tutorial                                                           | Homework   |
| ---- | ------------- | ------------------------------------------------------------------------- | -----------------------------  | -------------------------------------------                        | ---------- |
| 1    | `25/03/2021`  | Introduction ([🔗]({{site.baseurl}}/lectures/lecture_01/))                | Introduction                   | Env setup, numpy, torch
| -    | `01/04/2021`  | **NO CLASS**                                                              | (Passover Holiday)             |                                                                    |            |
| 2    | `08/04/2021`  | Supervised learning ([🔗]({{site.baseurl}}/lectures/lecture_02/))         | Supervised learning            | Supervised learning, PyTorch basics I                              | HW1        |
| -    | `15/04/2021`  | **NO CLASS**                                                              | (Independence Day)             |                                                                    |            |
| 3    | `22/04/2021`  | Neural networks, CNNs ([🔗]({{site.baseurl}}/lectures/lecture_03/))       | CNNs                           | MLP, PyTorch basics II                                             |            |
| 4    | `29/04/2021`  | Training ([🔗]({{site.baseurl}}/lectures/lecture_04))                     | Advanced training              | CNNs I                                                             |            |
| 5    | `06/05/2021`  | -                                                                         | Hardware aspects of training   | CNNs II, ResNets                                                   | HW2        |
| 6    | `13/05/2021`  | Sequence models ([🔗]({{site.baseurl}}/lectures/lecture_05))              | RNNs                           | Optimization I                                                     |            |
| 7    | `20/05/2021`  | -                                                                         | Attention and Transformers     | Optimization II                                                    |            |
| 8    | `27/05/2021`  | Unsupervised learning ([🔗]({{site.baseurl}}/lectures/lecture_06))        | Unsupervised learning I        | Sequence modeling, RNNs, TCNs Transfer learning, domain adaptation | HW3        |
| 9    | `03/06/2021`  | Deep reinforcement learning ([🔗]({{site.baseurl}}/lectures/lecture_07))  | Unsupervised learning II       | Attention                                                          |            |
| -    | `10/06/2021`  | **NO CLASS**                                                              | (Students' Day)                |                                                                    |            |
| 10   | `17/06/2021`  | Non-euclidean domains ([🔗]({{site.baseurl}}/lectures/lecture_11))        | Geometric deep learning        | Deep reinforcement learning                                        |            |
| 11   | `24/06/2021`  | -                                                                         | Adversarial examples           | Geometric deep learning                                            | HW4        |
| 12   | `01/07/2021`  | -                                                                         | DNN Compression                | CUDA Kernels                                                       |            |
| ---- | ------------- | ------------------------------------------------------------------------- | -----------------------------  | -------------------------------------------                        | ---------- |

