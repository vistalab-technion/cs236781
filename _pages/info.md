---
permalink: /info/
title: Course Info
classes: text-justify wide
---

Deep learning is a powerful and relatively new branch of machine learning. In
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

1.	Understand the key notions of deep learning, such as neural networks,
    learning regimes, optimization algorithms and training methodologies.
1.  Be able to apply deep learning algorithms to real-world data and problems.
1.	Know how to effectively use python and pytorch to implement models and
    algorithms from the recent literature.
1.	Perform a small research project using the studied notions and techniques.


## Administration

**Evaluation**: 100% Homework assignments.

**Language**: All course materials (including lecture and tutorial videos) are provided in English.

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
The old course staff:
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
the core topics. The in-class lectures will be short (1h), and
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

| #    | Date             | Lecture (video)                                                              | Supplemental (class)          | Tutorial                                    | Homework   |
| ---- | -------------    | -------------------------------------------------------------------------    | ----------------------------- | ------------------------------------------- | ---------- |
| 1    | `24/03/2022`     | Introduction ([🔗]({{site.baseurl}}/lectures/01-intro/))                     | Introduction                  | Env setup, numpy, torch
| 2    | `31/03/2022`     | Supervised learning ([🔗]({{site.baseurl}}/lectures/02-supervised/))         | Supervised learning           | Supervised learning, PyTorch basics I       | HW1        |
| 3    | `07/04/2022`     | Neural networks, CNNs ([🔗]({{site.baseurl}}/lectures/03-neural_nets/))      | CNNs I                        | MLP, PyTorch basics II                      |            |
| 4    | `14/04/2022`     | -                                                                            | CNNs II                       | CNNs                                        |            |
| -    | `21/04/2022`     | **NO CLASS**                                                                 | (Passover)                    |                                             |            |
| 5    | `28/04/2022`     | Training and Optimization ([🔗]({{site.baseurl}}/lectures/04-optimization/)) | Training                      | Optimization I                              | HW2        |
| -    | `05/05/2022`     | **NO CLASS**                                                                 | (Independence Day)            |                                             |            |
| 6    | `12/05/2022`     | Sequence models ([🔗]({{site.baseurl}}/lectures/05-sequence/))               | RNNs                          | Optimization II                             |            |
| 7    | `19/05/2022`     | -                                                                            | Attention and Transformers    | Sequence modeling, RNNs I                   |            |
| 8    | **`24/05/2022`** | Unsupervised learning ([🔗]({{site.baseurl}}/lectures/06-unsupervised/))     | Unsupervised learning I       | RNNs II, TCNs                               |            |
| 9    | `02/06/2022`     | Deep reinforcement learning ([🔗]({{site.baseurl}}/lectures/07-rl/))         | Unsupervised learning II      | Attention I                                 | HW3        |
| 10   | `09/06/2022`     | Non-euclidean domains ([🔗]({{site.baseurl}}/lectures/08-geometric/))        | Geometric DL I                | Attention II                                |            |
| 11   | `16/06/2022`     | -                                                                            | Geometric DL II               | Transfer learning, domain adaptation        |            |
| 12   | `23/06/2022`     | -                                                                            | Visual Attention              | Deep reinforcement learning                 |            |
| 13   | `30/06/2022`     | -                                                                            | Adversarial examples          | Geometric deep learning                     | HW4        |
| ---- | -------------    | -------------------------------------------------------------------------    | ----------------------------- | ------------------------------------------- | ---------- |

