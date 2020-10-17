---
permalink: /info/
title: Course Info
classes: text-justify
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

**Evaluation**: 100% Homework assignments.

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
   coding, debugging and write all answers yourself. We **will** run automatic
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

Due to the COVID-19 pandemic, this semester the course will be given using a
remote-learning approach.

We provide lecture videos, notes, and slides to facilitate self-learning of the
core topics.  The tutorials are based on detailed and self-contained Jupyter
notebooks, which guide you through a full implementation of one or more models
and techniques for solving a specific task. They are meant to teach you the
technical aspects of implementing deep learning systems.  The course also
includes hands-on homework assignments where you will implement working
real-world models and run them on GPUs in the course servers.

In addition, we provide videos and written material, available from the
course [Lectures]({{ site.baseurl }}/lectures) and [Tutorials]({{ site.baseurl
}}/tutorials) pages.  Viewing and/or reading the lecture and tutorial material is
highly recommended.


| #    | Date            | Lecture                                   | Tutorial                                    | Homework   |
| ---- | --------------- | ----------------------------------------- | ------------------------------------------- | ---------- |
| 1    | `22/10/2020`    | Course Introduction                       | Python, numpy, env setup                    |            |
| 2    | `29/10/2020`    | Supervised learning                       | Supervised learning, PyTorch basics I       | HW1        |
| 3    | `05/11/2020`    | Neural networks and CNNs                  | MLP, PyTorch basics II                      |            |
| 4    | `12/11/2020`    | Training                                  | CNNs, ResNets                               |            |
| 5    | `19/11/2020`    | Advanced training and hardware aspects    |                                             | HW2        |
| 6    | `26/11/2020`    | Sequence models                           | Sequence modeling, RNNs, TCNs               |            |
| 7    | `03/12/2020`    | Attention and Transformers                | Attention                                   |            |
| 8    | `10/12/2020`    | Unsupervised learning I                   | Transfer learning, domain adaptation        | HW3        |
| 9    | `17/12/2020`    | ** NO CLASS **                            |                                             |            |
| 10   | `24/12/2020`    | Unsupervised learning II                  | Deep reinforcement learning                 |            |
| 11   | `31/12/2020`    | Geometric deep learning                   | Geometric deep learning                     | HW4        |
| 12   | `07/01/2021`    | Adversarial robustness                    | TBD                                         |            |
| 13   | `14/01/2021`    | DNN Compression                           | TBD                                         |            |
| 14   | `21/01/2021`    | Hardware Architectures                    | CUDA                                        |            |
| ---- | --------------- | ----------------------------------------- | ------------------------------------------- | ---------- |

