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

**Evaluation**: 

First Homework assignment : 10% 
Second Homework assignment : 25%
Third Homework assignment : 25%
Final Project: 40%

The Project might have a competitive component to it. If there will be, the course staff will anounce it in advance.
We are aware for students worries and it wound **Not** effect your final grade by more then 5% (if at all will be a part of the grading), so do not worry for your grade.


**Language**: All course materials (including lecture and tutorial videos) are provided in English, possible to change upon request.

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


The lectures are **In person**, and suplementry matirial with teoretical background 
exist in the [Lectures]({{ site.baseurl }}/lectures) page.
For some classes, you would be asked to watch a complimenty video to the lecture.

The tutorials are based on detailed and self-contained Jupyter notebooks, which guide you through.

This semester we changed the tutorials and you can use the [Old version]({{ site.baseurl }}/tutorials) as a suplementry matirial.

The course also includes hands-on homework assignments in which you'll
implement working real-world models and run them on GPUs on the course servers.
Performing the assignments **by yourself**  is a crucial aspect of the course, which
will provide you with many of the technical skills required to be effective
with Deep Learning.

This semester's syllabus is provided below. Please watch the linked (🔗) video
lecture before each respective class.

| #    | Date             | Lecture                             | Supplemental (video)                                                      | Tutorial                          | Homework    |
| ---- | -------------    | -------------------------------     | ------------------------------------------------------------------------- | --------------------------------- | ----------  |
| 1    | `23/03/2023`     | Introduction + Supervised learning  |                                                                           | Supervised learning, PyTorch I 	|    HW1      |
| 2    | `30/03/2023`     | Neural networks, CNNs I             | Supervised learning([🔗]({{site.baseurl}}/lectures/02-supervised/))       | MLP, PyTorch II                   |             |
| -    | `06/04/2023`     | **Passover (NO CLASS)**             |                                                                           |                                   |             |
| -    | `13/04/2023`     | **Passover (NO CLASS)**             |                                                                           |                                   |             |
| 3    | `20/04/2023`     | Neural networks, CNNs II            | CNN    [🔗]({{site.baseurl}}/lectures/03-neural_nets/)                    | CNNs (+workshop)                  |             |
| 4    | `24/04/2023`     | Optimization and Training I         | Optimization ([🔗]({{site.baseurl}}/lectures/04-optimization/))           | Optimization                      |             |
| -    | `27/04/2023`     | **independence day (NO CLASS)**     |                                                                           |                                   |             |
| 5    | `04/05/2023`     | Dense prediction                    |                                                                           | Optimization II      	            |     HW2     |
| 6    | `11/05/2023`     | Self supervision                    |                                                                           | Dense Prediction 				    |             |
| -    | `18/05/2023`     | **student festival (NO CLASS)**     |                                                                           |                                   |             |
| 7    | `21/05/2023`     | Sequence models                     |                                                                           | Domain adaptation				    |             |
| -    | `25/05/2023`     | **sahvout (NO CLASS)**     			|                                                                           |                                   |             |
| 8    | `01/06/2023`     | Attention     	                    | Sequence models ([🔗]({{site.baseurl}}/lectures/05-sequence/))            | RNNs				                |    HW3      |
| 9    | `08/06/2023`  	  | Transformers            			| Unsupervised learning ([🔗]({{site.baseurl}}/lectures/06-unsupervised/))  | Attention	               		    |             |
| 10   | `15/06/2023`     | VIT                                 |                                                                           | Transformers                      |             |
| 11   | `22/06/2023`     | GAN                                 |                                                                           | GANs		                   	    |Final Project|
| 12   | `29/06/2023`     | VAE                                 |                                                                           | VAEs                              |             |
| 13   | `06/07/2023`     | Diffusion models                    |                                                                           | Diffusion models                  |             | 
| ---- | -------------    | --------------------------------    | ------------------------------------------------------------------------- | --------------------------------- | ----------- |
