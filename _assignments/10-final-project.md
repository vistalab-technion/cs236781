---
title: Final Project
permalink: assignments/final-project
toc: false
toc_label: Contents
toc_sticky: true
published: true
---

As part of this course, students must complete a small research project instead
of a final exam.

We provide you with a pool of project ideas mostly based on recent papers and
our current research interests.  The projects will be supervised by course TAs
or other graduate students from the VISTA lab.  During the project
period you will be able to meet with course TAs for guidance during dedicated
office hours.  The project can be performed in groups of at most two students.

Some of the projects are "reimplementation projects" where you should start from
a specific paper, implement it, suggest at least one improvement and evaluate it
using experiments which you must also devise and implement.  Selecting this kind
of project means we may expect more of you, in terms of the project ideas and
experimental methodology.

For graduate students only, we allow you to define a custom project that relates
to your own research topic.

If you have further **questions** regarding the course final project, please
contact [Chaim Baskin](mailto:chaimbaskin@cs.technion.ac.il).

The project topics spreadsheet link will be sent by email to all registered
students.

## Project procedure

1. Each group should read the topics and paper abstracts in the project topics spreadsheet.
1. Each group must fill out the registration form with their top-3 priorities
   or custom project proposal until date **01/02/2020**.
1. Soon after you submit your priorities, we will approve one of
   them. Submitting the form sooner will increase your chance of getting your
   first priority.
1. You will start working on your projects.
   You can schedule periodic meetings with course staff at dedicated office
   hours which will be published.
1. Your submission should be a detailed report. It should explain the problem
   and the paper you implemented (if relevant), explain your specific
   enhancements and modifications, showcase all your results (both reproduction
   of the paper and novel results, if any), compare your solution to the
   baseline method (if relevant), etc. See below for details submission
   instructions.
1. The submission date for the project is **16/04/2020**.
   Note that the course servers will not be available to Winter 19-20 students after this date.

## Registration links

Please view the project topics spreadsheet (link will be sent via emali)
and then fill out the [registration form](https://forms.gle/8rYUFWspfkeX3dBKA) with your priorities.


## Project-related office hours

Initial office hours will be given by Chaim on Monday 27/01, 11:30 at the VISTA
lab, Taub 120.
You can use these hours to consult regarding project selection.

We will publish additional office hours in the coming weeks.

## Report structure and evaluation

The following list details what your project report should contain and its
impact on the grade.

0. Abstract (10%). Summarize your work. Briefly introduce the problem, the
   methods and state the key results.

1. Intro (25%). Review the papers relevant to your project. Explain the
   problem domain, existing approaches and the specific contribution of the
   relevant paper(s). Also detail the drawbacks which you plan to address.
   If it's a custom project, explain your specific motivation and goals.
   Cite any other work as needed.

2. Methods (25%).  If implementing an existing paper,
   explain the original approach as well as your ideas for modifications,
   additions or improvements to the algorithm/task/domain etc., as relevant.
   Otherwise, provide a detailed explanation of your approach.
   In both cases, explain the empirical and/or theoretical motivation for what
   you are doing.
   Finally, describe the data you will be using for evaluation.

3. Implementation and experiments (20%). 
   Describe the experiments performed and their configurations, what was
   compared to what and the evaluation metrics used and why.
   Explain all implementation details such as model architectures used, data
   preprocessing/augmentation approaches, loss formulations, training methods
   and hyperparameter values.

   Note: You can use existing code, e.g. in your implementation but specify what
   you used and which parts you implemented yourself.

4. Results (20%). Present all results in an orderly table and include graphs or
   figures as you see fit.
   Discuss, analyze and explain your results.
   Compare to previous works and other approaches for your task.


## Submission

Create a zip file titled `proj-id1_id2.zip` (replace `id1`/`id2` with your
IDs) and email it to Chaim and Aviv.

The zip file should include:
1. A single PDF document, `report.pdf`, containing your project report.
   It **must** be structured according to the sections listed above.
2. A folder `src/` containing all your code.
3. A `README` file (plain text/markdown) explaining:
    1. The structure of the code in the `src/` folder: What is implemented in
       each package/module.
    2. Steps to reproduce your results using this code: Where to get and place
       the data, how to run all the data processing steps, how to run training
       and evaluation.

