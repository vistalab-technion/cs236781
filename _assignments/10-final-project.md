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

For this project you will either implement a current deep-learning
research paper or implement a custom project that relates to your own
research topic.

We provide you with a pool of project ideas mostly from recent papers. The idea
is to select a paper that seems interesting you you, implement it, suggest at
least one improvement and evaluate it as well. Some papers may be more advanced
than others. Selecting a less advanced paper means we may expect you to also
implement more of your own extensions and ideas to improve upon it.  Also, some
projects on the list are custom projects offered by the VISTA lab. 

During the project period you will be able to meet with course TAs for guidance
during dedicated office hours.  The project can be performed in groups of at
most two students.

For questions regarding the project, please contact [Chaim
Baskin](mailto:chaimbaskin@cs.technion.ac.il).

## Project procedure

1. Each group should read the topics and paper abstracts in the project topics spreadsheet.
1. Each group must fill out the registration form with their top-3 priorities
   or custom project proposal until date **27/06/2019**.
1. Soon after you submit your priorities, we will approve one of
   them. Submitting the form sooner will increase your chance of getting your
   first priority.
1. You will start working on your projects.
   If necessary, you can schedule periodic meetings with course staff at dedicated
   office hours which will be published.
1. Your submission should be a detailed report. It should explain the problem
   and the paper you implemented (if relevant), explain your specific
   enhancements and modifications, showcase all your results (both reproduction
   of the paper and novel results, if any), compare your solution to the
   baseline method (if relevant), etc.
1. The submission date for the project is **01/10/2019**.
   Note that we can't guarantee access to the course servers after this date.

## Registration links

Please view the
[project topics](https://docs.google.com/spreadsheets/d/1vH2CsoHo65EnT053YFwEWCXRNhZDwhWm5_-fMmebTzM)
and then fill out the
[registration form](https://forms.gle/qiXMXndNABJTYPbS7) with your priorities.


## Project-related office hours

Initial office hours will be given by Chaim on:
- Thursday 13/06 at 11:00
- Sunday 16/06 after class

Both at VISTA lab, Taub 120. You can use these hours to consult regarding
project selection.

Starting on 22/08/2019 we will hold project-related office hours 
every Thursday 16:00-17:30 at the VISTA lab.
Please email if you intend to attend.

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
   explain your ideas for modifications, additions or improvements to the
   algorithm/task/domain etc., as relevant. If working on a custom project,
   provide a detailed explanation of your approach.  In both cases, explain the
   empirical and/or theoretical motivation for what you are doing.  Finally,
   describe the data you will be using for evaluation.

3. Implementation and experiments (20%). 
   Describe the experiments performed and their configurations, what was
   compared to what and the evaluation metrics used and why.
   Explain all implementation details such as model architectures used, data
   preprocessing/augmentation approaches, loss formulations, training methods
   and hyperparameter values.

   Note: You can use any existing code, e.g. code implementing all or part of
   relevant papers, in your implementation but specify what you used and
   which parts you implemented yourself.

4. Results (20%). Present all results in an orderly table and include graphs or
   figures as you see fit.
   Discuss and explain your results. If implementing an existing paper,
   compare to the original approach in the paper as. If your task
   is different than the paper, or if you are working on a custom project,
   compare to previous works in the domain solving the same or similar tasks.


## Submission

Create a zip file titled `proj-id1_id2.zip` (replace `id1`/`id2` with your
IDs) and email it to Chaim and Aviv.

The zip file should include:
1. A single PDF document, `report.pdf`, containing your project report.
   It should be structured according to the sections listed above.
2. A folder `src/` containing all your code.
3. A `README` file (plain text/markdown) explaining:
    1. The structure of the code in the `src/` folder: What is implemented in
       each package/module.
    2. Steps to reproduce your results using this code: Where to get and place
       the data, how to run all the data processing steps, how to run training
       and evaluation.

