---
title: Detecting Bias in Language Models
---

# Detecting Bias in Language Models
=================================


  ### Summary ###
  This module introduces the idea of word embedding models and how they encode cultural biases. Students learn about one tool, Word Embedding Association Test, for uncovering these biases. Students apply their understanding to analyze the ethical issues that arise in real-world applications of machine learning.

  ### Topics ###
  Ethics in AI, Natural Language Processing, Machine Learning, Word Embeddings

  ### Audience ###
  Suitable for any CS course. Ideal for Introduction to AI, Machine Learning, NLP, or Intro CS. Can be used in any course looking to introduce ideas of ethical issues and bias in algorithms.

  ### Difficulty ###
  Very low technical difficulty - no programming experience is required although students must be comfortable using the command-line interface. The main difficulty for students is in learning how to frame the ethical issues that arise in the prompts. The entire module can be completed 3-4 hours, of which one or two hours is done in lab/lecture with the instructor.

  ### Strengths ###      
  The assignment is simple to set up on the backend. No coding is required, though it would be very easy to add some. Students are able to take complex mathematical models of bias and apply them to tangible examples. The exercises are straightforward and students are able to connect real-life examples to abstract philosophical discussions.
  **Weaknesses**     As with most ethical discussions, the questions that arise are complex and can frustrate students who desire a straight-forward answer. This assignment is an introduction to the topic and focuses more on identifying ethical issues and bias in algorithms, and not the question of how to address them.

  ### Dependencies ###
  The software uses Python and standard libraries (pandas, seaborn, matplotlib, numpy). Students do need to learn how to use the command-line to run the programs, but no programming experience is required (the assignment was developed for a first-year seminar in Philosophy with no pre-requisites). It would benefit students to use this assignment after introducing topics such as bias in algorithms and/or ethics, though this is not required.

  ### Variants ###
  This assignment can be adapted for more technical audiences. For example, students could implement the search algorithms and statistical tests given the original paper. Additional ethical case studies could be added and tailored to course content (e.g., facial recognition technology in hospitals for a computer vision course).


## Reading
\"[Semantics derived automatically from language corpora
contain human-like
biases](https://science.sciencemag.org/content/356/6334/183.full).\"
Aylin Caliskan, Joanna J. Bryson, and Arvind Narayanan. *Science*, Vol.
356, No. 6334, 2017, p. 183-186.

## Files

* [Source code](student_materials/) Files to be given to students,
including a `README.md`, software, and data.

 - [README](student_materials/README.md) - list of prerequisites and
    examples for running each program.
 - [findSimilarWords.py](student_materials/findSimilarWords.py) - main
    program that searches for the most related words to a query word
    using cosine similarity. This is used to show the usefulness of word
    embedding models
 - [weatTest.py](student_materials/weatTest.py) - main program that
    implements the bias test from Caliskan, Bryson, and Narayan.
    Students use this to design an experiment that tests for bias in the
    word embedding models.
 - [wordlists](student_materials/wordlists/) - word lists from the
    paper that were used to test for bias. Students/instructors can add
    new word lists or modify existing ones to develop new bias tests.

* [Lecture Notes](instructor_materials/lecture_slides.pdf) to motivate
assignment. A [Powerpoint
(pptx)](instructor_materials/lecture_slides.pptx) is also provided to
allow for editing of slides.

* [Assignment Handout](assignment.pdf) with both lab instructions (how to
use the provided code) in addition assignment instructions (writing
prompts). Note that the lab instructions correspond to the two prompts
in the lecture slides. The [source (tex)](assignment.tex) is also
available.
