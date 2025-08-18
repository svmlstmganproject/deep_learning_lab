# Introduction
# :label:`chap_introduction`

# Until recently, nearly every computer program
# that you might have interacted with during
# an ordinary day
# was coded up as a rigid set of rules
# specifying precisely how it should behave.
# Say that we wanted to write an application
# to manage an e-commerce platform.
# After huddling around a whiteboard
# for a few hours to ponder the problem,
# we might settle on the broad strokes
# of a working solution, for example:
# (i) users interact with the application through an interface
# running in a web browser or mobile application;
# (ii) our application interacts with a commercial-grade database engine
# to keep track of each user's state and maintain records
# of historical transactions;
# and (iii) at the heart of our application,
# the *business logic* (you might say, the *brains*) of our application
# spells out a set of rules that map every conceivable circumstance
# to the corresponding action that our program should take.

# To build the brains of our application,
# we might enumerate all the common events
# that our program should handle.
# For example, whenever a customer clicks
# to add an item to their shopping cart,
# our program should add an entry
# to the shopping cart database table,
# associating that user's ID
# with the requested product's ID.
# We might then attempt to step through
# every possible corner case,
# testing the appropriateness of our rules
# and making any necessary modifications.
# What happens if a user
# initiates a purchase with an empty cart?
# While few developers ever get it
# completely right the first time
# (it might take some test runs to work out the kinks),
# for the most part we can write such programs
# and confidently launch them
# *before* ever seeing a real customer.
# Our ability to manually design automated systems
# that drive functioning products and systems,
# often in novel situations,
# is a remarkable cognitive feat.
# And when you are able to devise solutions
# that work $100\%$ of the time,
# you typically should not be
# worrying about machine learning.

# Fortunately for the growing community
# of machine learning scientists,
# many tasks that we would like to automate
# do not bend so easily to human ingenuity.
# Imagine huddling around the whiteboard
# with the smartest minds you know,
# but this time you are tackling
# one of the following problems:

# * Write a program that predicts tomorrow's weather given geographic information, satellite images, and a trailing window of past weather.
# * Write a program that takes in a factoid question, expressed in free-form text, and  answers it correctly.
# * Write a program that, given an image, identifies every person depicted in it and draws outlines around each.
# * Write a program that presents users with products that they are likely to enjoy but unlikely, in the natural course of browsing, to encounter.

# For these problems,
# even elite programmers would struggle
# to code up solutions from scratch.
# The reasons can vary.
# Sometimes the program that we are looking for
# follows a pattern that changes over time,
# so there is no fixed right answer!
# In such cases, any successful solution
# must adapt gracefully to a changing world.
# At other times, the relationship (say between pixels,
# and abstract categories) may be too complicated,
# requiring thousands or millions of computations
# and following unknown principles.
# In the case of image recognition,
# the precise steps required to perform the task
# lie beyond our conscious understanding,
# even though our subconscious cognitive processes
# execute the task effortlessly.

# *Machine learning* is the study of algorithms
# that can learn from experience.
# As a machine learning algorithm accumulates more experience,
# typically in the form of observational data
# or interactions with an environment,
# its performance improves.
# Contrast this with our deterministic e-commerce platform,
# which follows the same business logic,
# no matter how much experience accrues,
# until the developers themselves learn and decide
# that it is time to update the software.

# In this book, we will teach you
# the fundamentals of machine learning,
# focusing in particular on *deep learning*,
# a powerful set of techniques
# driving innovations in areas as diverse as computer vision,
# natural language processing, healthcare, and genomics.

# ## A Motivating Example

# Before beginning writing, the authors of this book,
# like much of the work force, had to become caffeinated.
# We hopped in the car and started driving.
# Using an iPhone, Alex called out "Hey Siri",
# awakening the phone's voice recognition system.
# Then Mu commanded "directions to Blue Bottle coffee shop".
# The phone quickly displayed the transcription of his command.
# It also recognized that we were asking for directions
# and launched the Maps application (app)
# to fulfill our request.
# Once launched, the Maps app identified a number of routes.
# Next to each route, the phone displayed a predicted transit time.
# While this story was fabricated for pedagogical convenience,
# it demonstrates that in the span of just a few seconds,
# our everyday interactions with a smart phone
# can engage several machine learning models.

# Imagine just writing a program to respond to a *wake word*
# such as "Alexa", "OK Google", and "Hey Siri".
# Try coding it up in a room by yourself
# with nothing but a computer and a code editor,
# as illustrated in :numref:`fig_wake_word`.
# How would you write such a program from first principles?
# Think about it... the problem is hard.
# Every second, the microphone will collect roughly
# 44,000 samples.
# Each sample is a measurement of the amplitude of the sound wave.
# What rule could map reliably from a snippet of raw audio to confident predictions
# $\{\textrm{yes}, \textrm{no}\}$
# about whether the snippet contains the wake word?
# If you are stuck, do not worry.
# We do not know how to write such a program from scratch either.
# That is why we use machine learning.

# ![Identify a wake word.](../img/wake-word.svg)
# :label:`fig_wake_word`

# Here is the trick.
# Often, even when we do not know how to tell a computer
# explicitly how to map from inputs to outputs,
# we are nonetheless capable of performing the cognitive feat ourselves.
# In other words, even if you do not know
# how to program a computer to recognize the word "Alexa",
# you yourself are able to recognize it.
# Armed with this ability, we can collect a huge *dataset*
# containing examples of audio snippets and associated labels,
# indicating which snippets contain the wake word.
# In the currently dominant approach to machine learning,
# we do not attempt to design a system
# *explicitly* to recognize wake words.
# Instead, we define a flexible program
# whose behavior is determined by a number of *parameters*.
# Then we use the dataset to determine the best possible parameter values,
# i.e., those that improve the performance of our program
# with respect to a chosen performance measure.

# You can think of the parameters as knobs that we can turn,
# manipulating the behavior of the program.
# Once the parameters are fixed, we call the program a *model*.
# The set of all distinct programs (input--output mappings)
# that we can produce just by manipulating the parameters
# is called a *family* of models.
# And the "meta-program" that uses our dataset
# to choose the parameters is called a *learning algorithm*.

# Before we can go ahead and engage the learning algorithm,
# we have to define the problem precisely,
# pinning down the exact nature of the inputs and outputs,
# and choosing an appropriate model family.
# In this case,
# our model receives a snippet of audio as *input*,
# and the model
# generates a selection among
# $\{\textrm{yes}, \textrm{no}\}$ as *output*.
# If all goes according to plan
# the model's guesses will
# typically be correct as to
# whether the snippet contains the wake word.

# If we choose the right family of models,
# there should exist one setting of the knobs
# such that the model fires "yes" every time it hears the word "Alexa".
# Because the exact choice of the wake word is arbitrary,
# we will probably need a model family sufficiently rich that,
# via another setting of the knobs, it could fire "yes"
# only upon hearing the word "Apricot".
# We expect that the same model family should be suitable
# for "Alexa" recognition and "Apricot" recognition
# because they seem, intuitively, to be similar tasks.
