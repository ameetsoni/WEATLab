#!/usr/bin/python3
"""
Perform a similar test as Caliskan et al (this code does not do a
permutation test for statistical significance).  You  will provide, in order

  the name of the word embedding model (twitter, web, or wikipedia)
  the two target words (e.g, flowers and insect)
  the two attribute words (e.g., pleasant and unpleasant)

For example:
./weatTest.py twitter names_europe names_africa pleasant unpleasant

You can find the files containing the list of target/attribute words in your
 wordlists directory.
Copyright (C) 2019  Ameet Soni, Swarthmore College
Email: asoni1@swarthmore.edu

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
from utilities import *
import numpy as np
import os
import sys

datadir = "wordlists/"

def loadwordlist(filename, reference):
    """loads and returns all words in the given file.  Omits words not in the
    word embedding matrix"""
    if not os.path.exists(datadir+filename+".txt"):
        print("Illegal target or attribute: %s" % filename)
        print("Options are: ")
        for f in os.listdir(datadir):
            print(os.path.splitext(f)[0])
        sys.exit()
    with open(datadir+filename+".txt",'r') as f:
        words = []
        for w in f.readlines():
            toInsert = w.strip().lower()
            if toInsert in reference:
                words.append(toInsert)
            else:
                print("Warning: omitting %s, no word vector exists in the GloVe index" % toInsert)
    return words

def getAverageSimilarity(target, attribute, allWords, array, lengths):
    """Calculates average similarity between words in the target list and attribute
    list"""
    subArray = []
    subLengths = []
    sumSim = 0
    found = 0
    indexT = allWords.index(target)
    for sub in attribute:
        found +=1
        indexS = allWords.index(sub)
        sumSim += cosine_similarity(array[indexT], lengths[indexT], array[indexS],lengths[indexS])
    return sumSim/found


def main():
    if len(argv) != 6:
        print("usage: weatTest.py npyFile target1file target2file attribute1file attribute2file")
        print("   e.g., weatTest.py twitter.npy flowers.txt insect.txt pleasent.txt unpleasant.txt")
        return
    wordlist, array, lengths = load_glove_vectors(argv[1])
    target1Name = argv[2]
    target2Name = argv[3]
    attr1Name = argv[4]
    attr2Name = argv[5]
    target1 = loadwordlist(target1Name,wordlist)
    target2 = loadwordlist(target2Name,wordlist)

    attribute1 = loadwordlist(attr1Name,wordlist)
    attribute2 = loadwordlist(attr2Name,wordlist)
    if not (target1 and target2 and attribute1 and attribute2):
        return

    score1a = score1b = score2a = score2b = 0
    print()
    for target in target1:
        score1a += getAverageSimilarity(target, attribute1, wordlist, array, lengths)
        score1b += getAverageSimilarity(target, attribute2, wordlist, array, lengths)


    for target in target2:
        score2a += getAverageSimilarity(target, attribute1, wordlist, array, lengths)
        score2b += getAverageSimilarity(target, attribute2, wordlist, array, lengths)

    #This is not in the paper, but some adjustments to make the raw differences
    # easier to compare
    score1a /= len(target1)
    score1b /= len(target1)
    score2a /= len(target2)
    score2b /= len(target2)
    score1a *= 100
    score1b *= 100
    score2a *= 100
    score2b *= 100

    print("Positive differences mean the target correlates more with %s" % attr1Name)
    print("Negative differences correlate more with %s" % attr2Name)
    print()

    print("Simlarity between %s and %s: %.3f" % (target1Name, attr1Name, score1a))
    print("Simlarity between %s and %s: %.3f" % (target1Name, attr2Name, score1b))
    print("Difference: %.3f" % (score1a-score1b))
    print()
    print("Simlarity between %s and %s: %.3f" % (target2Name, attr1Name, score2a))
    print("Simlarity between %s and %s: %.3f" % (target2Name, attr2Name, score2b))
    print("Difference: %.3f" % (score2a-score2b))


main()
