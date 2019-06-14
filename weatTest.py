#!/usr/bin/python3
"""
Perform a similar test as Caliskan et al (this code does not do a
permutation test for statistical significance).  You  will provide, in order

  the name of the word embedding model (twitter, web, or wikipedia)
  the two target words (e.g, flowers and insect)
  the two attribute words (e.g., pleasant and unpleasant)

For example:
./weatTest.py twitter names_europe names_africa pleasant unpleasant

You can find the files containing the list of target/attribute words in the
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

    wrd_assoc = {}
    diffSumT1 = diffSumT2 = 0
    print()
    for target in target1:
        ta = getAverageSimilarity(target, attribute1, wordlist, array, lengths)
        tb = getAverageSimilarity(target, attribute2, wordlist, array, lengths)

        diff = ta - tb
        diffSumT1 += diff
        wrd_assoc[target] = diff


    for target in target2:
        ta = getAverageSimilarity(target, attribute1, wordlist, array, lengths)
        tb = getAverageSimilarity(target, attribute2, wordlist, array, lengths)
        diff = ta - tb
        diffSumT2 += diff
        wrd_assoc[target] = diff

    d = (diffSumT1/len(target1) - diffSumT2/len(target2))/np.std(list(wrd_assoc.values()))

    print("Effect size: %.2f" % d)
    print("The score is between +2.0 and -2.0.  Positive scores indicate that")
    print("%s is more associated with %s than %s." % (target1Name, attr1Name, target2Name))
    print("Or, equivalently, %s is more associated with %s than %s." % (target2Name, attr2Name, target1Name))
    print("Negative scores have the opposite relationship.")
    print("Scores close to 0 indicate little no effect.")


main()
