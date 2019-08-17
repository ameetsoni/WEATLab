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
 wordlists directory.  The program also provides a visualization of the
 similarities between the targets and attributes
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
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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
    omits = []
    with open(datadir+filename+".txt",'r') as f:
        words = []
        for w in f.readlines():
            toInsert = w.strip().lower()
            if toInsert in reference:
                words.append(toInsert)
            else:
                omits.append(toInsert)
        if len(omits):
            print("Warning: the following words from %s are not in the GloVe index: %s" % (filename, ', '.join(omits)))
    return words

def getAverageSimilarity(targetVec, targetLength, attrVectors, attrLengths):
    """Calculates average similarity between words in the target list and attribute
    list"""
    return np.average([cosine_similarity(targetVec, targetLength, attrVectors[i], attrLengths[i]) for i in range(attrVectors.shape[0])])

def getListData(conceptWords, allWords, allArray, allLengths):
    """Return the GloVe Vectors and Lengths for a subset of conceptWords"""
    inds = [allWords.index(target) for target in conceptWords]
    return (allArray[inds], allLengths[inds])

def rankAttributes(targetData, targetLengths, attrData, attrLengths, attrWords, n= 5):
    """Return the n highest similarity scores between the target and all attr"""
    attrSims = [getAverageSimilarity(attrData[i], attrLengths[i], targetData, targetLengths) for i in range(attrData.shape[0])]
    return attrWords[np.argsort(attrSims)[-n:]]

def main():
    if len(argv) != 6:
        print("usage: weatTest.py npyFile target1file target2file attribute1file attribute2file")
        print("   e.g., weatTest.py twitter flowers insect pleasant unpleasant")
        return

    #parse inputs, load in glove vectors and wordlists
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
        print("Error loadding one of the word lists; lists are either empty")
        print(" or not in your learned word embedding data set")
        return

    target1Data, target1Lengths = getListData(target1, wordlist, array, lengths)
    target2Data, target2Lengths = getListData(target2, wordlist, array, lengths)
    attr1Data, attr1Lengths = getListData(attribute1, wordlist, array, lengths)
    attr2Data, attr2Lengths = getListData(attribute2, wordlist, array, lengths)


    #Find more similar attribute words for each target list
    print()
    print("Top 5 most similar attribute words to %s:" % target1Name)
    topWordsT1 = rankAttributes(target1Data, target1Lengths, np.concatenate([attr1Data, attr2Data]),
            np.concatenate([attr1Lengths, attr2Lengths]), np.concatenate([attribute1, attribute2]))
    for word in topWordsT1:
        print("\t"+word)

    print()
    print("Top 5 most similar attribute words to %s:" % target2Name)

    topWordsT2 = rankAttributes(target2Data, target2Lengths, np.concatenate([attr1Data, attr2Data]),
            np.concatenate([attr1Lengths, attr2Lengths]), np.concatenate([attribute1, attribute2]))
    for word in topWordsT2:
        print("\t"+word)

    print()
    #calculate similarities between target 1 and both attributes
    targ1attr1Sims = [getAverageSimilarity(target1Data[i], target1Lengths[i], attr1Data, attr1Lengths)
        for i in range(target1Data.shape[0])]
    targ1attr2Sims = [getAverageSimilarity(target1Data[i], target1Lengths[i], attr2Data, attr2Lengths)
        for i in range(target1Data.shape[0])]
    targ1SimDiff = np.subtract(targ1attr1Sims, targ1attr2Sims)

    #calculate similarities between target 2 and both attributes
    targ2attr1Sims = [getAverageSimilarity(target2Data[i], target2Lengths[i], attr1Data, attr1Lengths)
        for i in range(target2Data.shape[0])]
    targ2attr2Sims = [getAverageSimilarity(target2Data[i], target2Lengths[i], attr2Data, attr2Lengths)
        for i in range(target2Data.shape[0])]
    targ2SimDiff = np.subtract(targ2attr1Sims, targ2attr2Sims)

    #effect size is avg difference in similarities divided by standard dev
    d = (np.average(targ1SimDiff) - np.average(targ2SimDiff))/np.std(np.concatenate((targ1SimDiff,targ2SimDiff)))


    print()
    print("Calculating effect size.  The score is between +2.0 and -2.0.  ")
    print("Positive scores indicate that %s is more associated with %s than %s." % (target1Name, attr1Name, target2Name))
    print("Or, equivalently, %s is more associated with %s than %s." % (target2Name, attr2Name, target1Name))
    print("Negative scores have the opposite relationship.")
    print("Scores close to 0 indicate little to no effect.")
    print()
    print("Effect size: %.2f" % d)

    print()
    print("Plotting similarity scores...")

    fig = plt.figure(figsize=(16,8))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.set_title("Similarities Scores for Target/Attribute Pairs")
    ax2.set_title("Difference Scores For Each Target")

    # Box plot of pairwise similarity scores
    df = pd.DataFrame()
    df["Similarity"] = np.concatenate([targ1attr1Sims,targ1attr2Sims,targ2attr1Sims,targ2attr2Sims])
    df["Pairs"] = [target1Name+"-"+attr1Name]*len(targ1attr1Sims)+[target1Name+"-"+attr2Name]*len(targ1attr2Sims)+[target2Name+"-"+attr1Name]*len(targ2attr1Sims) \
      +[target2Name+"-"+attr2Name]*len(targ2attr2Sims)
    df["Target"] = [target1Name]*len(targ1attr1Sims+targ1attr2Sims)+[target2Name]*len(targ2attr1Sims+targ2attr2Sims)
    df["Attribute"] = [attr1Name]*len(targ1attr1Sims)+[attr2Name]*len(targ1attr2Sims)+[attr1Name]*len(targ2attr1Sims) +[attr2Name]*len(targ2attr2Sims)
    sns.boxplot(x="Target", y="Similarity", hue="Attribute",data=df, ax=ax1)
    #Box plot of target bias in similarities
    df = pd.DataFrame()
    df["Difference"] = np.concatenate([targ1SimDiff, targ2SimDiff])
    df["Target"] = [target1Name]*len(targ1SimDiff) + [target2Name]*len(targ2SimDiff)
    ax = sns.boxplot(x="Target", y="Difference", data=df, ax=ax2)

    
    ticks = ax1.get_yticks()
    mx = max(abs(ticks[0]),ticks[-1])
    mx = int(mx*10+.99)/10.0
    ax1.yaxis.set_ticks(np.arange(-mx,mx+.1,.1))
    ticks = ax2.get_yticks()
    mx = max(abs(ticks[0]),ticks[-1])
    mx = int(mx*10+.99)/10.0
    ax2.yaxis.set_ticks(np.arange(-mx,mx+.1,.1))

    fig.subplots_adjust(wspace=0.5)
    fig.canvas.draw()

    labels = [item.get_text() for item in ax1.get_yticklabels()]
    labels[0] = "(less similar) " + labels[0]
    labels[-1] = "(more similar) " + labels[-1]
    ax1.set_yticklabels(labels)
    labels = [item.get_text() for item in ax2.get_yticklabels()]
    labels[0] = "(%s) " % attr2Name + labels[0]
    labels[len(labels)//2] = "(neutral) 0.0"
    labels[-1] = "(%s) " % attr1Name + labels[-1]
    ax2.set_yticklabels(labels)
    
    plt.show()


if __name__ == "__main__":
    main()
