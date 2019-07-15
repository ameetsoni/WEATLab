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
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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
    with open(datadir+filename+".txt",'r') as f:
        words = []
        for w in f.readlines():
            toInsert = w.strip().lower()
            if toInsert in reference:
                words.append(toInsert)
            else:
                print("Warning: omitting %s, no word vector exists in the GloVe index" % toInsert)
    return words

def getAverageSimilarity(targetVec, targetLength, attrVectors, attrLengths):
    """Calculates average similarity between words in the target list and attribute
    list"""
    return np.average([cosine_similarity(targetVec, targetLength, attrVectors[i], attrLengths[i]) for i in range(attrVectors.shape[0])])

def getListData(conceptWords, allWords, allArray, allLengths):
    inds = [allWords.index(target) for target in conceptWords]
    return (allArray[inds], allLengths[inds])

def rankAttributes(targetData, targetLengths, attrData, attrLengths, attrWords, n= 5):
    attrSims = [getAverageSimilarity(attrData[i], attrLengths[i], targetData, targetLengths) for i in range(attrData.shape[0])]
    return attrWords[np.argsort(attrSims)[-n:]]

def main():
    if len(argv) != 6:
        print("usage: weatTest.py npyFile target1file target2file attribute1file attribute2file")
        print("   e.g., weatTest.py twitter flowers insect pleasant unpleasant")
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

    target1Data, target1Lengths = getListData(target1, wordlist, array, lengths)
    target2Data, target2Lengths = getListData(target2, wordlist, array, lengths)
    attr1Data, attr1Lengths = getListData(attribute1, wordlist, array, lengths)
    attr2Data, attr2Lengths = getListData(attribute2, wordlist, array, lengths)


    feat_cols = [ 'dim'+str(i) for i in range(target1Data.shape[1]) ]

    #df = pd.DataFrame(np.concatenate([target1Data, target2Data, attr1Data, attr2Data]),columns=feat_cols)
    #df['y'] = [target1Name]*target1Data.shape[0]+[target2Name]*target2Data.shape[0]+[attr1Name]*attr1Data.shape[0]+[attr2Name]*attr2Data.shape[0]
    df = pd.DataFrame(np.concatenate([attr1Data, attr2Data]),columns=feat_cols)
    df['y'] = [attr1Name]*attr1Data.shape[0]+[attr2Name]*attr2Data.shape[0]
    df['label'] = df['y'].apply(lambda i: str(i))
    pca = PCA(n_components=3)
    pca.fit(df[feat_cols].values)
    pca_result = pca.transform(df[feat_cols].values)
    plt.figure()
    sns.kdeplot(pca_result[df['y'] == attr1Name,0], shade=True, legend=True)
    sns.kdeplot(pca_result[df['y'] == attr2Name,0], shade=True, legend=True)
    plt.legend(labels=[attr1Name, attr2Name])
    #sns.distplot(pca_result[df['y'] == attr2Name,0], hist=False, color="y", kde_kws={"shade": True})
    #plt.show()

    df = pd.DataFrame(np.concatenate([target1Data, target2Data]),columns=feat_cols)
    df['y'] = [target1Name]*target1Data.shape[0]+[target2Name]*target2Data.shape[0]#+[attr1Name]*attr1Data.shape[0]+[attr2Name]*attr2Data.shape[0]
    df['label'] = df['y'].apply(lambda i: str(i))
    pca_result = pca.transform(df[feat_cols].values)


    df['pca-one'] = pca_result[:,0]
    df['pca-two'] = np.zeros(pca_result[:,1].shape)
    df['pca-three'] = pca_result[:,2]
    print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))


    plt.figure()
    #sns.set(style="white", palette="muted", color_codes=True)
    sns.kdeplot(pca_result[df['y'] == target1Name,0], shade=True, legend=True)
    sns.kdeplot(pca_result[df['y'] == target2Name,0], shade=True, legend=True)
    plt.legend(labels=[target1Name, target2Name])

    # sns.scatterplot(
    # x="pca-one", y="pca-two",
    # hue="y",
    # palette=sns.color_palette("hls", 2),
    # data=df,
    # legend="full",
    # alpha=0.3
    # )
    #plt.show()
    """"
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(df[feat_cols].values)
    df['tsne-2d-one'] = tsne_results[:,0]
    df['tsne-2d-two'] = tsne_results[:,1]
    plt.figure(figsize=(16,10))
    sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="y",
    palette=sns.color_palette("hls", 4),
    data=df,
    legend="full",
    alpha=0.3
    )
    plt.show()
    """
    wrd_assoc = {}
    diffSumT1 = diffSumT2 = 0
    print()
    t1a1 = []
    t1a2 = []
    t2a1 = []
    t2a2 = []
    t1diff = []
    t2diff = []
    for i in range(target1Data.shape[0]):
        ta = getAverageSimilarity(target1Data[i], target1Lengths[i], attr1Data, attr1Lengths)
        tb = getAverageSimilarity(target1Data[i], target1Lengths[i], attr2Data, attr2Lengths)
        t1a1.append(ta)
        t1a2.append(tb)
        diff = ta - tb
        t1diff.append(diff)
        diffSumT1 += diff
        wrd_assoc[target1[i]] = diff


    for i in range(target2Data.shape[0]):
        ta = getAverageSimilarity(target2Data[i], target2Lengths[i], attr1Data, attr1Lengths)
        tb = getAverageSimilarity(target2Data[i], target2Lengths[i], attr2Data, attr2Lengths)
        t2a1.append(ta)
        t2a2.append(tb)
        diff = ta - tb
        t2diff.append(diff)
        diffSumT2 += diff
        wrd_assoc[target2[i]] = diff

    d = (diffSumT1/len(target1) - diffSumT2/len(target2))/np.std(list(wrd_assoc.values()))
    plt.figure()
    feats = [target1Name+"-"+attr1Name]*len(t1a1)+[target1Name+"-"+attr2Name]*len(t1a2)+[target2Name+"-"+attr1Name]*len(t2a1) \
      +[target2Name+"-"+attr2Name]*len(t2a2)
    df = pd.DataFrame()
    df["similarity"] = np.concatenate([t1a1,t1a2,t2a1,t2a2])
    df["pairs"] = feats
    df["target"] = [target1Name]*len(t1a1+t1a2)+[target2Name]*len(t2a1+t2a2)
    df["attr"] = [attr1Name]*len(t1a1)+[attr2Name]*len(t1a2)+[attr1Name]*len(t2a1) +[attr2Name]*len(t2a2)
    sns.violinplot(x="target", y="similarity", hue="attr",data=df, split=True, inner="box")
    plt.figure()
    feats = [target1Name]*len(t1diff) + [target2Name]*len(t2diff)
    df = pd.DataFrame()
    df["difference"] = np.concatenate([t1diff, t2diff])
    df["target"] = feats
    ax = sns.violinplot(x="target", y="difference", data=df, split=True)
    #ax.set_ylim(-1,1)

    plt.show()

    print("Effect size: %.2f" % d)
    print("The score is between +2.0 and -2.0.  Positive scores indicate that")
    print("%s is more associated with %s than %s." % (target1Name, attr1Name, target2Name))
    print("Or, equivalently, %s is more associated with %s than %s." % (target2Name, attr2Name, target1Name))
    print("Negative scores have the opposite relationship.")
    print("Scores close to 0 indicate little to no effect.")
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


main()
