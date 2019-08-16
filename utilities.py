#!/usr/bin/python3
"""
Utility methods for main programs.  This can also be used to convert GloVe
files to numpy.
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
import numpy as np
from sys import argv, exit
import os

modeldir = "models/"
def countlines(filename):
    """returns number of number of words in the file (rows) and number
    of features per word (cols)"""
    with open(filename,'r') as f:
        lines = f.readlines()
        rows = len(lines)
        cols = len(lines[0].split())-1

    return rows, cols

def readGlove(filename):
    """parses the pretrained model.  returns the list of words and 2D numpy
    array of word vectors (each row corresponds to the same index in word)"""
    rows, cols = countlines(filename)
    data = np.zeros((rows,cols))
    words = []
    with open(filename,'r') as f:
        i = 0
        for line in f:
           info = line.split()
           if(len(info) != cols+1):
               continue
           words.append(info[0])
           info = [float(info[j+1]) for j in range(len(info)-1)]
           data[i] = info
           i+=1
    return words, data

def save_glove_vectors(outfile, data, words):
    """save loaded model as a numpy file with words. word vectors,
    and vector lengths"""
    fp = open(outfile, 'wb')
    np.save(fp, words)
    np.save(fp, data)
    np.save(fp, compute_lengths(data))
    fp.close()

def load_glove_vectors(infile):
    """loads pretrained model from numpy file containg words, word vectors,
    and lengths"""
    infile = modeldir+infile+".npy"
    if not os.path.exists(infile):
        print("Error: vector file does not exist")
        print("Options are: ")
        for f in os.listdir(modeldir):
            if(os.path.splitext(f)[1] == ".npy"):
              print(os.path.splitext(f)[0])
        exit()
    fp = open(infile, 'rb')
    words = list(np.load(fp))
    vectors = np.load(fp)
    lengths = np.load(fp)
    fp.close()
    return words, vectors, lengths

def compute_lengths(npa):
    """returns length of vector"""
    return np.linalg.norm(npa,axis=1)

def cosine_similarity(vec1, len1, vec2, len2):
    """calculates the cosine similarity between two vectors"""
    dot_product = np.dot(vec1, vec2)
    return dot_product / (len1 * len2)


def closest_vectors(v, length, words, array, lengths, n):
    """
    returns most similar words
    @param v - word vector for search word
    @param length - length of v
    @param words - list of words to search
    @param array - corresponding matrix of word vectors (each row corresponds
    to the row in words)
    @param lengths - corresponding lengths of each word vector
    @param n - how many similar words to return
    @return result - a list of tuples containing the similar word and the
      similarity score.  result is sorted from most similar to nth most similar
    """
    sims = []
    for i in range(len(words)):
        sims.append(cosine_similarity(v,length,array[i],lengths[i]))
    indsort = np.argsort(sims)
    result = []
    for i in range(n):
        ind = indsort[-1-i]
        result.append((words[ind],sims[ind]))
    return result


if __name__ == "__main__":
    if len(argv) != 3:
        print("usage: utilities.py GloVeFile npyFile")
    elif argv[2][-4:] != ".npy":
        print("usage: npyFile must have an .npy extension")
    else:
        words, data = readGlove(argv[1])
        save_glove_vectors(argv[2],data,words)
