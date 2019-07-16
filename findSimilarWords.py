#!/usr/bin/python3
"""
This program will use word embeddings to find the most similar words to a
provided search word

In order, this program takes
  - the name of the word embedding model
  - a search word
  - the number of results to display

The legal options for word embedding model are the base name of any file in
 the models folder e.g., twitter for twitter.npy

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
import sys

def main():
    if len(sys.argv) != 4:
        print("usage: findSimilarWords.py npyFile queryWord numResults")
        return

    wordvecFile = argv[1]
    queryWord = argv[2]
    nresults = int(argv[3])

    wordlist, wordvecs, lengths = load_glove_vectors(wordvecFile)
    index = wordlist.index(queryWord)
    vector = wordvecs[index]
    length = lengths[index]
    sim_words = closest_vectors(vector, length, wordlist, wordvecs, lengths, nresults+1)

    print("Printing %d most similar words to %s\n"%(nresults,queryWord))
    print("%10s %10s" % ("word","score"))
    print("-"*10+" "+"-"*10)
    for i in range(1, len(sim_words)):
        print("%10s %10.3f" % (sim_words[i]))

if __name__ == "__main__":
    main()
