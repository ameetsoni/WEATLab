## PREREQUISITE SKILLS

You need to know how to use the command-line in order to run these programs.
No programming skills are required.  These programs use `python3` and were tested
on Ubuntu systems.

-------------

## SETUP

1) Download your word embedding models of interest from
https://nlp.stanford.edu/projects/glove/

You can generate new models using the GloVe software, or download the pre-trained
word vectors.

2) Run `utilities.py`

  ./utilities.py GloVeFile npyFile

where `GloVeFile` is the word vector model downloaded/created in the above step.
The numpy version of this will be named `npyFile`.  The output file should be
placed in a subdirectory called `models` to be properly accessed by the main programs.


-------------------------
## MAIN PROGRAMS

There are two programs:
* `findSimilarWords.py` - searches for the most related words to a query word
using cosine similarity.
* `weatTest.py` - performs the Word Embedding Association Tests described in:

"Semantics derived automatically from language corpora contain human-like biases."
Caliskan, Aylin; Bryson, Joanna J; Narayanan, Arvind. *Science*, Vol. 356,
No. 6334, 14.04.2017, p. 183-186.

-------------------------

### `findSimilarWords.py`

This program will use word embeddings to find the most similar words to a
provided search word.  For the purpose of this assignment, this program is to
designed to demonstrate the useful aspect of word embeddings (i.e., it is
  good at finding related words).

In order, this program takes
  - the name of the word embedding model
  - a search word
  - the number of results to display

The legal options for word embedding model are those created in the [Setup](#setup)
step.  E.g., `twitter` will load the file `models/twitter.npy`.

here is an example run

  ./findSimilarWords.py twitter cat 5

This will search for the top 5 most similar words to cat based on word embedding
model trained on the Twitter corpus.

  $ ./findSimilarWords.py twitter cat 5
  Printing 5 most similar words to cat

        word      score
  ---------- ----------
         dog      0.943
         pet      0.873
       kitty      0.868
        bear      0.866
      monkey      0.851

--------------------

### `weatTest.py`

This will perform a similar test as the Caliskan et al. paper to find assocations
 between concepts in our learned models.  You  will provide, in order
  - the name of the word embedding model (e.g., twitter, web, or wikipedia)
  - the two *target* words (e.g, flowers and insect)
  - the two *attribute* words (e.g., pleasant and unpleasant)

For example:

  ./weatTest.py twitter names_europe names_africa pleasant unpleasant

This will use the word embedding vectors trained on Twitter created in the
 [Setup](#setup) step (i.e., `models/twitter.npy`).  
You can find the files containing the list of target/attribute words in your
 `wordlists` directory.  E.g., `names_europe` will use the words in `wordlists/names_europe.txt`.

 The possible concepts for targets or attributes must be drawn from this list:

- `names_male`
- `art`
- `career`
- `gender_m`
- `flowers`
- `family`
- `pleasant`
- `names_africa`
- `names_europe`
- `gender_f`
- `insect`
- `unpleasant`
- `names_female`
- `science`

To add a list, create a file in `wordlist/` with a `txt` extension.  The base part
of the file name can be added to the list above (e.g. `religion.txt` can be
invoked as `religion` for either the attribute or target concept).

When run, the program will print a warning for each word not in the word vector
model (this varies based on data sets, particularly if your word list contains
names).  Then, it will output the similarity scores between each pair
of target and attributes and their differences(e.g., names_europe/pleasant vs
names_europe/unpleasant; names_africa/pleasant vs names_africa/unpleasant).

In the example below, European names are more closely associated with pleasant
words, while African names are more closely associated with unpleasant names.
The values have been scaled by 100 for readability purposes; a permutation
test is omitted from this program (but shown in the original paper).

  $ ./weatTest.py twitter names_europe names_africa pleasant unpleasant
  Warning: omitting bobbie-sue, no word vector exists in the GloVe index
  Warning: omitting sue-ellen, no word vector exists in the GloVe index
  Warning: omitting percell, no word vector exists in the GloVe index
  Warning: omitting everol, no word vector exists in the GloVe index
  Warning: omitting terryl, no word vector exists in the GloVe index
  Warning: omitting wardell, no word vector exists in the GloVe index
  Warning: omitting lashelle, no word vector exists in the GloVe index
  Warning: omitting temeka, no word vector exists in the GloVe index
  Warning: omitting tameisha, no word vector exists in the GloVe index
  Warning: omitting teretha, no word vector exists in the GloVe index
  Warning: omitting sharise, no word vector exists in the GloVe index
  Warning: omitting tashika, no word vector exists in the GloVe index
  Warning: omitting lashandra, no word vector exists in the GloVe index
  Warning: omitting shavonn, no word vector exists in the GloVe index

  Positive differences mean the target correlates more with pleasant
  Negative differences correlate more with unpleasant

  Simlarity between names_europe and pleasant: 31.354
  Simlarity between names_europe and unpleasant: 26.379
  Difference: 4.975

  Simlarity between names_africa and pleasant: 8.856
  Simlarity between names_africa and unpleasant: 12.042
  Difference: -3.186
