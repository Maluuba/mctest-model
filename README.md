# [A Parallel-Hierarchical Model for Machine Comprehension on Sparse Data](http://arxiv.org/abs/1603.08884)

Adam Trischler, Zheng Ye, Xingdi Yuan, Jing He, Philip Bachman, Kaheer Suleman

# Introduction
Understanding unstructured text is a major goal within natural language processing.
Comprehension tests pose questions based on short text passages to evaluate such understanding.
In this work, we investigate machine comprehension on the challenging *MCTest* benchmark.
Partly because of its limited size, prior work on *MCTest* has focused mainly on engineering better features.
We tackle the dataset with a neural approach, harnessing simple neural networks arranged in a parallel hierarchy.
The parallel hierarchy enables our model to compare the passage, question, and answer from a variety of trainable perspectives,
as opposed to using a manually designed, rigid feature set.
Perspectives range from the word level to sentence fragments to sequences of sentences;
the networks operate only on word-embedding representations of text.
When trained with a methodology designed to help cope with limited training data,
our Parallel-Hierarchical model sets a new state of the art for *MCTest*,
outperforming previous feature-engineered approaches slightly
and previous neural approaches by a significant margin (over 15% absolute).

### Citing the paper

If you find this paper useful in your research, please cite:

    @article{adam16,
        Author = {Adam Trischler, Zheng Ye, Xingdi Yuan, Jing He, Philip Bachman, Kaheer Suleman},
        Title = {A Parallel-Hierarchical Model for Machine Comprehension on Sparse Data},
        Journal = {arXiv preprint arXiv:1603.08884 [cs.CL]},
        Year = {2016}
    }


## Requirement
* Keras1.x
* Theano 0.8

## How to run a test
* Download Google's pre-trained word2vec (`.bin.gz`), and convert it to h5 format using the script `embedding_2_h5.py`.
* Open a terminal and input `python run.py`. The results on test would be around 70% accuracy.
* If you want to test different settings, you can change the parameters in `model.yaml`.

# License
## The MIT License (MIT)

Copyright (c) 2016 Maluuba Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
