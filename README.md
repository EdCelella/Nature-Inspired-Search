# Nature-Inspired Search Solutions to the Travelling Salesman Problem

This project implements three different search algorithms, and applies them to the travelling salesman problem. The algorithms which have been implemented are:

1. Simulated Annealing.
2. Tabu Search.
3. Evolutionary Algorithm.

A description of each algorithm, as well as its implementation, can be found in the Report directory. The report itself can be found as a markdown and PDF file. It should be noted that the markdown version uses Mathjax, thus a markdown editor/viewer that can parse Mathjax (e.g. [Typora](https://typora.io)) is required. The report also contains results and analysis for each algorithm when run for 30 times with 3000 iterations.

This project was submitted as coursework for my MSc Advanced Computer Science and achieved a grade of 100%.

## Prerequisites

Python 3.8

- This project was built using Python 3.8, and so is only guaranteed to work with 3.8 and above. However, any version of Python 3 should run the code.

## How To Use

Simply run the [nis\_project.py](Code/nis_project.py) file found in the Code directory in a python IDE, or via the command line:

```
python nis_project.py
```

The program attempts to solve the travelling salesman problem specified in the file [ATT48.tsp](Code/ATT48.tsp), in which the optimal route is given in the file [att48.opt.tour](Code/att48.opt.tour). Although the program has been built only to solve this specific travelling salesman problem, if different files containing a travelling salesman problem are provided in the same format (and file names changed in the code), then the program should solve the given problem.

## License

This project is licensed under the terms of the [Creative Commons Attribution 4.0 International Public license](License.md).

