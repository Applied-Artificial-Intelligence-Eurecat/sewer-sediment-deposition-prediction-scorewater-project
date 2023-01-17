**USE [GUIDELINES](GUIDELINES.md) to use the template following the steps.**

# Name of the Project

Populate this *README* file with the following sections.  

Update the badge [![Unit tests](https://github.com/Applied-Artificial-Intelligence-Eurecat/markov-chain-example/actions/workflows/test.yaml/badge.svg)](https://github.com/Applied-Artificial-Intelligence-Eurecat/markov-chain-example/actions/workflows/test.yaml)

- Provide a descriptive project name that makes others easily understand what the project is about.
- Include in this section a one-sentence description.
- Include bellow a detailed description of what the project does.


# Dependencies, Installation, and Usage

```
If the software requires 3rd-party dependencies make a list of it.   
May be useful to include external links to the code and help info.
```

Example: 

- [Boost 1.67.0](https://www.boost.org/users/history/version_1_67_0.html)   
- [CMake 3.2](https://cmake.org/download)

```
Moreover, you can include the list of requirements and how to create
 a virtual environment if needed.
```
Example:

To install the requirements, it is recommended to use a virtual environment. Otherwise, you can use your local
environment.

```sh
$ python -m venv venv
$ ... # activate the venv
$ python -m pip install -r requirements.txt
``` 

## Usage

Provide descriptions of how to make this project work.  
Including command line instructions to compile, run the code and access to usage information.

Some simple lines could be of great helpful:

**Example 1:**

```
$ mkdir build && cd build && cmake .. && make    
$ ./my_exe --help
```

**Example 2:**


You can check the arguments of the program by executing `python main.py -h`. The first argument refers to the path of
the markov chain, the second and the third corresponds to the initial and final stat respectively. The last argument is
the number of transitions. It must be an integer.

```sh
$ python main.py <markov file> <initial state index> <final state index> <days>
```

An example can be:

```shell
$ python main.py chains/chain1.txt  1 0 3
> The probability of starting at Run and ending at Sleep in 3 days is 0.137
```

## Tests

The tests are done using `unittest`. To run all the tests you can do:

```shell
python -m unittest discover
```


**Consider to break this section down for widespread descriptions by including new sections `#` or subsections `##`.**


# Contributors

If you are interested in contributing, please look at the [Contributing](CONTRIBUTING.md) guide.

The current mantainers of the project are:

- [Name of Contributor](https://github.com/username/)

# Publications

Include a list of related publications if needed.   
Otherwise remove this section.


# Changelog

If the project requires versioning, it may be convenient to summarize changes after every release.


-----------------------

Copyright 2018 Eurecat