# Guidelines

This repository contains a template *README* to document a repository for a new open source project.

## Pusblishing Steps

1. Write a note to describe briefly your code, including:
    - Title of the project.
    - Brief description of the code's purpose.
    - Code access link (private for the moment).
    - Any additional consideration that you consider relevant for the publishing review.
2. The administrator must grant access to the repository. Afterwards the pushes to that repository are freely granted
   for the developer.
3. Check [How to use this template](#how-to-use-this-template) to populate the repository with your code.
4. Further versions of the code affecting any publishing aspect considered throughout the above steps must be
   documented.

## How to Use this template

1. Check it out from GitHub (no need to fork it).
2. Use the template to create a new project.
3. Populate the copied *README* file with the correspoding info. 
4. Change the name of the [non-CONTRIBUTING.md](non-CONTRIBUTING.md) file and delete [CONTRIBUTING.md](CONTRIBUTING.md)
   in case the repository cannot be updated by thirds.
5. Once updated the license, delete [the license guide](licenses.pdf).
6. Delete this file.
7. Update your requirements and check Github Actions work.
8. Keep on coding! and documenting! Check [Project Structure](#project-structure).

### Project structure

The current project structure must be:

```
- .gitub/            <- Folder for repository management
- src/               <- Folder of code
- *tests/
- *data/
- README.md
- LICENSE
- CONTRIBUTING.md
```

Folders and files marked with `*` are optional. Consider the data folder can be a data sample.

## General Guidelines

- Use **ENGLISH** as unique language for coding and documenting.
- Document often the code, the README file and additional info files.
- [Markdown](https://guides.github.com/features/mastering-markdown) flavor makes receipts tasty.

# Project Screening

Remove names, email address, IP addresses and in general sensible information including internal paths or filenames,
unless explicit permission for that.

You may find useful screening commands like:

```shell
egrep -r '\.eurecat\.com|@eurecat\.com|eurecat?/|([0-9]+\.){3}[0-9]+' <path-to-source-directory>
```

## Source Code Headers

EVERY file containing source code must include copyright and license information.

Apache header:

    Copyright 2018 Eurecat

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        https://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

MIT and Apache 2.0 licenses can be favoured, but check [the license guide](licenses.pdf).. See the documentation for instructions on using alternate license. You may
find useful this [autogen](https://github.com/mbrukman/autogen) tool to generate license headers including some sample
outputs.


-----------------------

Copyright 2018 Eurecat