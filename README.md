# Large Language Models and Generative AI for NLP

Course materials for the 2026 University of Helsinki course.

## Repository structure

- `Week-1`: language-model foundations, pretraining and Lab 1
- `Week-2`: post-training and Lab 2
- `Week-3`: inference and Lab 3
- `Week-4`–`Week-7`: directories reserved for later course materials

Each completed week contains:

- `Slides`: LaTeX source and shared slide files
- `PDF`: compiled lecture and lab slides
- `Lab-N`: a runnable reference implementation where applicable

## Assessment

The course is graded through the final project. Students develop the project
throughout the course, write a report section connected to each week's
material, and submit the final report together with their code repository.
There are no separately graded weekly submissions.

Reusing one task and model across several weeks often makes the final analysis
more coherent, but it is not required.

## Using the lab examples

The examples provide tested experiment pipelines. Students should adapt the
data, prompts, parsing and evaluation to their own task, inspect raw model
outputs, and explain their experimental choices and findings.

Run each lab's local smoke test before requesting GPU resources:

```bash
bash run_smoke_test.sh
```

The bundled datasets and mock results are small pipeline checks, not results
for the final report. See the README inside each lab directory for setup and
Roihu instructions.

The CSC course project is `project_2021893`.

## Building the slides

The slide source for each week is self-contained within its `Slides`
directory. Use XeLaTeX, for example:

```bash
cd Week-1/Slides
xelatex Lecture-1-language-models.tex
```

The compiled versions are also available in the corresponding `PDF`
directory.

## Contact

Aarne Talman
`aarne.talman@helsinki.fi`
