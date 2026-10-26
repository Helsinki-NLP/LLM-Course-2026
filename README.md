# Large Language Models and Generative AI for NLP

Course materials for the 2026 University of Helsinki course.

## Course at a glance

- **Course:** Natural Language Generation / LLMs and Generative AI
  (`LDA-T318`)
- **Credits:** 5 ECTS
- **Teaching language:** English
- **Period:** 27 October–10 December 2026
- **Lectures:** Tuesdays 16:15–17:45, Metsätalo B525 (sali 25)
- **Labs:** Thursdays 16:15–17:45, Metsätalo B314 (sali 7)
- **Audience:** Master's students in language technology and data science
- **Teachers:** Aarne Talman (weeks 1–3 and 7), Jussi Karlgren (week 4),
  Dmitry Kan (weeks 5–6)
- **Computing:** CSC Roihu, using course project `project_2021893`
- **Assessment:** final project, consisting of a report and code repository

**Prerequisites:** Python programming, basic machine learning and familiarity
with a Unix command line. PyTorch experience is useful but not required.

## Learning outcomes

After the course, students should be able to:

1. explain how decoder-only language models process text and generate tokens;
2. train a tokenizer and a small language model, and interpret training and
   scaling results;
3. compare prompting with supervised and preference-based post-training;
4. serve a model and measure latency, throughput, memory use and quality;
5. evaluate language models critically, including multilingual evaluation;
6. build retrieval-augmented and tool-using systems around a model; and
7. run machine-learning workloads on an HPC system using modules and Slurm.

## Course outline

- **Language-model foundations:** next-token prediction, tokenization,
  perplexity, attention and transformer architecture
- **Pretraining:** data pipelines, optimization, scaling laws and distributed
  training
- **Post-training:** chat templates, supervised fine-tuning, preferences,
  verifiable rewards and LoRA
- **Inference:** decoding, KV-cache memory, batching, quantization and serving
- **Evaluation:** metrics, benchmarks, contamination, model-based evaluation
  and multilingual variation
- **Retrieval-augmented generation:** retrieval, reranking, context assembly
  and RAG evaluation
- **Agents:** tool use, agent loops, context management and failure modes

## Schedule

### Week 1 — 27 and 29 October — Aarne Talman

- Tuesday: Language models
- Thursday: Pretraining
- Lab 1: From text to a tiny GPT

### Week 2 — 3 and 5 November — Aarne Talman

- Tuesday: Post-training
- Thursday: Lab 2 — does fine-tuning beat prompting?

### Week 3 — 10 and 12 November — Aarne Talman

- Tuesday: Inference
- Thursday: Lab 3 — serve and measure your model

### Week 4 — 17 and 19 November — Jussi Karlgren

- Tuesday: Evaluation
- Thursday: Lab 4 — evaluate your models

### Week 5 — 24 and 26 November — Dmitry Kan

- Tuesday: Retrieval-augmented generation
- Thursday: Lab 5 — RAG for the project task

### Week 6 — 1 and 3 December — Dmitry Kan

- Tuesday: Agents
- Thursday: Lab 6 — build a tool-using agent

### Week 7 — 8 and 10 December — all teachers

- Tuesday: Course recap
- Thursday: Special topics and project support

There are no intermediate submissions. The final report and code repository
are due on 31 December 2026.

## Repository structure

- `Week-1`: language-model foundations, pretraining and Lab 1
- `Week-2`: post-training and Lab 2
- `Week-3`: inference and Lab 3
- `Week-4`–`Week-7`: directories reserved for later course materials

Each completed week contains:

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

## Contact

Aarne Talman
`aarne.talman@helsinki.fi`
