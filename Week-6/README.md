# Week 6

## Lecture overview

The lecture covers tool calling and the agent loop: selecting an action,
calling a tool, observing its result and deciding what to do next. It also
examines context management, MCP, reliability, prompt injection and common
failure modes.

## Lab 6 assignment

Build an agent around a language model:

1. Give the model access to at least one useful tool.
2. Implement and log the full tool-use loop.
3. Define test cases that require the tool and cases where it should not be
   used.
4. Measure task success and analyse incorrect calls, invalid arguments,
   ungrounded answers and unsuccessful recovery.
5. Add basic safeguards for untrusted tool output or retrieved content.

The Week 5 retriever may be used as the tool, but this is optional. Keep code,
traces, evaluation results and failure analysis for the agents section of the
final report. Detailed materials will be added before this week.
