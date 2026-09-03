# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""Bounded tool-calling planner that coordinates web and code agents.

This module defines :class:`GaiaPlanner`, the **default** GAIA planner engine
(``--engine loop``): a hand-rolled OpenAI tool-calling loop in the same spirit as
:func:`pace_agent.toolloop.run_tool_loop`.

:mod:`gaia_planner.langgraph_planner` provides an **alternative engine**
(``--engine langgraph``): :class:`~gaia_planner.langgraph_planner.LangGraphGaiaPlanner`
subclasses :class:`GaiaPlanner` and re-expresses the control flow as a LangGraph
``StateGraph``, but **imports all shared behavior from here** — system/task prompts,
tool schemas, :meth:`_call_tool`, context-budget trimming, FINAL-ANSWER nudges, and
:class:`~gaia_planner.models.PlannerRunResult` shaping. Both engines must stay
bit-compatible for PACE Agent record/replay, so this file remains the single source of
truth for planner semantics; LangGraph is orchestration only.
"""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import Any

from pace_agent.toolloop import assistant_message as _assistant_message

from .adapters import (
    ANALYZE_AUDIO_TOOL,
    ANALYZE_IMAGE_TOOL,
    ANALYZE_VIDEO_TOOL,
    BROWSE_WEB_TOOL,
    FETCH_URL_TOOL,
    PROVISION_TOOL,
    AudioAnalysisAdapter,
    ExecutionAdapter,
    ImageAnalysisAdapter,
    VideoAnalysisAdapter,
    WebVoyagerAdapter,
)
from .models import GaiaTask, PlannerConfig, PlannerRunResult

DEFAULT_PLANNER_PROMPT = """You are an expert GAIA problem-solving planner.
You may delegate focused online research to browse_web, watch video content with
analyze_video, listen to audio with analyze_audio, interpret visual content with
analyze_image, and do ALL other work - reading files, parsing pages, computation,
running programs - with execute_code (a self-provisioning Python/shell sandbox).
Decide which evidence is actually needed, then get it by DOING, not by recalling.

Core operating principles (these, not per-case recipes, are how you solve tasks):
- INSPECT before you assume. Never guess a file's name, type, structure, or contents,
  and never assume what a script/page/dataset contains. List the files, sniff the real
  type, print the first rows/bytes/keys, THEN act on what you actually observed.
- SOLVE BY EXECUTING, not by reasoning about what code or data "would" contain. If a
  program is given, RUN it and read its stdout - do not predict its output or
  reimplement it. If a page/file holds the answer, OPEN and PARSE it - do not paraphrase
  from memory. Prefer a real tool/library/engine over your own guess every time.
- Use the bundled helper for reading things: `import gaia_toolkit` exposes
  `gaia_toolkit.read(path)`, which turns ANY local file into text by sniffing its real
  type (pdf, docx, pptx, xlsx/csv, html, image via OCR, zip listing, plain text) - so you
  rarely need format-specific code. It also has `extract_code(html, language)` (pulls a
  given language's code block from a page, entities already unescaped, skipping unrelated
  blocks), `html_to_text(html)`, `sniff_type(path)`, and `list_zip`/`extract_zip`. If it
  reports a missing package, provision it and retry.
- SELF-PROVISION whatever is missing instead of giving up. The sandbox has no internet, but
  the provision tool downloads from trusted hosts (PyPI, GitHub, Hugging Face, arXiv, ...):
  `pip` installs Python packages that then import normally, and `downloads` fetches files -
  data, documents, model weights, or a CLI/engine BINARY (mark it executable, run it as
  ./<name>) - into the workspace. To read an online document/dataset, download it host-side
  (fetch_url or provision) then parse it with gaia_toolkit/execute_code; do NOT rely on
  browse_web to read a file's contents - use browse_web only to discover the direct URL.
  This applies to ordinary web PAGES too: once you know the page URL (or browse_web has
  returned it), to pull a SPECIFIC value from it (a name, number, date, table cell) fetch_url
  that exact URL and parse it with gaia_toolkit/execute_code - do NOT issue another browse_web
  for a page you already have. browse_web stalls on reading buried details; it is for
  discovery/navigation, not extraction.
- DERIVE what is not stated outright. A needed quantity is often computable from data that
  IS present (e.g. a percentage/ratio/total from raw figures on a label, table, or page) -
  extract the raw inputs and calculate it in execute_code rather than concluding it is
  unavailable. Only after genuinely exhausting derivation should you commit an estimate.
- VERIFY BY EXECUTION before answering. Re-run or cross-check the result with code, and
  sanity-check its shape against the question: a step that should yield a URL must start
  with http (else your parse/OCR is wrong - fix it, don't web-search the garbled string); a
  numeric answer must be numeric; the unit and format must match what was asked.
- Inspect tool errors and recover with a different approach; never invent tool output.

Capability routing (use the right tool for content the sandbox cannot read as bytes):
- VIDEO -> analyze_video (browse_web cannot see moving images). Resolve the exact URL with a
  browse_web search first; if analyze_video says the video is unavailable, the URL is wrong.
- AUDIO -> analyze_audio (the sandbox has NO speech-to-text; never try to transcribe audio in
  execute_code). Pass the attachment basename and a precise question.
- IMAGES: for PRINTED TEXT or CODE, use OCR (deterministic) - gaia_toolkit.read(image) or
  pytesseract - and then RUN the literal text you extracted, fixing only obvious OCR noise;
  never fabricate what the code does. Use analyze_image only to INTERPRET what OCR cannot read
  - the spatial layout of a board/diagram, values in a chart, or what a photo/scene depicts.
  Never hallucinate contents (e.g. do not invent a chess FEN); transcribe with analyze_image,
  then compute the result with a real library/engine you provision (e.g. a chess engine),
  reporting exactly what it outputs.
- Compiling/running other languages: a C/C++ toolchain and interpreters are available via
  subprocess in execute_code. If a build fails on a language-version issue, retry with a
  different standard/flags before concluding the source is broken.

Web research strategy:
- PREFER A STRUCTURED API OVER SCRAPING. When a site exposes a REST/JSON API, fetch_url the
  exact endpoint (it runs host-side and saves the response) and parse the JSON in execute_code
  - this is far more reliable than browse_web on dynamic, JavaScript-heavy pages. This is the
  right way to answer questions about repository issues/PRs/labels/commits, wiki revisions,
  papers, etc. For example, GitHub's api.github.com gives issues/labels, and its issue
  *timeline* endpoint (/repos/OWNER/REPO/issues/N/timeline) records WHEN a label was added (a
  "labeled" event with the label name and timestamp) - which page scraping cannot reliably
  read. Likewise use the MediaWiki api.php for Wikipedia and the export.arxiv.org API for arXiv.
- The EXACT value to filter on (a label, tag, category, username, enum) is often NOT the word
  used in the question - e.g. a repo's label may be "06 - Regression" or "type: bug", not
  "Regression". Do not pass the question's literal wording as a filter and trust it: first LIST
  the real values from the source (e.g. GET /repos/OWNER/REPO/labels, or the API's search/
  facet endpoint) and match the intended one. An EMPTY or zero-result response ([], count 0)
  means your filter/parameter is wrong, NOT that the thing does not exist - fix the query (try
  the discovered exact name, a search endpoint, or a looser filter) instead of concluding
  "none found".
- Prefer going straight to the authoritative source, and construct the precise search URL (or
  the site's API/search endpoint) yourself instead of dumping a broad goal at a home page and
  expecting the browser to navigate a complex search (e.g. for a date/category-filtered arXiv
  search, build the arxiv.org/search or export.arxiv.org/api query directly).
- NEVER invent a hostname or URL. If unsure of the exact page, run a DuckDuckGo search first
  to discover the real URL, then open it.
- Do NOT start at google.com (CAPTCHAs block automation); use DuckDuckGo. If a result reports a
  CAPTCHA or "automated traffic" block, retry a different site rather than repeating it.
- Web evidence may be passed into later code calls by embedding it in the generated script.
- Attached files already exist in the execution workspace under their listed basenames.

execute_code sandbox constraints (critical):
- Send Python source with language "python" (the default). Only use language "bash" for
  an actual shell script. Never label Python code as bash, and never mix the two in one
  call. Preinstalled and importable: numpy, pandas, openpyxl, opencv (cv2), pytesseract +
  Pillow (OCR), beautifulsoup4/lxml (HTML), python-docx, python-pptx, pypdf/pdfplumber,
  python-chess, plus the gaia_toolkit helper. Provision anything else you need.
- The sandbox has NO network: requests/urllib/socket calls to the internet inside
  execute_code WILL fail with a connection error. Never fetch URLs or hit an API from inside
  execute_code. Bring web data in host-side with fetch_url (or provision) FIRST, then open and
  parse the saved file in execute_code.
- Each execute_code call runs in a FRESH process. Python variables, functions, and
  imports do NOT persist between calls. Every call MUST be a self-contained script:
  re-import modules, re-read the attached file, and re-define anything you need. A
  variable or DataFrame created in a previous call is GONE - never reference it. (pandas
  as pd and numpy as np are always pre-imported for convenience, but import anything else
  you use.) If you need results across steps, print them and use the printed values, or
  do the whole computation in ONE script.
- The sandbox returns stdout. The last bare expression is auto-displayed (like a notebook
  cell), but print() every OTHER value you want to see - intermediate expressions are not
  shown. If a call returns empty stdout you have NO result: add prints and re-run; never
  guess the value from earlier context.
- To find the OUTPUT of an attached program/script, actually RUN it as a script and read
  its stdout - e.g. subprocess.run([sys.executable, "<file>"], capture_output=True,
  text=True, timeout=100) then print(result.stdout). Do NOT re-implement it, do NOT guess a
  result variable name, and do NOT exec() it with a custom globals/locals dict (that leaves
  __name__ != "__main__", so an `if __name__ == "__main__":` block never runs). Report
  exactly what it prints; the "final output" is typically the last printed line.

Before answering:
- Re-read the question and make the answer match the exact quantity, unit, and format
  requested (e.g. "how many thousand hours" means report 17, not 17000).
- A "rounded to the nearest <unit>" instruction sets the PRECISION, not the unit to
  report in. Report the value in the unit the question asks the answer to be given in,
  rounded to that precision - do NOT convert the number into the rounding unit. E.g.
  "report in Angstroms, rounded to the nearest picometer" means 1.456 (Angstroms to 3
  decimals), not 146 (picometers).
- Put ONLY the value in the FINAL ANSWER line - the bare number, word, or short phrase.
  Do NOT append units or labels (write "0.1777", not "0.1777 m^3") and do NOT wrap it in
  an explanatory sentence (write "No", not "No, Earl cannot..."), unless the question
  explicitly asks for those. Any reasoning belongs ABOVE the FINAL ANSWER line, not on it.
- Base the final answer only on values actually printed by tools; if the evidence is
  missing, run another tool call instead of inventing a number.
- Sanity-check the answer against the question. If it asks for a number and your result is
  "None", empty, or non-numeric, your approach is WRONG - fix it and re-run, do not submit
  that value. (A program that prints nothing under exec() usually needs to be run as a real
  script instead.)
- When enough evidence exists, finish with a last line exactly:
FINAL ANSWER: <answer>

Answering discipline:
- This is usually a multi-hop question. Solve each hop with its OWN focused tool
  call; do not pack several independent lookups into one browse_web goal.
- A tool saying something "does not exist" or "was not found" is NOT an answer. Treat
  it as a failed attempt and try a different query, site, or approach before concluding.
- Never end your turn with prose like "I could not find it" or a request for more
  detail. Either make another tool call, or commit to the single best-supported
  answer. The reply MUST end with a `FINAL ANSWER:` line containing only the concise
  answer (a word, number, or short phrase) in the exact format requested.
- The FINAL ANSWER must be a CONCRETE value - never a refusal or placeholder such as
  "unknown", "unable to determine", "insufficient data", "not available", "no match",
  "N/A", or similar. Such answers are always scored wrong, so they are strictly worse
  than a committed guess. If you have exhausted your research, infer the most probable
  answer from the partial evidence and the question's own constraints, and give that
  concrete value in the exact requested format (a specific name, number, or phrase).
"""

CLOSED_BOOK_PROMPT = """You are an expert problem solver working entirely from your own
knowledge and reasoning. You have NO tools: no web browser, no code execution, no
external lookups of any kind. Do not ask for tools and do not claim you will "look
something up" or "run" anything — none of that is available. Solve the task by thinking
it through yourself.

If a file is attached, its full extracted contents are provided inline in the question
below (between '--- begin <name> ---' / '--- end <name> ---' markers). Read those contents
directly from the prompt and base your answer on them — do not say you cannot open the
file. You still have no tools to fetch or open anything else.

How to solve:
- Reason step by step. Work through the problem in your head: recall the relevant facts,
  set up any calculation explicitly, and carry it out carefully (show intermediate values
  so you can check your arithmetic).
- Break multi-part questions into sub-questions, answer each from what you know, then
  combine them into the final result.
- Be decisive. If you are unsure of an exact value, reason to the most probable answer
  from what you do know and commit to it.

Before answering:
- Re-read the question and make the answer match the exact quantity, unit, and format
  requested (e.g. "how many thousand hours" means report 17, not 17000).
- A "rounded to the nearest <unit>" instruction sets the PRECISION, not the unit to
  report in. Report the value in the unit the question asks the answer to be given in,
  rounded to that precision - do NOT convert the number into the rounding unit. E.g.
  "report in Angstroms, rounded to the nearest picometer" means 1.456 (Angstroms to 3
  decimals), not 146 (picometers).
- Put ONLY the value in the FINAL ANSWER line - the bare number, word, or short phrase.
  Do NOT append units or labels (write "0.1777", not "0.1777 m^3") and do NOT wrap it in
  an explanatory sentence (write "No", not "No, Earl cannot..."), unless the question
  explicitly asks for those. Any reasoning belongs ABOVE the FINAL ANSWER line, not on it.
- Sanity-check the result against the question: a numeric question needs a number; the
  unit and format must match what was asked.
- Finish with a last line exactly:
FINAL ANSWER: <answer>

Answering discipline:
- The reply MUST end with a `FINAL ANSWER:` line containing only the concise answer (a
  word, number, or short phrase) in the exact format requested.
- The FINAL ANSWER must be a CONCRETE value — never a refusal or placeholder such as
  "unknown", "unable to determine", "insufficient data", "not available", "N/A", or
  similar. Such answers are always scored wrong, so they are strictly worse than a
  committed best guess. If you are not fully certain, infer the most probable answer from
  what you know and the question's own constraints, and give that concrete value.
"""

FINAL_ANSWER_NUDGE = (
    "Your reply did not end with a 'FINAL ANSWER:' line. Do not stop with an "
    "explanation or a request for more detail. If evidence is still missing, make "
    "another browse_web or execute_code call now. Otherwise, respond with the single "
    "best-supported answer, ending with a line exactly:\nFINAL ANSWER: <answer>"
)

EMPTY_ANSWER_NUDGE = (
    "Your 'FINAL ANSWER:' line was blank. A blank answer is never acceptable. If you "
    "still lack evidence, make another browse_web or execute_code call now. Otherwise "
    "commit to your single best-supported guess and end with a line exactly:\n"
    "FINAL ANSWER: <answer>"
)

FORCE_ANSWER_INSTRUCTION = (
    "You are out of research turns and cannot call any more tools. Do NOT ask for "
    "more information, do NOT explain what you would do next, and do NOT reply with "
    "a preamble such as 'Let me...'. Using ONLY the evidence already gathered above, "
    "commit to your single best-supported answer right now. Even if you are not "
    "fully certain, give the most likely concrete value in the exact format the "
    "question requested. Reply with exactly one line and nothing else:\n"
    "FINAL ANSWER: <answer>"
)

PLACEHOLDER_ANSWER_NUDGE = (
    "Your 'FINAL ANSWER:' was a refusal/placeholder (e.g. 'unknown', 'unable to "
    "determine', 'insufficient data'), which is always scored wrong. Do NOT give up. "
    "If the exact quantity was not stated outright, DERIVE it from the data you already "
    "have (for example, compute a percentage/ratio/total from the raw figures a document "
    "or label shows) with an execute_code calculation, or gather the missing piece with "
    "one more focused tool call. Then commit to a concrete value in the exact requested "
    "format, ending with a line exactly:\nFINAL ANSWER: <answer>"
)

# Closed-book (no_tools) variants of the nudges: identical intent, but they must NOT tell
# the model to call tools (there are none). They push it to reason to a concrete answer.
CB_FINAL_ANSWER_NUDGE = (
    "Your reply did not end with a 'FINAL ANSWER:' line. Do not stop with an "
    "explanation or a request for more detail, and do not say you will look anything "
    "up — you have no tools. Reason it through and commit to your single best-supported "
    "answer, ending with a line exactly:\nFINAL ANSWER: <answer>"
)

CB_EMPTY_ANSWER_NUDGE = (
    "Your 'FINAL ANSWER:' line was blank. A blank answer is never acceptable. Reason "
    "from what you know and commit to your single best-supported guess, ending with a "
    "line exactly:\nFINAL ANSWER: <answer>"
)

CB_PLACEHOLDER_ANSWER_NUDGE = (
    "Your 'FINAL ANSWER:' was a refusal/placeholder (e.g. 'unknown', 'unable to "
    "determine', 'insufficient data'), which is always scored wrong. Do NOT give up and "
    "do NOT ask for tools. If the exact value was not stated, DERIVE it by reasoning from "
    "what you know (for example, compute it from figures you can recall) and commit to a "
    "concrete value in the exact requested format, ending with a line exactly:\n"
    "FINAL ANSWER: <answer>"
)

CB_FORCE_ANSWER_INSTRUCTION = (
    "This is your last turn. Do NOT ask for information, do NOT explain what you would do "
    "next, and do NOT reply with a preamble such as 'Let me...'. Using ONLY your own "
    "reasoning and knowledge, commit to your single best-supported answer right now. Even "
    "if you are not fully certain, give the most likely concrete value in the exact format "
    "the question requested. Reply with exactly one line and nothing else:\n"
    "FINAL ANSWER: <answer>"
)

# Refusal/placeholder answers are scored wrong, so they are never accepted as final;
# the model is nudged to derive or research a concrete value instead.
_PLACEHOLDER_ANSWERS: tuple[str, ...] = (
    "unknown",
    "unable to determine",
    "cannot determine",
    "could not determine",
    "insufficient data",
    "insufficient information",
    "not available",
    "not enough information",
    "no match",
    "no answer",
    "n/a",
    "na",
    "none",
    "undetermined",
    "indeterminate",
    "unclear",
)


def _is_placeholder_answer(value: str) -> bool:
    normalized = re.sub(r"[\s\.\!\?]+$", "", value.strip().lower())
    normalized = normalized.strip("`\"'*")
    return normalized in _PLACEHOLDER_ANSWERS


def _select_nudge(answer_value: str | None, no_tools: bool) -> str:
    """Pick the corrective nudge for a turn that produced no acceptable answer.

    ``answer_value`` is a non-empty placeholder answer, ``""`` for a blank FINAL ANSWER
    line, or ``None`` when the line was missing entirely. In closed-book mode the nudge
    must not mention tools, so the ``CB_*`` variants are used."""
    if answer_value:
        return CB_PLACEHOLDER_ANSWER_NUDGE if no_tools else PLACEHOLDER_ANSWER_NUDGE
    if answer_value == "":
        return CB_EMPTY_ANSWER_NUDGE if no_tools else EMPTY_ANSWER_NUDGE
    return CB_FINAL_ANSWER_NUDGE if no_tools else FINAL_ANSWER_NUDGE


ExecutionFactory = Callable[[GaiaTask], AbstractContextManager[Any]]


class GaiaPlanner:
    """Hand-rolled GAIA planner loop (default ``--engine loop``).

    Subclassed by :class:`~gaia_planner.langgraph_planner.LangGraphGaiaPlanner` for
    the LangGraph engine; override :meth:`run` there, not the tool/prompt helpers here.
    """

    def __init__(
        self,
        *,
        client: Any,
        web_adapter: WebVoyagerAdapter,
        execution_factory: ExecutionFactory | None,
        config: PlannerConfig,
        video_adapter: VideoAnalysisAdapter | None = None,
        audio_adapter: AudioAnalysisAdapter | None = None,
        image_adapter: ImageAnalysisAdapter | None = None,
    ) -> None:
        self.client = client
        self.web_adapter = web_adapter
        self.execution_factory = execution_factory or (
            lambda task: ExecutionAdapter(task)
        )
        self.config = config
        self.video_adapter = video_adapter
        self.audio_adapter = audio_adapter
        self.image_adapter = image_adapter

    def run(self, task: GaiaTask) -> PlannerRunResult:
        system_prompt = self.config.system_prompt or self._default_prompt()
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": self._task_prompt(task)},
        ]
        tool_events: list[dict[str, Any]] = []
        web_state: dict[str, int] = {"fail_streak": 0}

        with self._execution_cm(task) as execution:
            # Closed-book mode: no tools are built and none are advertised to the model.
            tools = None if execution is None else self._build_tools(execution)

            for _turn in range(self.config.max_turns):
                self._enforce_context_budget(messages)
                response = self._agent_call(messages, tools, tool_choice="auto")
                message = response.choices[0].message
                if not message.tool_calls:
                    raw_answer = message.content or ""
                    messages.append({"role": "assistant", "content": raw_answer})
                    answer_value = _final_answer_value(raw_answer)
                    if answer_value and not _is_placeholder_answer(answer_value):
                        return _result(task, raw_answer, messages, tool_events)
                    # No FINAL ANSWER line (None), one that was blank (""), or a
                    # refusal/placeholder ("unknown", etc.). We don't accept any of
                    # these: push the model to keep researching, derive the value, or
                    # commit to a concrete, non-empty answer.
                    nudge = _select_nudge(answer_value, self.config.no_tools)
                    messages.append({"role": "user", "content": nudge})
                    continue

                messages.append(_assistant_message(message))
                for tool_call in message.tool_calls:
                    output = self._call_tool(tool_call, task, execution, web_state)
                    stored = _truncate_text(output, self.config.max_tool_output_chars)
                    event = {
                        "tool_call_id": tool_call.id,
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                        "output": stored,
                    }
                    tool_events.append(event)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": stored,
                        }
                    )

            raw_answer = self._force_final_answer(messages, tools)
            return _result(task, raw_answer, messages, tool_events)

    def _task_prompt(self, task: GaiaTask) -> str:
        """Build the user task prompt for this run's mode. In closed-book mode the attached
        files' text is inlined (the model cannot open them); otherwise only names are listed.
        """
        return _task_prompt(
            task,
            no_tools=self.config.no_tools,
            file_char_budget=self.config.attached_file_char_budget,
        )

    def _default_prompt(self) -> str:
        """The built-in system prompt for the active mode (overridden by config.system_prompt)."""
        return CLOSED_BOOK_PROMPT if self.config.no_tools else DEFAULT_PLANNER_PROMPT

    def _execution_cm(self, task: GaiaTask) -> AbstractContextManager[Any]:
        """Execution sandbox context, or a no-op yielding ``None`` in closed-book mode
        (where no tools run, so the sandbox is never opened)."""
        if self.config.no_tools:
            from contextlib import nullcontext

            return nullcontext(None)
        return self.execution_factory(task)

    def _agent_call(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        *,
        tool_choice: str,
    ) -> Any:
        """One chat-completions call. When ``tools`` is falsy (closed-book mode) neither
        ``tools`` nor ``tool_choice`` is sent, so the model is never told any tool exists.
        """
        kwargs: dict[str, Any] = {"model": self.config.model, "messages": messages}
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice
        return self.client.chat.completions.create(**kwargs)

    def _force_final_answer(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> str:
        """Post-loop forced answer, with retries, when the research budget is spent.

        A single ``tool_choice="none"`` call sometimes returns empty content or a
        "Let me..." preamble instead of a committed value — this is the dominant
        failure mode on hard tasks that exhaust ``max_turns`` with the answer
        nearly in hand. Retry with an escalating, answer-only instruction so a
        concrete ``FINAL ANSWER:`` is extracted from the evidence already gathered.
        Mutates ``messages`` in place and returns the last raw answer seen.
        """
        raw_answer = ""
        no_tools = self.config.no_tools
        attempts = max(1, self.config.final_answer_retries + 1)
        for attempt in range(attempts):
            if attempt == 0:
                nudge = CB_FINAL_ANSWER_NUDGE if no_tools else FINAL_ANSWER_NUDGE
            else:
                nudge = (
                    CB_FORCE_ANSWER_INSTRUCTION
                    if no_tools
                    else FORCE_ANSWER_INSTRUCTION
                )
            messages.append({"role": "user", "content": nudge})
            self._enforce_context_budget(messages)
            # tool_choice="none" is only valid alongside a tools list; closed-book sends
            # neither, so the forced-answer call is a plain completion.
            response = self._agent_call(messages, tools, tool_choice="none")
            raw_answer = response.choices[0].message.content or ""
            messages.append({"role": "assistant", "content": raw_answer})
            answer_value = _final_answer_value(raw_answer)
            if answer_value and not _is_placeholder_answer(answer_value):
                break
        return raw_answer

    def _build_tools(self, execution: Any) -> list[dict[str, Any]]:
        """Assemble the OpenAI tool schemas offered to the model for a task. Shared by the
        loop engine (GaiaPlanner) and the LangGraph engine (LangGraphGaiaPlanner)."""
        execute_schema = execution.schema
        if execute_schema is None:
            raise RuntimeError("Execution adapter did not expose a tool schema")
        tools = [BROWSE_WEB_TOOL, execute_schema, PROVISION_TOOL, FETCH_URL_TOOL]
        if self.video_adapter is not None:
            tools.append(ANALYZE_VIDEO_TOOL)
        if self.audio_adapter is not None:
            tools.append(ANALYZE_AUDIO_TOOL)
        if self.image_adapter is not None:
            tools.append(ANALYZE_IMAGE_TOOL)
        return tools

    def _call_tool(
        self,
        tool_call: Any,
        task: GaiaTask,
        execution: Any,
        web_state: dict[str, int] | None = None,
    ) -> str:
        try:
            arguments = json.loads(tool_call.function.arguments)
            if not isinstance(arguments, dict):
                raise ValueError("tool arguments must be a JSON object")
            if tool_call.function.name == "browse_web":
                goal = arguments.get("goal")
                if not isinstance(goal, str) or not goal.strip():
                    raise ValueError("browse_web requires a non-empty goal")
                goal = goal.strip()
                start_url = arguments.get("start_url") or task.start_url
                output = self.web_adapter.call(goal, str(start_url))
                fail_streak = 0
                if web_state is not None:
                    if _web_stalled(output):
                        web_state["fail_streak"] = web_state.get("fail_streak", 0) + 1
                    else:
                        web_state["fail_streak"] = 0
                    fail_streak = web_state["fail_streak"]
                return output + _web_followup_hint(output, fail_streak)
            if tool_call.function.name == "execute_code":
                return execution.call(arguments)
            if tool_call.function.name == "provision":
                return execution.provision(arguments)
            if tool_call.function.name == "fetch_url":
                return execution.fetch(arguments)
            if tool_call.function.name == "analyze_video":
                if self.video_adapter is None:
                    raise ValueError("analyze_video tool is not configured")
                return self.video_adapter.call(arguments)
            if tool_call.function.name == "analyze_audio":
                if self.audio_adapter is None:
                    raise ValueError("analyze_audio tool is not configured")
                return self.audio_adapter.call(arguments, task.files)
            if tool_call.function.name == "analyze_image":
                if self.image_adapter is None:
                    raise ValueError("analyze_image tool is not configured")
                return self.image_adapter.call(arguments, task.files)
            raise ValueError(f"unknown tool: {tool_call.function.name}")
        except Exception as exc:
            return json.dumps(
                {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
            )

    def _enforce_context_budget(self, messages: list[dict[str, Any]]) -> None:
        """Keep the running history under a character budget.

        A single oversized tool output can otherwise push a request past the
        model's context limit (observed: one task ballooned to ~1.36M tokens).
        We shrink the oldest tool-result messages first, always preserving the
        most recent ones and the assistant/tool_call pairing, so requests stay
        valid while the context stays bounded.
        """
        budget = self.config.context_char_budget
        if budget <= 0 or _messages_size(messages) <= budget:
            return
        tool_indexes = [
            i for i, message in enumerate(messages) if message.get("role") == "tool"
        ]
        keep = max(0, self.config.keep_recent_tool_outputs)
        trimmable = tool_indexes[:-keep] if keep else tool_indexes
        placeholder = "[trimmed: earlier tool output dropped to fit the context window]"
        for index in trimmable:
            if _messages_size(messages) <= budget:
                break
            content = messages[index].get("content")
            if isinstance(content, str) and len(content) > len(placeholder):
                messages[index]["content"] = placeholder


def _truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    head = max_chars // 2
    tail = max_chars - head
    omitted = len(text) - max_chars
    return f"{text[:head]}\n...[{omitted} chars truncated]...\n{text[-tail:]}"


def _messages_size(messages: list[dict[str, Any]]) -> int:
    return sum(
        len(json.dumps(message, default=str, ensure_ascii=False))
        for message in messages
    )


def _web_stalled(output: str) -> bool:
    """True if a browse_web result carries no usable answer.

    WebVoyager returns ``status="failed"`` (often with ``iterations`` at the cap)
    and a null/empty answer when it cannot complete a goal - the failure mode of
    packed multi-hop goals and unreadable pages.
    """
    try:
        payload = json.loads(output)
    except (json.JSONDecodeError, TypeError):
        return False
    if not isinstance(payload, dict):
        return False
    return payload.get("status") != "success" or payload.get("answer") in (None, "")


def _web_followup_hint(output: str, fail_streak: int) -> str:
    """Append corrective guidance when a browse_web call returns no answer.

    A single stalled call gets decomposition advice (break the task into one
    atomic hop). When several web calls fail in a row, escalate: the model is
    usually stuck re-searching the same way, or an earlier sub-answer it is
    relying on is wrong - both need a change of approach, not another retry.
    This just-in-time feedback is far more effective than upfront prose.
    """
    if not _web_stalled(output):
        return ""
    hint = (
        "\n\n[web strategy] this call returned no answer (WebVoyager likely hit its "
        "step limit or could not read the page). The goal is probably too broad or "
        "multi-step. Break the task into ONE atomic hop and issue a fresh browse_web "
        "call for just that step (e.g. first find only the article/page URL, then in a "
        "separate call open that exact URL to read the specific detail). Carry each "
        "hop's result (a concrete URL or fact) into the next call's goal or start_url."
    )
    if fail_streak >= 3:
        hint += (
            f" NOTE: {fail_streak} web calls in a row have failed. Do NOT reword and "
            "retry the same search again. Change approach: (a) re-examine whether an "
            "earlier sub-answer you are relying on (e.g. a name or journal) might be "
            "WRONG and re-derive it; (b) try a different, authoritative source or "
            "start_url; and (c) if the target is a document/PDF, open its direct file "
            "URL rather than a listing page."
        )
    return hint


def _final_answer_value(response: str) -> str | None:
    """Return the text after the last 'FINAL ANSWER:' line.

    Returns None when no such line exists, and an empty string when the line is
    present but blank. Callers treat both as "not a valid final answer".
    """
    for line in reversed(response.splitlines()):
        stripped = line.strip()
        if stripped.upper().startswith("FINAL ANSWER:"):
            return stripped.split(":", 1)[1].strip()
    return None


def extract_final_answer(response: str) -> str:
    for line in reversed(response.splitlines()):
        stripped = line.strip()
        if stripped.upper().startswith("FINAL ANSWER:"):
            return stripped.split(":", 1)[1].strip()
    return response.strip()


def _read_attached_text(task: GaiaTask, max_chars: int) -> str:
    """Extract the text of each attached file (host-side, via ``sandbox_toolkit.read``) so
    it can be inlined into the prompt for closed-book runs. Reading a file is not a tool the
    model chooses — it is just context — so a no-tools run still gets the document content.
    Each file is truncated to ``max_chars`` and read failures are reported inline rather than
    raised, so one unreadable attachment never aborts the task.

    A read failure is ALSO surfaced loudly on stderr: in closed-book mode an unreadable
    attachment means the model silently gets no data (and can only guess), so the operator
    must see it rather than have it swallowed into an inline stub."""
    from . import sandbox_toolkit

    sections: list[str] = []
    for path in task.files:
        try:
            text = sandbox_toolkit.read(path)
        except Exception as exc:  # noqa: BLE001 - surface the failure as context
            text = f"[could not read {path.name}: {type(exc).__name__}: {exc}]"
            print(
                f"WARNING: closed-book attachment inlining FAILED for {path.name}: "
                f"{type(exc).__name__}: {exc}\n"
                "  The model will receive NO contents for this file and can only guess. "
                "Install the matching reader (e.g. `pip install openpyxl` for xlsx, "
                "`python-docx` for docx, `pypdf`/`pdfminer.six` for pdf) and re-run.",
                file=sys.stderr,
                flush=True,
            )
        sections.append(
            f"--- begin {path.name} ---\n"
            f"{_truncate_text(text, max_chars)}\n"
            f"--- end {path.name} ---"
        )
    return "\n\n".join(sections)


def _task_prompt(
    task: GaiaTask, *, no_tools: bool = False, file_char_budget: int = 0
) -> str:
    prompt = f"Task ID: {task.task_id}\nQuestion: {task.question}"
    if task.files:
        prompt += "\nAttached files: " + ", ".join(path.name for path in task.files)
        # Closed-book mode: the model cannot open the file, so inline its extracted text.
        if no_tools and file_char_budget > 0:
            body = _read_attached_text(task, file_char_budget)
            if body:
                prompt += (
                    "\n\nThe full extracted contents of the attached file(s) are "
                    "provided below; base your answer on them.\n\n" + body
                )
    # The web-research hint only makes sense when tools are available; omit it in
    # closed-book mode, where the system prompt states there is no web access.
    if not no_tools:
        prompt += f"\nSuggested starting URL for web research: {task.start_url}"
    return prompt


def _result(
    task: GaiaTask,
    raw_answer: str,
    messages: list[dict[str, Any]],
    tool_events: list[dict[str, Any]],
    per_node: dict[str, Any] | None = None,
) -> PlannerRunResult:
    return PlannerRunResult(
        task_id=task.task_id,
        answer=extract_final_answer(raw_answer),
        raw_answer=raw_answer,
        messages=messages,
        tool_events=tool_events,
        per_node=per_node,
    )
