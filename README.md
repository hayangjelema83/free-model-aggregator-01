# Free Model Aggregator

One server that answers the OpenAI Chat Completions API and the Anthropic Messages API
(JSON and server-sent events) by proxying each request to the best free model that meets
it, speaking that provider's own wire (openai / anthropic / gemini) with the provider's own
auth header. Built for the make.fun bounty `free-model-aggregator-01`.

## Run

```sh
node main.ts            # trial mode: reads $MAKEFUN_PROMPT_FILE, registers with $MAKEFUN_ENV_URL
node main.ts --list     # print the catalog document and exit 0
```

Node 22+, standard library only. No dependencies, no build step.

## How it works

- **Brief.** At startup the server reads the brief at `$MAKEFUN_PROMPT_FILE` (provider
  directory, quality table, workload manifest) and probes every provider's catalog route in
  that provider's own wire, building one catalog document. The same document is printed by
  `--list` and sent in the `POST /register` body.
- **Routing.** For each request the candidates are every free model meeting the request's
  requirements, ordered by the quality table's `rating_milli`, highest first. Ties prefer
  the provider whose wire matches the client's wire (no translation to get wrong), then the
  provider with the most per-trial budget headroom left. The first candidate that answers
  serves the request; a refusal (402 / 429 / 5xx, or unreachable) moves to the next one.
  A provider that says its budget is spent, or that it is out, is remembered as exhausted
  and skipped for the rest of the trial; a refusal on its failure schedule is not exhaustion.
  The order is a pure function of the brief, the requests and the provider answers seen so
  far: no wall clock, no unseeded randomness, so a replay reproduces the trial exactly.
- **Wires.** Requests pass through untouched in the provider's wire: system prompt, user
  text (images included), tools, stop sequences, `max_tokens`, JSON mode. A JSON-mode
  request from an OpenAI client never goes to an anthropic-wire provider, because that wire
  cannot carry `response_format`. Answers come back in the wire the client used, streamed
  or not as asked, with the provider's text whole, usage as that wire reports it, and the
  stop reason mapped (`end_turn` -> `stop`, `max_tokens` -> `length`,
  `tool_use` -> `tool_calls`, `refusal` -> `content_filter`, and the reverse).
- **Trial.** The server listens on an ephemeral port, registers it with the env, serves the
  client's requests one at a time, polls `GET /work`, and exits 0 when the env says done.

## Files

- `main.ts` — the whole server: brief parsing, catalog probe, routing, wire translation,
  HTTP server, register and poll.
- `README.md` — this file.
