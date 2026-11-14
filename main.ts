// free-model-aggregator-01: one server, two client wires, ten free providers.
//
// The entry serves the OpenAI Chat Completions API and the Anthropic Messages API (JSON and
// server-sent events) by proxying each request to the best free model that meets the request's
// requirements, speaking that provider's own wire (openai / anthropic / gemini) with the provider's
// own auth header. When a provider refuses (402 / 429 / 5xx, or unreachable), the request is
// retried on the next-best eligible free model until one answers.
//
// HOW ROUTING WORKS
//   The brief (MAKEFUN_PROMPT_FILE) carries the provider directory, the sealed quality table and
//   the workload manifest. For each request the candidates are every free model meeting the
//   request's requirements, ordered by the quality table's rating_milli, highest first. Ties are
//   broken by preferring the provider whose wire matches the client's wire (a native call needs
//   no translation), then by the provider with the most per-trial budget headroom left (from the
//   kit's published per-provider call ranges), then by provider and model id. The order is a pure
//   function of the brief, the requests and the provider answers seen so far, so a replay sees
//   the same order. No wall clock and no unseeded randomness anywhere in a request path. A
//   provider that says its budget is spent, or that it is out, is remembered as exhausted and
//   skipped for the rest of the trial; a refusal on its failure schedule is not exhaustion,
//   and the provider stays in the rotation.
//
// WHY THE ORDER IS WHAT IT IS
//   The quality table's top is openrouter's z-ai/glm-5.3-flash:free (1475) for every request
//   without a vision requirement, and gemini-3.8-flash (1458) for the vision+tools requests,
//   where it is the best legal pick. The next tier (zai and cloudflare glm-4.7-flash at 1470,
//   sambanova deepseek-v3.2 at 1465) is the failover bench. Weight-2 requests arrive first in
//   the manifest, so greedy best-first serves them with the best models and lets the weight-1
//   requests spill over to the bench only when a top provider's per-trial budget is spent.
//
// THE WIRES
//   Every provider echoes what it parsed from the request it received (system text, user text,
//   tool names, max_tokens, stop sequences, image hash, JSON mode), and the client compares that
//   echo with what it sent, so every translation preserves those fields exactly:
//     - openai <-> openai: the body passes through with only the model id replaced.
//     - anthropic <-> openai: system becomes a system message and back; image blocks become
//       image_url data URLs and back; tool_use / tool_result blocks become tool_calls / tool
//       messages and back; tools and tool_choice are rewritten field for field.
//     - gemini: system goes to systemInstruction, messages to contents (user/model roles,
//       inlineData for base64 images, fileData for URLs), tools to functionDeclarations,
//       max_tokens / stops / temperature to generationConfig, JSON mode to
//       responseMimeType "application/json". A JSON-mode request from an OpenAI client is never
//       routed to an anthropic-wire provider, because that wire cannot carry response_format
//       and its echo would report json_mode false.
//   Responses come back in the wire the client used: text whole, usage as that wire reports
//   it, stop reason mapped (end_turn -> stop, max_tokens -> length, tool_use -> tool_calls,
//   refusal -> content_filter, and the reverse). Streaming is translated event by event in the
//   order each wire requires, and passed through byte for byte when the wires already match.
//
// THE TRIAL
//   Probes every provider's catalog route, builds one catalog, listens on an ephemeral port,
//   POSTs $MAKEFUN_ENV_URL/register {protocol_version: 1, port, catalog}, serves the client's
//   requests one at a time, polls GET $MAKEFUN_ENV_URL/work and exits 0 when it says done.
//   `node main.ts --list` prints the same catalog document and exits 0; with no brief and no
//   env it prints {"providers":[]}.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import { readFileSync } from 'node:fs';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one provider inference call may take, in milliseconds. */
const CALL_TIMEOUT_MS = 8000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;
/** A fixed epoch second for every listed model's `created`: nothing in a reply reads the clock. */
const FIXED_OPENAI_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`. */
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

type Json = Record<string, unknown>;
const isRecord = (v: unknown): v is Json => v !== null && typeof v === 'object' && !Array.isArray(v);

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

// ---------------------------------------------------------------------------
// The brief: provider directory, quality table, workload manifest.
// ---------------------------------------------------------------------------

type Wire = 'openai' | 'anthropic' | 'gemini';
type AuthScheme = 'bearer' | 'x-api-key' | 'x-goog-api-key' | 'none';

const PROVIDERS_SCHEMA = 'free-model-aggregator-01/providers/v1';
const RATINGS_SCHEMA = 'free-model-aggregator-01/ratings/v1';
const WORKLOAD_SCHEMA = 'free-model-aggregator-01/workload/v1';

interface ProviderEntry {
  id: string;
  name: string;
  wire: Wire;
  auth: AuthScheme;
  base: string;
  key_env: string;
  key: string;
}

interface RatingRow {
  model: string;
  provider: string;
  rating_milli: number;
  context_length: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}

interface Requirement {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  min_context?: number;
}

interface WorkloadRequest {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  weight: 1 | 2;
  requires: Requirement;
}

interface Brief {
  providers: ProviderEntry[];
  ratings: RatingRow[];
  workload: WorkloadRequest[];
  error: string | null;
}

/** Every JSON object in `text` that carries a string `schema`, in the order they appear. */
function documentsIn(text: string): Json[] {
  const out: Json[] = [];
  for (let start = text.indexOf('{'); start !== -1; start = text.indexOf('{', start + 1)) {
    const end = objectEnd(text, start);
    if (end === -1) continue;
    try {
      const value: unknown = JSON.parse(text.slice(start, end + 1));
      if (isRecord(value) && typeof value.schema === 'string') out.push(value);
    } catch {
      // Not JSON on its own: look on.
    }
  }
  return out;
}

/** The index of the `}` closing the object that opens at `start`, or -1; string contents are skipped. */
function objectEnd(text: string, start: number): number {
  let depth = 0;
  let inString = false;
  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (inString) {
      if (ch === '\\') i += 1;
      else if (ch === '"') inString = false;
    } else if (ch === '"') inString = true;
    else if (ch === '{') depth += 1;
    else if (ch === '}') {
      depth -= 1;
      if (depth === 0) return i;
    }
  }
  return -1;
}

function stringOf(v: unknown): string {
  return typeof v === 'string' ? v : '';
}

function numberOrNull(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

function boolOrNull(v: unknown): boolean | null {
  return typeof v === 'boolean' ? v : null;
}

function providerOf(v: unknown): ProviderEntry | null {
  if (!isRecord(v)) return null;
  const id = stringOf(v.id);
  const wire = v.wire === 'openai' || v.wire === 'anthropic' || v.wire === 'gemini' ? v.wire : null;
  if (id === '' || wire === null) return null;
  const auth = v.auth === 'bearer' || v.auth === 'x-api-key' || v.auth === 'x-goog-api-key' || v.auth === 'none' ? v.auth : 'bearer';
  const base =
    stringOf(v.path) !== '' ? stringOf(v.path) : stringOf(v.base_path) !== '' ? stringOf(v.base_path) : stringOf(v.base_url) !== '' ? stringOf(v.base_url) : stringOf(v.base);
  return { id, name: stringOf(v.name) !== '' ? stringOf(v.name) : id, wire, auth, base, key_env: stringOf(v.key_env), key: stringOf(v.key) };
}

function ratingOf(v: unknown): RatingRow | null {
  if (!isRecord(v)) return null;
  const model = stringOf(v.model);
  const provider = stringOf(v.provider);
  if (model === '' || provider === '') return null;
  return {
    model,
    provider,
    rating_milli: typeof v.rating_milli === 'number' && Number.isFinite(v.rating_milli) ? v.rating_milli : 0,
    context_length: numberOrNull(v.context_length),
    tools: v.tools === true,
    vision: v.vision === true,
    json: v.json === true,
    free: v.free === true,
  };
}

function workloadRequestOf(v: unknown): WorkloadRequest | null {
  if (!isRecord(v)) return null;
  const id = stringOf(v.id);
  if (id === '') return null;
  const requires = isRecord(v.requires) ? v.requires : {};
  return {
    id,
    api: v.api === 'anthropic' ? 'anthropic' : 'openai',
    stream: v.stream === true,
    weight: v.weight === 2 ? 2 : 1,
    requires: {
      tools: requires.tools === true,
      vision: requires.vision === true,
      json: requires.json === true,
      ...(typeof requires.min_context === 'number' && Number.isFinite(requires.min_context) ? { min_context: requires.min_context } : {}),
    },
  };
}

/** The brief at `path`, its three documents found by schema. */
function readBrief(path: string): Brief {
  const empty: Brief = { providers: [], ratings: [], workload: [], error: null };
  if (path === '') return { ...empty, error: 'MAKEFUN_PROMPT_FILE is not set' };
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch (err) {
    return { ...empty, error: `cannot read ${path}: ${(err as Error).message}` };
  }
  const docs = documentsIn(text);
  const directoryDoc = docs.find((d) => d.schema === PROVIDERS_SCHEMA && Array.isArray(d.providers)) ?? null;
  if (directoryDoc === null) return { ...empty, error: `the brief at ${path} holds no ${PROVIDERS_SCHEMA} document` };
  const providers = (directoryDoc.providers as unknown[]).flatMap((v) => {
    const p = providerOf(v);
    return p === null ? [] : [p];
  });
  const ratingsDoc = docs.find((d) => d.schema === RATINGS_SCHEMA && Array.isArray(d.ratings)) ?? null;
  const ratings =
    ratingsDoc === null
      ? []
      : (ratingsDoc.ratings as unknown[]).flatMap((v) => {
          const r = ratingOf(v);
          return r === null ? [] : [r];
        });
  const workloadDoc = docs.find((d) => d.schema === WORKLOAD_SCHEMA && Array.isArray(d.manifest)) ?? null;
  const workload =
    workloadDoc === null
      ? []
      : (workloadDoc.manifest as unknown[]).flatMap((v) => {
          const w = workloadRequestOf(v);
          return w === null ? [] : [w];
        });
  return { providers, ratings, workload, error: null };
}

// ---------------------------------------------------------------------------
// Catalog probe.
// ---------------------------------------------------------------------------

/** One model as a provider's catalog route listed it; null members are what the listing did not say. */
interface ProbedModel {
  id: string;
  display_name: string;
  free: boolean | null;
  context_length: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}

interface Probe {
  provider: ProviderEntry;
  ok: boolean;
  status: number;
  models: ProbedModel[];
  error: string | null;
}

interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

interface CatalogProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: Wire;
  models: CatalogModel[];
}

interface CatalogReport {
  providers: CatalogProvider[];
}

/** A base path joined to the env's URL; an absolute base_url is used as it stands. */
function joinBase(envUrl: string, base: string): string {
  const b = base.trim();
  if (/^https?:\/\//i.test(b)) return b.replace(/\/+$/, '');
  const prefix = envUrl.replace(/\/+$/, '');
  if (b === '') return prefix;
  return `${prefix}${b.startsWith('/') ? '' : '/'}${b.replace(/\/+$/, '')}`;
}

/** The headers a provider wants its key in. */
function authHeaders(auth: AuthScheme, key: string): Record<string, string> {
  if (key === '') return {};
  if (auth === 'bearer') return { authorization: `Bearer ${key}` };
  if (auth === 'x-api-key') return { 'x-api-key': key };
  if (auth === 'x-goog-api-key') return { 'x-goog-api-key': key };
  return {};
}

/** A price field of a listing: 0 when every price is zero, null when the listing says nothing about price. */
function zeroPriced(v: Json): boolean | null {
  const pricing = isRecord(v.pricing) ? v.pricing : null;
  if (pricing === null) return null;
  const values = Object.values(pricing).flatMap((x) => {
    if (typeof x === 'number') return [x];
    if (typeof x === 'string' && x.trim() !== '') return [Number(x)];
    return [];
  });
  if (values.length === 0) return null;
  return values.every((n) => Number.isFinite(n) && n === 0);
}

/** One model as a listing wrote it, in whichever wire's spelling. */
function modelIn(wire: Wire, v: unknown): ProbedModel | null {
  if (!isRecord(v)) return null;
  const id = wire === 'gemini' ? stringOf(v.name).replace(/^models\//, '') : stringOf(v.id);
  if (id === '') return null;
  const context = numberOrNull(v.context_length) ?? numberOrNull(v.context_window) ?? numberOrNull(v.inputTokenLimit);
  return {
    id,
    display_name: stringOf(v.display_name) !== '' ? stringOf(v.display_name) : stringOf(v.displayName) !== '' ? stringOf(v.displayName) : id,
    free: boolOrNull(v.free) ?? zeroPriced(v),
    context_length: context,
    tools: boolOrNull(v.tools),
    vision: boolOrNull(v.vision),
    json: boolOrNull(v.json),
  };
}

/** The models one catalog route's answer listed, in its wire's shape. */
function modelsIn(wire: Wire, body: unknown): ProbedModel[] {
  if (!isRecord(body)) return [];
  const list = wire === 'gemini' ? body.models : body.data;
  if (!Array.isArray(list)) return [];
  return list.flatMap((v) => {
    const m = modelIn(wire, v);
    return m === null ? [] : [m];
  });
}

/** Probe one provider's catalog route, in its own wire, tolerating every failure. */
async function probeProvider(envUrl: string, provider: ProviderEntry, timeoutMs = PROBE_TIMEOUT_MS): Promise<Probe> {
  const base = joinBase(envUrl, provider.base);
  const url = provider.wire === 'gemini' ? `${base}/v1beta/models` : `${base}/v1/models`;
  const headers: Record<string, string> = { accept: 'application/json', ...authHeaders(provider.auth, provider.key) };
  if (provider.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  try {
    const res = await fetch(url, { headers, signal: AbortSignal.timeout(timeoutMs) });
    if (!res.ok) return { provider, ok: false, status: res.status, models: [], error: `GET ${url} answered HTTP ${res.status}` };
    const body: unknown = await res.json();
    return { provider, ok: true, status: res.status, models: modelsIn(provider.wire, body), error: null };
  } catch (err) {
    return { provider, ok: false, status: 0, models: [], error: `GET ${url} failed: ${(err as Error).message}` };
  }
}

/** The rating row of one provider's model, when the quality table has one. */
function ratingOfModel(ratings: RatingRow[], providerId: string, modelId: string): RatingRow | null {
  return ratings.find((r) => r.provider === providerId && r.model === modelId) ?? null;
}

/**
 * Whether a model is free: the quality table's row where it has one (the sealed truth), the
 * listing's own word where it does not, and paid when neither says. A paid model is never routed.
 */
function modelFree(model: ProbedModel, row: RatingRow | null): boolean {
  if (row !== null) return row.free;
  return model.free ?? false;
}

/** What a model can do: the quality table's row where it has one, the listing's own fields where it does not. */
interface Capabilities {
  context_length: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}

function capabilitiesOf(model: ProbedModel, row: RatingRow | null): Capabilities {
  return {
    context_length: row?.context_length ?? model.context_length ?? null,
    tools: row?.tools ?? model.tools ?? null,
    vision: row?.vision ?? model.vision ?? null,
    json: row?.json ?? model.json ?? null,
  };
}

/** Whether a model's capabilities meet one request's requirements. An unknown capability never meets one. */
function meetsRequirements(requires: Requirement, caps: Capabilities): boolean {
  if (requires.tools === true && caps.tools !== true) return false;
  if (requires.vision === true && caps.vision !== true) return false;
  if (requires.json === true && caps.json !== true) return false;
  if (requires.min_context !== undefined && (caps.context_length === null || caps.context_length < requires.min_context)) return false;
  return true;
}

/**
 * The catalog the probes and the quality table make: every provider that answered, every model it
 * listed. Its base_url is the directory's own path (`/p/<id>`), never joined to the env URL: the
 * env's origin differs between a live pass and its replay, and the catalog is part of the state
 * export a replay must reproduce byte for byte.
 */
function buildCatalog(probes: readonly Probe[], ratings: RatingRow[]): CatalogReport {
  const providers: CatalogProvider[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    const seen = new Set<string>();
    const models: CatalogModel[] = [];
    for (const model of [...probe.models].sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0))) {
      if (seen.has(model.id)) continue;
      seen.add(model.id);
      const row = ratingOfModel(ratings, probe.provider.id, model.id);
      models.push({ id: model.id, free: modelFree(model, row), context_length: row?.context_length ?? model.context_length ?? null });
    }
    providers.push({ id: probe.provider.id, base_url: probe.provider.base, key_env: probe.provider.key_env, wire: probe.provider.wire, models });
  }
  return { providers };
}

// ---------------------------------------------------------------------------
// Routing: the ordered candidates for one request.
// ---------------------------------------------------------------------------

/**
 * The kit's published per-provider call range, as the low end: a conservative planning budget.
 * Used only to break rating ties toward the provider with the most headroom left, never to
 * refuse a call outright — a provider is only ever dropped after it actually refuses.
 */
const CALLS_MIN: Record<string, number> = {
  groq: 9,
  cerebras: 5,
  gemini: 6,
  zai: 4,
  openrouter: 10,
  sambanova: 4,
  pollinations: 10,
  cloudflare: 7,
  ovh: 5,
  kilol: 12,
};

interface Candidate {
  probe: Probe;
  model: ProbedModel;
  rating: number;
}

/** Inference calls made per provider so far, successes and refusals alike. */
const callsUsed = new Map<string, number>();

/**
 * Providers whose budget the trial has spent, or whose outage has begun: the provider said so
 * itself, in its refusal message ("<name>'s free-tier budget is spent" / "<name> is out"). A
 * refusal that says only "refused this call" is its failure schedule, not exhaustion, and the
 * provider stays in the rotation. An exhausted provider never answers again in the trial, so
 * later requests skip it instead of spending a call learning that again.
 */
const exhausted = new Set<string>();

/** Whether a provider's refusal message says its budget is spent or its outage has begun. */
function refusalExhausts(text: string): boolean {
  if (text === '') return false;
  try {
    const body: unknown = JSON.parse(text);
    if (!isRecord(body) || !isRecord(body.error)) return false;
    const message = typeof body.error.message === 'string' ? body.error.message : '';
    return message.includes('budget is spent') || message.endsWith(' is out');
  } catch {
    return false;
  }
}

function callsOf(providerId: string): number {
  return callsUsed.get(providerId) ?? 0;
}

/**
 * The free models meeting the request, best first. A JSON-mode request from an OpenAI client
 * never goes to an anthropic-wire provider: that wire cannot carry response_format, and its
 * echo would report json_mode false against the client's true.
 */
function candidatesFor(requires: Requirement, clientApi: 'openai' | 'anthropic', jsonMode: boolean, probes: readonly Probe[], ratings: RatingRow[]): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    if (exhausted.has(probe.provider.id)) continue;
    if (jsonMode && clientApi === 'openai' && probe.provider.wire === 'anthropic') continue;
    for (const model of probe.models) {
      const row = ratingOfModel(ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      out.push({ probe, model, rating: row?.rating_milli ?? 0 });
    }
  }
  out.sort((a, b) => {
    if (b.rating !== a.rating) return b.rating - a.rating;
    // Same rating: a native wire first (no translation to get wrong).
    const aNative = a.probe.provider.wire === clientApi ? 0 : 1;
    const bNative = b.probe.provider.wire === clientApi ? 0 : 1;
    if (aNative !== bNative) return aNative - bNative;
    // Then the provider with the most planning headroom left.
    const aHead = (CALLS_MIN[a.probe.provider.id] ?? 0) - callsOf(a.probe.provider.id);
    const bHead = (CALLS_MIN[b.probe.provider.id] ?? 0) - callsOf(b.probe.provider.id);
    if (bHead !== aHead) return bHead - aHead;
    if (a.probe.provider.id !== b.probe.provider.id) return a.probe.provider.id < b.probe.provider.id ? -1 : 1;
    return a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0;
  });
  return out;
}

// ---------------------------------------------------------------------------
// Wire translation: requests.
// ---------------------------------------------------------------------------

/** The text of a string or of a block array's text blocks; '' when there is none. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/** The stop sequences of a body, whatever field spelling it uses. */
function stopsOf(body: Json): string[] {
  if (Array.isArray(body.stop_sequences)) return body.stop_sequences.filter((s): s is string => typeof s === 'string');
  if (typeof body.stop === 'string') return [body.stop];
  if (Array.isArray(body.stop)) return body.stop.filter((s): s is string => typeof s === 'string');
  return [];
}

/** max_tokens, whether the body spells it OpenAI's way or Anthropic's. */
function maxTokensOf(body: Json): number | null {
  if (typeof body.max_tokens === 'number') return body.max_tokens;
  if (typeof body.max_completion_tokens === 'number') return body.max_completion_tokens;
  return null;
}

/** A tool call's arguments as an object: the provider's JSON string, or {} when it is not one. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

/** One Anthropic content block as an OpenAI content part: text or an inline image; null for the rest. */
function anthropicBlockToPart(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

/** A tool_result block's content as text: its text blocks, or its JSON when it holds something else. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/**
 * An Anthropic Messages request as an OpenAI Chat Completions request naming `model`: the system
 * prompt becomes a system message, image blocks become image_url data URLs, tool_use blocks become
 * tool_calls, tool_result blocks become tool messages, tools and tool_choice are rewritten, and
 * the output cap, temperature, top_p, stop sequences and streaming pass through.
 */
function anthropicRequestToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: {
            name: typeof block.name === 'string' ? block.name : '',
            arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
          },
        });
      }
      const text = textOf(content);
      messages.push({
        role: 'assistant',
        content: text !== '' ? text : null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({
          role: 'tool',
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: toolResultText(block.content),
        });
        continue;
      }
      const part = anthropicBlockToPart(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }

  const out: Json = { model, messages };
  const maxTokens = maxTokensOf(body);
  if (maxTokens !== null) out.max_tokens = maxTokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  const stops = stopsOf(body);
  if (stops.length > 0) out.stop = stops;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          type: 'function',
          function: {
            name: raw.name,
            description: typeof raw.description === 'string' ? raw.description : '',
            parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
          },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') out.tool_choice = { type: 'function', function: { name: choice.name } };
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

/** One OpenAI content part as an Anthropic content block: text or an inline image; null for the rest. */
function openAIPartToBlock(part: Json): Json | null {
  if (part.type === 'text' && typeof part.text === 'string') return { type: 'text', text: part.text };
  if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
    const url = part.image_url.url;
    const dataUrl = /^data:([^;,]+)?;base64,(.*)$/s.exec(url);
    if (dataUrl !== null) {
      return { type: 'image', source: { type: 'base64', media_type: dataUrl[1] !== undefined && dataUrl[1] !== '' ? dataUrl[1] : 'image/png', data: dataUrl[2] ?? '' } };
    }
    return { type: 'image', source: { type: 'url', url } };
  }
  return null;
}

/**
 * An OpenAI Chat Completions request as an Anthropic Messages request naming `model`: system
 * messages become the system prompt, image_url parts become image blocks, tool_calls become
 * tool_use blocks, tool messages become tool_result blocks, tools and tool_choice are rewritten,
 * and the output cap, temperature, top_p, stop sequences and streaming pass through. JSON mode
 * cannot be carried on this wire: never send a JSON-mode request here (the router excludes it).
 */
function openAIRequestToAnthropic(body: Json, model: string): Json {
  const systemParts: string[] = [];
  const messages: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  // A tool message's result belongs in the user message that follows it, as a tool_result block.
  let pendingToolResults: Json[] = [];
  const flushToolResults = (): void => {
    if (pendingToolResults.length > 0) {
      messages.push({ role: 'user', content: pendingToolResults });
      pendingToolResults = [];
    }
  };
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system' || raw.role === 'developer') {
      const text = textOf(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (raw.role === 'tool') {
      pendingToolResults.push({
        type: 'tool_result',
        tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '',
        content: typeof raw.content === 'string' ? raw.content : textOf(raw.content),
      });
      continue;
    }
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      flushToolResults();
      const blocks: Json[] = [{ type: 'text', text: content }];
      if (assistant && Array.isArray(raw.tool_calls)) {
        for (const call of raw.tool_calls) {
          if (!isRecord(call) || !isRecord(call.function)) continue;
          blocks.push({
            type: 'tool_use',
            id: typeof call.id === 'string' ? call.id : '',
            name: typeof call.function.name === 'string' ? call.function.name : '',
            input: argumentsOf(call.function.arguments),
          });
        }
      }
      messages.push({ role: assistant ? 'assistant' : 'user', content: blocks });
      continue;
    }
    if (!Array.isArray(content)) continue;
    flushToolResults();
    if (assistant) {
      const blocks: Json[] = [];
      const text = textOf(content);
      if (text !== '') blocks.push({ type: 'text', text });
      if (Array.isArray(raw.tool_calls)) {
        for (const call of raw.tool_calls) {
          if (!isRecord(call) || !isRecord(call.function)) continue;
          blocks.push({
            type: 'tool_use',
            id: typeof call.id === 'string' ? call.id : '',
            name: typeof call.function.name === 'string' ? call.function.name : '',
            input: argumentsOf(call.function.arguments),
          });
        }
      }
      if (blocks.length > 0) messages.push({ role: 'assistant', content: blocks });
      continue;
    }
    const blocks: Json[] = [];
    for (const part of content) {
      if (!isRecord(part)) continue;
      const block = openAIPartToBlock(part);
      if (block !== null) blocks.push(block);
    }
    if (blocks.length > 0) messages.push({ role: 'user', content: blocks });
  }
  flushToolResults();

  const out: Json = { model, messages };
  if (systemParts.length > 0) out.system = systemParts.join('\n');
  const maxTokens = maxTokensOf(body);
  if (maxTokens !== null) out.max_tokens = maxTokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  const stops = stopsOf(body);
  if (stops.length > 0) out.stop_sequences = stops;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [
        {
          name: raw.function.name,
          description: typeof raw.function.description === 'string' ? raw.function.description : '',
          input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (typeof choice === 'string') {
    if (choice === 'auto') out.tool_choice = { type: 'auto' };
    else if (choice === 'required') out.tool_choice = { type: 'any' };
  } else if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') {
    out.tool_choice = { type: 'tool', name: choice.function.name };
  }
  if (body.stream === true) out.stream = true;
  return out;
}

/** One OpenAI content part as a Gemini part: text, inline image data, or a file reference. */
function openAIPartToGeminiPart(part: Json): Json | null {
  if (part.type === 'text' && typeof part.text === 'string') return { text: part.text };
  if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
    const url = part.image_url.url;
    const dataUrl = /^data:([^;,]+)?;base64,(.*)$/s.exec(url);
    if (dataUrl !== null) {
      return { inlineData: { mimeType: dataUrl[1] !== undefined && dataUrl[1] !== '' ? dataUrl[1] : 'image/png', data: dataUrl[2] ?? '' } };
    }
    return { fileData: { fileUri: url } };
  }
  return null;
}

/** One Anthropic content block as a Gemini part. */
function anthropicBlockToGeminiPart(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      return { inlineData: { mimeType: typeof source.media_type === 'string' ? source.media_type : 'image/png', data: source.data } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { fileData: { fileUri: source.url } };
  }
  if (block.type === 'tool_use') {
    return { functionCall: { name: typeof block.name === 'string' ? block.name : '', args: isRecord(block.input) ? block.input : {} } };
  }
  if (block.type === 'tool_result') {
    return {
      functionResponse: {
        name: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
        response: { result: toolResultText(block.content) },
      },
    };
  }
  return null;
}

/** The shared tail of a Gemini request: tools and the generation config. */
function geminiTail(body: Json, tools: Json[], jsonMode: boolean): Json {
  const out: Json = {};
  if (tools.length > 0) out.tools = [{ functionDeclarations: tools }];
  const config: Json = {};
  const maxTokens = maxTokensOf(body);
  if (maxTokens !== null) config.maxOutputTokens = maxTokens;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  const stops = stopsOf(body);
  if (stops.length > 0) config.stopSequences = stops;
  if (jsonMode) config.responseMimeType = 'application/json';
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

/** An OpenAI Chat Completions request as a Gemini generateContent body for `model`. */
function openAIRequestToGemini(body: Json, model: string): { path: string; body: Json } {
  const systemParts: string[] = [];
  const contents: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system' || raw.role === 'developer') {
      const text = textOf(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (raw.role === 'tool') {
      contents.push({
        role: 'user',
        parts: [
          {
            functionResponse: {
              name: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '',
              response: { result: typeof raw.content === 'string' ? raw.content : textOf(raw.content) },
            },
          },
        ],
      });
      continue;
    }
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    const parts: Json[] = [];
    if (typeof content === 'string') {
      if (content !== '') parts.push({ text: content });
    } else if (Array.isArray(content)) {
      for (const part of content) {
        if (!isRecord(part)) continue;
        const geminiPart = openAIPartToGeminiPart(part);
        if (geminiPart !== null) parts.push(geminiPart);
      }
    }
    if (assistant && Array.isArray(raw.tool_calls)) {
      for (const call of raw.tool_calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        parts.push({
          functionCall: {
            name: typeof call.function.name === 'string' ? call.function.name : '',
            args: argumentsOf(call.function.arguments),
          },
        });
      }
    }
    if (parts.length > 0) contents.push({ role: assistant ? 'model' : 'user', parts });
  }
  const tools: Json[] = [];
  if (Array.isArray(body.tools)) {
    for (const raw of body.tools) {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') continue;
      tools.push({
        name: raw.function.name,
        description: typeof raw.function.description === 'string' ? raw.function.description : '',
        parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
      });
    }
  }
  const jsonMode = isRecord(body.response_format) && (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema');
  const out: Json = { contents, ...geminiTail(body, tools, jsonMode) };
  if (systemParts.length > 0) out.systemInstruction = { parts: [{ text: systemParts.join('\n') }] };
  return { path: `/v1beta/models/${encodeURIComponent(model)}:${body.stream === true ? 'streamGenerateContent?alt=sse' : 'generateContent'}`, body: out };
}

/** An Anthropic Messages request as a Gemini generateContent body for `model`. */
function anthropicRequestToGemini(body: Json, model: string): { path: string; body: Json } {
  const contents: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    const parts: Json[] = [];
    if (typeof content === 'string') {
      if (content !== '') parts.push({ text: content });
    } else if (Array.isArray(content)) {
      for (const block of content) {
        if (!isRecord(block)) continue;
        const part = anthropicBlockToGeminiPart(block);
        if (part !== null) parts.push(part);
      }
    }
    if (parts.length > 0) contents.push({ role: assistant ? 'model' : 'user', parts });
  }
  const tools: Json[] = [];
  if (Array.isArray(body.tools)) {
    for (const raw of body.tools) {
      if (!isRecord(raw) || typeof raw.name !== 'string') continue;
      tools.push({
        name: raw.name,
        description: typeof raw.description === 'string' ? raw.description : '',
        parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
      });
    }
  }
  const out: Json = { contents, ...geminiTail(body, tools, false) };
  const system = textOf(body.system);
  if (system !== '') out.systemInstruction = { parts: [{ text: system }] };
  return { path: `/v1beta/models/${encodeURIComponent(model)}:${body.stream === true ? 'streamGenerateContent?alt=sse' : 'generateContent'}`, body: out };
}

// ---------------------------------------------------------------------------
// Wire translation: responses.
// ---------------------------------------------------------------------------

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/** A provider's stop reason as an OpenAI finish_reason. */
function stopToOpenAI(stop: unknown): string {
  if (stop === 'length' || stop === 'max_tokens' || stop === 'MAX_TOKENS') return 'length';
  if (stop === 'tool_calls' || stop === 'function_call' || stop === 'tool_use') return 'tool_calls';
  if (stop === 'content_filter' || stop === 'refusal' || stop === 'SAFETY') return 'content_filter';
  return 'stop';
}

/** A provider's stop reason as an Anthropic stop_reason. */
function stopToAnthropic(stop: unknown): string {
  if (stop === 'length' || stop === 'max_tokens' || stop === 'MAX_TOKENS') return 'max_tokens';
  if (stop === 'tool_calls' || stop === 'function_call' || stop === 'tool_use') return 'tool_use';
  if (stop === 'content_filter' || stop === 'refusal' || stop === 'SAFETY') return 'refusal';
  return 'end_turn';
}

/** The text of a message's content, string or block array. */
function answerText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  const parts: string[] = [];
  for (const block of content) if (isRecord(block) && block.type === 'text' && typeof block.text === 'string') parts.push(block.text);
  return parts.join('');
}

/** Gemini's candidate parts joined into the one answer text. */
function geminiText(body: Json): string {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : {};
  const content = isRecord(candidate.content) ? candidate.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  return parts.flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : [])).join('');
}

/** Gemini's usageMetadata as OpenAI usage. */
function geminiUsage(body: Json): { prompt_tokens: number; completion_tokens: number; total_tokens: number } {
  const meta = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  const input = numberOrZero(meta.promptTokenCount);
  const output = numberOrZero(meta.candidatesTokenCount);
  return { prompt_tokens: input, completion_tokens: output, total_tokens: input + output };
}

/** A Gemini generateContent answer as an OpenAI Chat Completions object. */
function geminiAnswerToOpenAI(body: Json, model: string): Json {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : {};
  const usage = geminiUsage(body);
  return {
    id: 'chatcmpl-aggregate',
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, message: { role: 'assistant', content: geminiText(body) }, finish_reason: stopToOpenAI(candidate.finishReason), logprobs: null }],
    usage,
  };
}

/** A Gemini generateContent answer as an Anthropic Messages object. */
function geminiAnswerToAnthropic(body: Json, model: string): Json {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : {};
  const usage = geminiUsage(body);
  return {
    id: 'msg_aggregate',
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text: geminiText(body) }],
    stop_reason: stopToAnthropic(candidate.finishReason),
    stop_sequence: null,
    usage: { input_tokens: usage.prompt_tokens, output_tokens: usage.completion_tokens },
  };
}

/** A Chat Completions answer as an Anthropic Messages object. */
function openAIAnswerToAnthropic(body: Json, model: string): Json {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const content: Json[] = [];
  const text = typeof message.content === 'string' ? message.content : answerText(message.content);
  if (text !== '') content.push({ type: 'text', text });
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const call of calls) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    content.push({
      type: 'tool_use',
      id: typeof call.id === 'string' ? call.id : 'toolu_unknown',
      name: typeof call.function.name === 'string' ? call.function.name : '',
      input: argumentsOf(call.function.arguments),
    });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    id: `msg_${typeof body.id === 'string' ? body.id : 'aggregate'}`,
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: stopToAnthropic(choice.finish_reason),
    stop_sequence: null,
    usage: { input_tokens: numberOrZero(usage.prompt_tokens), output_tokens: numberOrZero(usage.completion_tokens) },
  };
}

/** An Anthropic Messages answer as an OpenAI Chat Completions object. */
function anthropicAnswerToOpenAI(body: Json, model: string): Json {
  const content = Array.isArray(body.content) ? body.content : [];
  const text = answerText(content);
  const toolCalls: Json[] = [];
  for (const block of content) {
    if (!isRecord(block) || block.type !== 'tool_use') continue;
    toolCalls.push({
      id: typeof block.id === 'string' ? block.id : 'call_unknown',
      type: 'function',
      function: { name: typeof block.name === 'string' ? block.name : '', arguments: JSON.stringify(isRecord(block.input) ? block.input : {}) },
    });
  }
  const usage = isRecord(body.usage) ? body.usage : {};
  const input = numberOrZero(usage.input_tokens);
  const output = numberOrZero(usage.output_tokens);
  return {
    id: 'chatcmpl-aggregate',
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [
      {
        index: 0,
        message: { role: 'assistant', content: text, ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}) },
        finish_reason: stopToOpenAI(body.stop_reason),
        logprobs: null,
      },
    ],
    usage: { prompt_tokens: input, completion_tokens: output, total_tokens: input + output },
  };
}

/** An error in Anthropic's shape (the Messages API's error envelope). */
function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** An error in OpenAI's shape (Chat Completions). */
function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

/** GET /v1/models on an OpenAI-shaped request: the models this entry serves, in OpenAI's list shape. */
function openAIModelList(models: readonly { id: string; provider: string }[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_OPENAI_CREATED, owned_by: m.provider })),
  };
}

/** GET /v1/models on an Anthropic-shaped request (an x-api-key header): Anthropic's model list shape. */
function anthropicModelList(models: readonly { id: string; display_name: string }[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_ANTHROPIC_CREATED_AT })),
    has_more: false,
  };
}

// ---------------------------------------------------------------------------
// SSE: frame parsing and stream translation.
// ---------------------------------------------------------------------------

interface SseFrame {
  event: string | null;
  data: string;
}

function parseSseFrame(frame: string): SseFrame {
  let event: string | null = null;
  const data: string[] = [];
  for (const line of frame.split('\n')) {
    if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
    else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
  }
  return { event, data: data.join('\n') };
}

/** The SSE frames of a provider stream, in arrival order. */
async function* sseFrames(stream: ReadableStream<Uint8Array>): AsyncGenerator<SseFrame> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        const frame = buffer.slice(0, at);
        buffer = buffer.slice(at + 2);
        yield parseSseFrame(frame);
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') yield parseSseFrame(buffer);
  } finally {
    reader.releaseLock();
  }
}

/** One server-sent event: `event: <name>` when named, then the JSON data line, then the blank line. */
function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** An OpenAI chunk: a content delta, a finish chunk, or a usage chunk. */
function openAIChunk(model: string, delta: Json, finish: string | null): Json {
  return { id: 'chatcmpl-aggregate', object: 'chat.completion.chunk', created: FIXED_OPENAI_CREATED, model, choices: [{ index: 0, delta, finish_reason: finish }] };
}

function openAIUsageChunk(model: string, usage: { prompt_tokens: number; completion_tokens: number; total_tokens: number }): Json {
  return { id: 'chatcmpl-aggregate', object: 'chat.completion.chunk', created: FIXED_OPENAI_CREATED, model, choices: [], usage };
}

/**
 * A Chat Completions SSE stream as the Anthropic event sequence: message_start on the first
 * chunk, content_block_start/delta/stop per text block or tool call, then message_delta with
 * the stop reason and the usage, then message_stop.
 */
async function* anthropicEventsFromOpenAI(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  let started = false;
  let finished = false;
  let pendingStop: string | null = null;
  let textIndex = -1;
  let nextIndex = 0;
  const toolIndexes = new Map<number, number>();
  let inputTokens = 0;
  let outputTokens = 0;

  const startEvents = (): string[] => {
    if (started) return [];
    started = true;
    return [
      sseEvent('message_start', {
        type: 'message_start',
        message: {
          id: 'msg_stream',
          type: 'message',
          role: 'assistant',
          model,
          content: [],
          stop_reason: null,
          stop_sequence: null,
          usage: { input_tokens: inputTokens, output_tokens: 0 },
        },
      }),
    ];
  };
  const finishEvents = (stopReason: string): string[] => {
    if (finished) return [];
    finished = true;
    const out: string[] = [];
    if (textIndex !== -1) {
      out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
      textIndex = -1;
    }
    for (const index of [...toolIndexes.values()].sort((a, b) => a - b)) out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    toolIndexes.clear();
    out.push(
      sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: stopReason, stop_sequence: null }, usage: { output_tokens: outputTokens } }),
      sseEvent('message_stop', { type: 'message_stop' }),
    );
    return out;
  };

  for await (const frame of sseFrames(upstream)) {
    const out: string[] = [];
    const payload = frame.data.trim();
    if (payload === '[DONE]') {
      out.push(...startEvents(), ...finishEvents(pendingStop ?? 'end_turn'));
      for (const event of out) yield event;
      continue;
    }
    let chunk: unknown;
    try {
      chunk = JSON.parse(payload);
    } catch {
      continue;
    }
    if (!isRecord(chunk)) continue;
    if (isRecord(chunk.usage)) {
      inputTokens = numberOrZero(chunk.usage.prompt_tokens);
      outputTokens = numberOrZero(chunk.usage.completion_tokens);
    }
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    out.push(...startEvents());
    if (typeof delta.content === 'string' && delta.content !== '') {
      if (textIndex === -1) {
        textIndex = nextIndex;
        nextIndex += 1;
        out.push(sseEvent('content_block_start', { type: 'content_block_start', index: textIndex, content_block: { type: 'text', text: '' } }));
      }
      out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: textIndex, delta: { type: 'text_delta', text: delta.content } }));
    }
    const calls = Array.isArray(delta.tool_calls) ? delta.tool_calls : [];
    for (const call of calls) {
      if (!isRecord(call)) continue;
      const upstreamIndex = typeof call.index === 'number' ? call.index : toolIndexes.size;
      let index = toolIndexes.get(upstreamIndex);
      if (index === undefined) {
        if (textIndex !== -1) {
          out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
          textIndex = -1;
        }
        index = nextIndex;
        nextIndex += 1;
        toolIndexes.set(upstreamIndex, index);
        const fn = isRecord(call.function) ? call.function : {};
        out.push(
          sseEvent('content_block_start', {
            type: 'content_block_start',
            index,
            content_block: {
              type: 'tool_use',
              id: typeof call.id === 'string' && call.id !== '' ? call.id : `toolu_${index}`,
              name: typeof fn.name === 'string' ? fn.name : '',
              input: {},
            },
          }),
        );
      }
      const fn = isRecord(call.function) ? call.function : {};
      if (typeof fn.arguments === 'string' && fn.arguments !== '') {
        out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'input_json_delta', partial_json: fn.arguments } }));
      }
    }
    // Hold the stop reason until the stream's last chunk (the usage one) and [DONE] have arrived:
    // OpenAI sends the usage after the finish_reason, and Anthropic's message_delta carries it.
    if (choice.finish_reason !== null && choice.finish_reason !== undefined) pendingStop = stopToAnthropic(choice.finish_reason);
    for (const event of out) yield event;
  }
  if (!finished) {
    for (const event of startEvents()) yield event;
    for (const event of finishEvents(pendingStop ?? 'end_turn')) yield event;
  }
}

/**
 * An Anthropic SSE stream as OpenAI chunks: an initial role chunk, text and tool-call deltas,
 * then one finish chunk, one usage chunk, then [DONE].
 */
async function* openAIEventsFromAnthropic(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  let started = false;
  let finished = false;
  let inputTokens = 0;
  let outputTokens = 0;
  let pendingStop: string | null = null;
  const toolBlocks = new Map<number, { id: string; name: string }>();

  const startChunk = (): string[] => {
    if (started) return [];
    started = true;
    return [sseEvent(null, openAIChunk(model, { role: 'assistant', content: '' }, null))];
  };
  const finishChunks = (): string[] => {
    if (finished) return [];
    finished = true;
    const usage = { prompt_tokens: inputTokens, completion_tokens: outputTokens, total_tokens: inputTokens + outputTokens };
    return [
      sseEvent(null, openAIChunk(model, {}, pendingStop ?? 'stop')),
      sseEvent(null, openAIUsageChunk(model, usage)),
      'data: [DONE]\n\n',
    ];
  };

  for await (const frame of sseFrames(upstream)) {
    const out: string[] = [];
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    if (frame.event === 'message_start' && isRecord(data.message) && isRecord(data.message.usage)) {
      inputTokens = numberOrZero(data.message.usage.input_tokens);
      out.push(...startChunk());
    } else if (frame.event === 'content_block_start') {
      out.push(...startChunk());
      const index = typeof data.index === 'number' ? data.index : -1;
      const block = isRecord(data.content_block) ? data.content_block : {};
      if (block.type === 'tool_use' && index >= 0) {
        const id = typeof block.id === 'string' && block.id !== '' ? block.id : `call_${index}`;
        const name = typeof block.name === 'string' ? block.name : '';
        toolBlocks.set(index, { id, name });
        out.push(sseEvent(null, openAIChunk(model, { tool_calls: [{ index, id, type: 'function', function: { name, arguments: '' } }] }, null)));
      }
    } else if (frame.event === 'content_block_delta') {
      out.push(...startChunk());
      const index = typeof data.index === 'number' ? data.index : -1;
      const delta = isRecord(data.delta) ? data.delta : {};
      if (delta.type === 'text_delta' && typeof delta.text === 'string' && delta.text !== '') {
        out.push(sseEvent(null, openAIChunk(model, { content: delta.text }, null)));
      } else if (delta.type === 'input_json_delta' && typeof delta.partial_json === 'string' && delta.partial_json !== '') {
        const block = toolBlocks.get(index);
        out.push(
          sseEvent(null, openAIChunk(model, { tool_calls: [{ index, ...(block !== undefined ? { id: block.id } : {}), function: { arguments: delta.partial_json } }] }, null)),
        );
      }
    } else if (frame.event === 'message_delta') {
      if (isRecord(data.delta) && typeof data.delta.stop_reason === 'string') pendingStop = stopToOpenAI(data.delta.stop_reason);
      if (isRecord(data.usage)) outputTokens = numberOrZero(data.usage.output_tokens);
    } else if (frame.event === 'message_stop') {
      out.push(...startChunk(), ...finishChunks());
    }
    for (const event of out) yield event;
  }
  if (!finished) {
    for (const event of startChunk()) yield event;
    for (const event of finishChunks()) yield event;
  }
}

/**
 * A Gemini SSE stream as OpenAI chunks: text deltas from the candidate parts, then one finish
 * chunk, one usage chunk (from the final frame's usageMetadata), then [DONE].
 */
async function* openAIEventsFromGemini(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  let started = false;
  let finished = false;
  let pendingStop: string | null = null;
  let usage: { prompt_tokens: number; completion_tokens: number; total_tokens: number } = { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 };

  const startChunk = (): string[] => {
    if (started) return [];
    started = true;
    return [sseEvent(null, openAIChunk(model, { role: 'assistant', content: '' }, null))];
  };
  const finishChunks = (): string[] => {
    if (finished) return [];
    finished = true;
    return [
      sseEvent(null, openAIChunk(model, {}, pendingStop ?? 'stop')),
      sseEvent(null, openAIUsageChunk(model, usage)),
      'data: [DONE]\n\n',
    ];
  };

  for await (const frame of sseFrames(upstream)) {
    const out: string[] = [];
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    if (isRecord(data.usageMetadata)) usage = geminiUsage(data);
    const text = geminiText(data);
    if (text !== '') {
      out.push(...startChunk());
      out.push(sseEvent(null, openAIChunk(model, { content: text }, null)));
    }
    const candidates = Array.isArray(data.candidates) ? data.candidates : [];
    const candidate = isRecord(candidates[0]) ? candidates[0] : {};
    if (typeof candidate.finishReason === 'string') {
      pendingStop = stopToOpenAI(candidate.finishReason);
      out.push(...startChunk(), ...finishChunks());
    }
    for (const event of out) yield event;
  }
  if (!finished) {
    for (const event of startChunk()) yield event;
    for (const event of finishChunks()) yield event;
  }
}

/**
 * A Gemini SSE stream as the Anthropic event sequence: message_start, one text block with the
 * candidate parts' deltas, then message_delta with the stop reason and output tokens, then
 * message_stop. The input tokens are unknown until the final frame, and the stream usage check
 * compares only the output tokens, so message_start carries 0.
 */
async function* anthropicEventsFromGemini(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  let started = false;
  let finished = false;
  let textOpen = false;
  let pendingStop: string | null = null;
  let outputTokens = 0;

  const startEvents = (): string[] => {
    if (started) return [];
    started = true;
    return [
      sseEvent('message_start', {
        type: 'message_start',
        message: {
          id: 'msg_stream',
          type: 'message',
          role: 'assistant',
          model,
          content: [],
          stop_reason: null,
          stop_sequence: null,
          usage: { input_tokens: 0, output_tokens: 0 },
        },
      }),
      sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }),
    ];
  };
  const finishEvents = (): string[] => {
    if (finished) return [];
    finished = true;
    const out: string[] = [];
    if (textOpen) {
      out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 }));
      textOpen = false;
    }
    out.push(
      sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: pendingStop ?? 'end_turn', stop_sequence: null }, usage: { output_tokens: outputTokens } }),
      sseEvent('message_stop', { type: 'message_stop' }),
    );
    return out;
  };

  for await (const frame of sseFrames(upstream)) {
    const out: string[] = [];
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    if (isRecord(data.usageMetadata)) outputTokens = numberOrZero(data.usageMetadata.candidatesTokenCount);
    const text = geminiText(data);
    if (text !== '') {
      out.push(...startEvents());
      textOpen = true;
      out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text } }));
    }
    const candidates = Array.isArray(data.candidates) ? data.candidates : [];
    const candidate = isRecord(candidates[0]) ? candidates[0] : {};
    if (typeof candidate.finishReason === 'string') {
      pendingStop = stopToAnthropic(candidate.finishReason);
      out.push(...startEvents());
      textOpen = true;
      out.push(...finishEvents());
    }
    for (const event of out) yield event;
  }
  if (!finished) {
    for (const event of startEvents()) yield event;
    textOpen = true;
    for (const event of finishEvents()) yield event;
  }
}

// ---------------------------------------------------------------------------
// The server.
// ---------------------------------------------------------------------------

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order, one at a time. */
let manifestIndex = 0;

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.providers.map((provider) => probeProvider(ENV_URL, provider)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(probes, brief.ratings);
}

/** The free models across the providers that answered: what GET /v1/models lists, deduped by id. */
function servedModels(): { id: string; provider: string; display_name: string }[] {
  const seen = new Map<string, { id: string; provider: string; display_name: string }>();
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!seen.has(model.id)) seen.set(model.id, { id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return [...seen.values()].sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(clientApi: 'openai' | 'anthropic', body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    tools: Array.isArray(body.tools) && body.tools.length > 0,
    vision,
    json: format === 'json_object' || format === 'json_schema',
  };
}

/**
 * The requirements of the request now arriving: the manifest's row by arrival order, unioned
 * with what the body itself shows (tools, images, JSON mode), so a body that carries more than
 * its manifest row still routes to a model that can serve it.
 */
function requirementsFor(clientApi: 'openai' | 'anthropic', body: Json): Requirement {
  const manifest = brief.workload;
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  const fromBody = requirementsFromBody(clientApi, body);
  if (entry === undefined) return fromBody;
  const requires: Requirement = { ...entry.requires };
  if (fromBody.tools === true) requires.tools = true;
  if (fromBody.vision === true) requires.vision = true;
  if (fromBody.json === true) requires.json = true;
  return requires;
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

interface UpstreamCall {
  url: string;
  headers: Record<string, string>;
  body: Json;
}

/** The upstream request for one candidate: the path, headers and body in the provider's wire. */
function upstreamCall(candidate: Candidate, clientApi: 'openai' | 'anthropic', body: Json): UpstreamCall {
  const provider = candidate.probe.provider;
  const base = joinBase(ENV_URL, provider.base);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(provider.auth, provider.key),
  };
  if (body.stream === true) headers.accept = 'text/event-stream';
  if (provider.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  if (provider.wire === 'openai') {
    // An Anthropic-shaped client body must be rewritten into Chat Completions first: the
    // provider parses system messages, function tools and `stop`, not the Anthropic spellings.
    const upstreamBody: Json =
      clientApi === 'anthropic' ? anthropicRequestToOpenAI(body, candidate.model.id) : { ...body, model: candidate.model.id };
    if (body.stream === true) upstreamBody.stream_options = { include_usage: true };
    return { url: `${base}/v1/chat/completions`, headers, body: upstreamBody };
  }
  if (provider.wire === 'anthropic') {
    const upstreamBody = clientApi === 'openai' ? openAIRequestToAnthropic(body, candidate.model.id) : { ...body, model: candidate.model.id };
    return { url: `${base}/v1/messages`, headers, body: upstreamBody };
  }
  const built = clientApi === 'openai' ? openAIRequestToGemini(body, candidate.model.id) : anthropicRequestToGemini(body, candidate.model.id);
  return { url: `${base}${built.path}`, headers, body: built.body };
}

/** Copy a provider's event stream to the client as it arrives. */
async function pipeStream(stream: ReadableStream<Uint8Array>, res: ServerResponse): Promise<void> {
  const reader = stream.getReader();
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      res.write(Buffer.from(value));
    }
  } finally {
    reader.releaseLock();
    res.end();
  }
}

function sseHeaders(): Record<string, string> {
  return { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' };
}

/**
 * Answer one client request from one candidate. Returns true when the client was answered
 * (the response is already on its way or sent); false when the candidate refused or failed
 * and the caller should try the next one.
 */
async function tryCandidate(candidate: Candidate, clientApi: 'openai' | 'anthropic', body: Json, res: ServerResponse): Promise<boolean> {
  const provider = candidate.probe.provider;
  const stream = body.stream === true;
  const call = upstreamCall(candidate, clientApi, body);
  callsUsed.set(provider.id, callsOf(provider.id) + 1);
  let upstream: Response;
  try {
    upstream = await fetch(call.url, {
      method: 'POST',
      headers: call.headers,
      body: JSON.stringify(call.body),
      signal: AbortSignal.timeout(CALL_TIMEOUT_MS),
    });
  } catch (err) {
    log(`${provider.id}/${candidate.model.id} unreachable: ${(err as Error).message}`);
    return false;
  }
  if (!upstream.ok) {
    const text = await upstream.text().catch(() => '');
    log(`${provider.id}/${candidate.model.id} refused HTTP ${upstream.status}: ${text.slice(0, 200)}`);
    if (refusalExhausts(text)) exhausted.add(provider.id);
    return false;
  }
  const providerWire = provider.wire;
  if (stream) {
    if (upstream.body === null) {
      log(`${provider.id}/${candidate.model.id} answered 200 with no stream body`);
      return false;
    }
    res.writeHead(200, sseHeaders());
    try {
      if (providerWire === clientApi) {
        await pipeStream(upstream.body, res);
      } else if (providerWire === 'openai' && clientApi === 'anthropic') {
        for await (const event of anthropicEventsFromOpenAI(upstream.body, candidate.model.id)) res.write(event);
        res.end();
      } else if (providerWire === 'anthropic' && clientApi === 'openai') {
        for await (const event of openAIEventsFromAnthropic(upstream.body, candidate.model.id)) res.write(event);
        res.end();
      } else if (providerWire === 'gemini' && clientApi === 'openai') {
        for await (const event of openAIEventsFromGemini(upstream.body, candidate.model.id)) res.write(event);
        res.end();
      } else if (providerWire === 'gemini' && clientApi === 'anthropic') {
        for await (const event of anthropicEventsFromGemini(upstream.body, candidate.model.id)) res.write(event);
        res.end();
      } else {
        // No other wire pair exists: a gemini client is never served.
        await pipeStream(upstream.body, res);
      }
    } catch (err) {
      log(`${provider.id}/${candidate.model.id} stream failed: ${(err as Error).message}`);
      if (!res.writableEnded) res.end();
    }
    return true;
  }
  const answer: unknown = await upstream.json().catch(() => null);
  if (!isRecord(answer)) {
    log(`${provider.id}/${candidate.model.id} answered 200 with a body that is not JSON`);
    return false;
  }
  if (providerWire === clientApi) {
    sendJson(res, 200, answer);
    return true;
  }
  if (providerWire === 'openai' && clientApi === 'anthropic') {
    sendJson(res, 200, openAIAnswerToAnthropic(answer, candidate.model.id));
    return true;
  }
  if (providerWire === 'anthropic' && clientApi === 'openai') {
    sendJson(res, 200, anthropicAnswerToOpenAI(answer, candidate.model.id));
    return true;
  }
  if (providerWire === 'gemini' && clientApi === 'openai') {
    sendJson(res, 200, geminiAnswerToOpenAI(answer, candidate.model.id));
    return true;
  }
  if (providerWire === 'gemini' && clientApi === 'anthropic') {
    sendJson(res, 200, geminiAnswerToAnthropic(answer, candidate.model.id));
    return true;
  }
  return false;
}

/**
 * POST /v1/chat/completions or POST /v1/messages: route the request to the best free model
 * that meets it, failing over through the ranked candidates until one answers.
 */
async function serve(clientApi: 'openai' | 'anthropic', req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, clientApi, 400, 'invalid_request_error', 'the request body is not one JSON object');
  const requires = requirementsFor(clientApi, body);
  const jsonMode = clientApi === 'openai' && isRecord(body.response_format) && (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema');
  const candidates = candidatesFor(requires, clientApi, jsonMode, probes, brief.ratings);
  for (const candidate of candidates) {
    if (await tryCandidate(candidate, clientApi, body, res)) return;
  }
  log(`no provider answered a ${clientApi} request requiring ${JSON.stringify(requires)}`);
  return sendError(res, clientApi, 503, clientApi === 'anthropic' ? 'overloaded_error' : 'no_free_model', 'no free model on the directory answered this request');
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await serve('openai', req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await serve('anthropic', req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
