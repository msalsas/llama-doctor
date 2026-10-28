# T-fix: reduce the base URL to the server origin before joining paths

I wrote this contract from the code, tests and docs. I changed no repo files and ran no git commands. I couldn't run my check script because the scratch write to `/tmp` wasn't approved. So the input/output tables below were traced by hand and still need to be confirmed by the implementation's tests.

## 0. Decisions you asked for

One of your preferences conflicts with a documented feature, so I chose differently:

- **A path like `/foo/bar` is kept, not rejected.** `docs/contracts/T1-core.md:159` says "Keep any path prefix (servers behind a reverse proxy)". `tests/test_client.py:51` checks that `http://h/proxy/` gives `http://h/proxy`. Rejecting every unknown path would break that test and anyone running behind an nginx subpath.
- **Typos still fail loudly.** After trimming the accepted suffixes, the URL is rejected if its last path segment looks like an API endpoint or a version number. That covers `/v2`, a leftover `/v1` and `/v1/models`.
- **`http://h/v1/v1` is rejected.** `/v1` is stripped once, which leaves `/v1`. That is a version segment, so it raises `ValueError`. There is no repeated stripping.
- **`/v1/models` is rejected, not trimmed.** Only the two suffixes from a pasted SDK base or curl URL are trimmed. Any other API path means the user pasted the wrong thing, so it fails instead of quietly probing something unintended.

If you want strict origin-only behaviour instead, it's a one-line change: reject any non-empty `path` in step 7. But then the T1 proxy promise and the `/proxy` test have to go too.

## 1. Files to touch

1. `src/llama_doctor/client.py`: `normalize_base_url` and two module constants.
2. `tests/test_client.py`: new tests in `TestNormalizeBaseUrl` and `TestLlamaClient`.
3. `tests/test_agentcheck_cli.py`: three new tests.
4. `docs/contracts/T1-core.md`: the `normalize_base_url` bullet (lines 157–159) and the test description (lines 265–267).
5. `docs/contracts/T6-agentcheck.md`: §5, §8 and §10.
6. `README.md`: the Quick start paragraph (line 41) and the agentcheck section (line 112).

No other code changes. `LlamaClient`, `CHAT_PATH`, `family.py` and the report code keep joining `base_url + path`. The `target:` line prints `client.base_url`, so it shows the trimmed URL without any further change.

## 2. Implementation (`client.py`)

Add `import re` in alphabetical order with the other stdlib imports. Add these after `MAX_BODY_BYTES`:

```python
_ORIGIN_SUFFIXES: tuple[str, ...] = ("/chat/completions", "/v1")
_ENDPOINT_SEGMENTS: frozenset[str] = frozenset(
    {"chat", "completion", "completions", "embeddings", "health", "metrics", "models", "props", "slots"}
)
```

The signature is unchanged:

```python
def normalize_base_url(url: str) -> str:
```

### Algorithm (exact order; no comments in the code)

1. If `"?" in url or "#" in url`, raise `ValueError("base URL must not contain '?' or '#'")`. This is unchanged.
2. `cleaned = url.strip().rstrip("/")`. Unchanged.
3. `parsed = urllib.parse.urlsplit(cleaned)`. Unchanged.
4. Run the existing checks, unchanged and in the same order: scheme, host, userinfo, port digits, port range. Their messages stay word for word. They run **before** the new path logic, so a bad port or scheme still reports its own error.
5. `path = parsed.path`.
6. For each `suffix` in `_ORIGIN_SUFFIXES`, in order and at most once each: if `path.endswith(suffix)`, then `path = path[: -len(suffix)].rstrip("/")`.
   - Every suffix starts with `/`, so matches only happen on whole segments: `/xv1` doesn't match.
   - Matching uses `parsed.path`, so a host named `v1` (`http://v1`) is never touched.
   - Matching is case-sensitive.
7. `last = path.rpartition("/")[2].lower()`. If `last in _ENDPOINT_SEGMENTS or re.fullmatch(r"v[0-9]+", last)`, raise:
   `ValueError("base URL path must be a server origin or proxy prefix, not an API path such as /v1/models")`
   The message doesn't echo the user's input, which matches the existing messages.
8. `return cleaned[: len(cleaned) - len(parsed.path)] + path`.
   - Steps 1–2 guarantee there's no query or fragment, so `cleaned` ends with `parsed.path`.
   - Slicing keeps the user's scheme and host spelling exactly, as today. Don't use `urlunsplit`: it lowercases the scheme.

## 3. Tests

### `tests/test_client.py`, `TestNormalizeBaseUrl`

Keep `test_normalize_base_url` unchanged. It must still pass.

Add `REJECT = "base URL path must be a server origin or proxy prefix, not an API path such as /v1/models"` at module level.

**`test_normalize_trims_openai_suffixes`**: in a `subTest` loop, `assertEqual(normalize_base_url(given), expected)` for each row:

| # | given | expected |
|---|---|---|
| 1 | `http://h:8888` (origin is unchanged) | `http://h:8888` |
| 2 | `http://h:8888/v1` | `http://h:8888` |
| 3 | `http://h:8888/v1/` | `http://h:8888` |
| 4 | `http://h:8888/v1/chat/completions` | `http://h:8888` |
| 5 | `" http://h:8888/v1/chat/completions/ "` | `http://h:8888` |
| 6 | `https://h/v1` | `https://h` |
| 7 | `http://h/chat/completions` | `http://h` |
| 8 | `http://h/v1//chat/completions` | `http://h` |
| 9 | `http://h/proxy/v1` | `http://h/proxy` |
| 10 | `http://h/proxy/v1/chat/completions/` | `http://h/proxy` |
| 11 | `http://h/foo/bar` (proxy prefix is kept) | `http://h/foo/bar` |
| 12 | `http://v1` | `http://v1` |
| 13 | `http://h/xv1` | `http://h/xv1` |
| 14 | `http://[::1]:8080/v1` | `http://[::1]:8080` |

**`test_normalize_rejects_api_paths`**: for each URL below, use `subTest` and `with self.assertRaises(ValueError) as ctx`, then `assertEqual(str(ctx.exception), REJECT)`:

- `http://h:8888/v1/models`
- `http://h/v2`
- `http://h/v1/v1`
- `http://h/V1`
- `http://h/v1/completions`
- `http://h/v1/chat`
- `http://h/props`
- `http://h/metrics`
- `http://h/slots`
- `http://h/proxy/v1/models`
- `http://h/v1/chat/completions/chat/completions`

**`test_normalize_existing_errors_take_precedence`**:

- `http://h:99999/v1/models` gives the message `"base URL port must be between 1 and 65535"`.
- `http://h/v1/models?x=1` gives `"base URL must not contain '?' or '#'"`.
- `ftp://h/v1` gives `"base URL scheme must be http or https"`.

### `tests/test_client.py`, `TestLlamaClient`

**`test_v1_base_url_requests_origin_paths`**:

1. Start `FakeLlamaServer` with route `/v1/models` 200, using `load_fixture("models_full.json")` with `application/json`.
2. Create `client = LlamaClient(server.url + "/v1")`.
3. Assert `client.base_url == server.url`.
4. Call `client.fetch_models()`. It must not raise.
5. Assert `[r[1] for r in server.requests] == ["/v1/models"]`.

### `tests/test_agentcheck_cli.py`

1. **`test_openai_style_base_url_is_trimmed`**:
   - Run `run_cli(["agentcheck", server.url + "/v1", "--replays", "2"])` inside `agent_server()`.
   - Assert `code == 0` and `f"target: {server.url}\n" in stdout`.
   - Assert `not any(r[1].startswith("/v1/v1") for r in server.requests)`.
   - Assert `("POST", CHAT_PATH)` and `("GET", "/v1/models")` are both in `{(r[0], r[1]) for r in server.requests}`.
2. **`test_full_chat_url_is_trimmed`**:
   - Run `run_cli(["agentcheck", server.url + "/v1/chat/completions", "--replays", "2", "--json"])`.
   - Assert `code == 0` and `json.loads(stdout)["target"] == server.url`.
3. **`test_api_path_url_is_rejected_without_requests`**:
   - Run `run_cli(["agentcheck", server.url + "/v1/models"])`.
   - Assert `code == 3`, `stdout == ""`, `stderr == "llama-doctor: " + REJECT + "\n"` and `server.requests == []`.
   - Define `REJECT` locally in this file.

Finish with `python3 -m unittest discover`, which must stay green.

## 4. Doc lines to update

**`docs/contracts/T1-core.md` lines 157–159.** Replace the bullet with:

> - `normalize_base_url`: strip whitespace and trailing `/`. Require scheme `http`/`https` and a host. Reject userinfo, query, and fragment with `ValueError`. Then trim a trailing `/chat/completions` and then a trailing `/v1`, at most once each, so OpenAI-SDK bases and full curl URLs reduce to the origin. Keep any other path prefix (servers behind a reverse proxy). Reject a remaining path whose last segment is an API endpoint name or a version like `v2`.

**`docs/contracts/T1-core.md` lines 265–267.** After `http://h/proxy`, add:

> `http://h:8888/v1` and `http://h:8888/v1/chat/completions` give `http://h:8888`; `http://h/v1/models`, `http://h/v2` and `http://h/v1/v1` raise ValueError

**`docs/contracts/T6-agentcheck.md` §5.** Add this paragraph after the parser block:

> The URL argument goes through `normalize_base_url` (T1). `http://host:8888/v1` and `http://host:8888/v1/chat/completions` both target `http://host:8888`, and the report's `target:` line prints that trimmed URL. Before this, the client sent `/v1/v1/chat/completions` (HTTP 405 on Unsloth Studio).

**`docs/contracts/T6-agentcheck.md` §10.** Add this entry:

> **URL prefixes are kept, not rejected.** Only a version or API-endpoint last segment fails, so reverse-proxy subpaths keep working.

**`docs/contracts/T6-agentcheck.md` §8 and README line 112.** Append the same sentence to the agentcheck paragraph in both places:

> URL is the server origin; an OpenAI-style base ending in `/v1`, or a full `/v1/chat/completions` URL, also works.

**README line 41.** After the colon-terminated sentence and before the code block, add:

> The URL is the server origin, optionally with a reverse-proxy prefix. A base ending in `/v1` or a full `/v1/chat/completions` URL is trimmed to the origin; other API paths such as `/v1/models` are rejected.

No current README or contract example shows a `/v1` URL. T6 line 831 already shows `target: http://127.0.0.1:8080`, so no examples need replacing.
