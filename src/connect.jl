# ══════════════════════════════════════════════════════════════════════════════
#  connect.jl — integration and connectivity
#
#  Emissions data does not appear in a spreadsheet by magic: it is pulled from
#  meters, historians, ERPs, utility portals and supplier APIs. This module
#  defines a small connector contract with the properties an audited inventory
#  needs:
#
#    • incremental sync — only records newer than the stored high-water mark
#      return (`since=…`), so a nightly job stays cheap;
#    • idempotency — every row carries a deterministic key, so a replayed page
#      cannot double-count emissions;
#    • resilience — transient failures are retried with exponential back-off and
#      jitter, and the attempt history is available for the audit trail;
#    • provenance — the connector returns a `source_hash` (SHA-256 of the
#      payload or of the files touched) and a watermark for the next run.
# ══════════════════════════════════════════════════════════════════════════════

"The contract every data source implements: pull rows newer than the watermark."
abstract type Connector end

"""
    PullResult(rows, watermark, source, source_hash, fetched_at, n_raw, pages, attempts)

Outcome of one synchronisation: normalised-but-unparsed rows (still strings, to
be handed to [`normalize_row`](@ref)), the new high-water mark, the provenance
hash, and how many HTTP attempts it took (for the audit trail).
"""
struct PullResult
    rows::Vector{Dict{String,String}}
    watermark::DateTime
    source::String
    source_hash::String
    fetched_at::DateTime
    n_raw::Int
    pages::Int
    attempts::Int
end

function Base.show(io::IO, r::PullResult)
    print(io, "PullResult(", length(r.rows), " rows, ", r.pages, " page(s), ",
          r.attempts, " attempt(s), watermark=", r.watermark, ", sha256=", first(r.source_hash, 12), "…)")
end

"""
    SyncState(watermark, seen_keys)

Persisted synchronisation state: the high-water mark τ and the set of
idempotency keys already accepted.
"""
mutable struct SyncState
    watermark::DateTime
    seen_keys::Set{String}
    history::Vector{NamedTuple{(:ts, :source, :rows, :attempts, :hash),Tuple{DateTime,String,Int,Int,String}}}
end
SyncState(; watermark::DateTime=DateTime(1970, 1, 1)) = SyncState(watermark, Set{String}(), [])

"Deterministic idempotency key of one row of one connector."
idempotency_key(source::AbstractString, row::AbstractDict) =
    sha256_hex(String(source) * "|" * join(sort([string(k, "=", v) for (k, v) in row]), "&"))

"""
    retry_with_backoff(f; attempts=4, base_delay=0.25, max_delay=4.0, isok=isnothing, rng)

Call `f()`, retrying transient failures with exponential back-off plus jitter —
the pattern that keeps nightly syncs alive through gateway timeouts and 503s.
Returns `(value, attempts)`.
"""
function retry_with_backoff(f::Function; attempts::Integer=4, base_delay::Real=0.25,
                            max_delay::Real=4.0, rng::AbstractRNG=Random.default_rng(),
                            retry_on=Exception -> true)
    last_error = nothing
    for i in 1:attempts
        try
            return (f(), i)
        catch e
            retry_on(e) || rethrow(e)
            last_error = e
            i == attempts && break
            sleep(min(max_delay, base_delay * 2.0^(i - 1)) * (0.5 + rand(rng)))
        end
    end
    throw(last_error)
end

# ── connector 1: a folder of exports (SFTP drop, share folder, mailbox) ──────
"""
    CSVFolderConnector(root, pattern; delim=nothing)

Pulls every delimited file in `root` matching `pattern` whose modification time is
newer than the watermark — the classic "our utility sends a CSV every month"
interface.
"""
struct CSVFolderConnector <: Connector
    root::String
    pattern::Regex
    delim::Union{Nothing,Char}
    name::String
end
CSVFolderConnector(root::AbstractString, pattern::Regex=r"\.csv$"i; delim::Union{Nothing,Char}=nothing) =
    CSVFolderConnector(String(root), pattern, delim, "csvfolder:" * basename(String(root)))

function pull(c::CSVFolderConnector; since::Union{Nothing,DateTime}=nothing,
              ledger::Union{Nothing,AuditLedger}=nothing, actor::Union{Nothing,Actor}=nothing,
              context::AbstractString="incremental sync")
    isdir(c.root) || throw(ArgumentError("connector root not found: $(c.root)"))
    files = sort(filter(f -> occursin(c.pattern, f), readdir(c.root; join=true)))
    fresh = since === nothing ? files : filter(f -> Dates.unix2datetime(mtime(f)) > since, files)
    rows, hashes = Dict{String,String}[], String[]
    for f in fresh
        h = sha256_file(f)
        push!(hashes, h)
        rd = read_delimited(f; delim=c.delim)
        for (i, r) in enumerate(rd.rows)
            r["__source"] = basename(f)
            r["__row"] = string(i + 1)
            push!(rows, r)
        end
    end
    τ = isempty(files) ? (since === nothing ? DateTime(1970, 1, 1) : since) :
        Dates.unix2datetime(maximum(mtime.(files)))
    res = PullResult(rows, τ, c.name, sha256_hex(join(hashes, "|")), now(UTC),
                     length(rows), length(fresh), 1)
    if ledger !== nothing && actor !== nothing
        record!(ledger, actor, "ingest", c.name; why=context, source_hash=res.source_hash,
                after="$(length(fresh)) new file(s), $(length(rows)) row(s)")
    end
    res
end

# ── connector 2: a paginated REST/JSON API ───────────────────────────────────
"""
    RESTConnector(base_url; endpoint, page_param, page_size, since_param, token, field_map)

Pulls activity data from an HTTP JSON API (ERP extract service, utility portal,
IoT gateway). The API contract assumed is the common one:

    GET {base_url}{endpoint}?page=1&size=50&since=2024-01-01T00:00:00
    → {"records":[{"quantity":"12.5 MWh","period":"2024-01"}, ...], "total":137}
"""
struct RESTConnector <: Connector
    base_url::String
    endpoint::String
    page_param::String
    page_size::Int
    since_param::String
    token::String
    field_map::Dict{String,String}
    max_pages::Int
    name::String
end

function RESTConnector(base_url::AbstractString; endpoint::AbstractString="/api/v1/activity",
                       page_param::AbstractString="page", page_size::Integer=50,
                       since_param::AbstractString="since", token::AbstractString="",
                       max_pages::Integer=1000,
                       field_map::AbstractDict=DEFAULT_FIELD_MAP)
    url = String(base_url)
    RESTConnector(String(url), String(endpoint), String(page_param), Int(page_size),
                  String(since_param), String(token), Dict{String,String}(field_map),
                  Int(max_pages), "rest:" * replace(String(url), r"^https?://" => ""))
end

function pull(c::RESTConnector; since::Union{Nothing,DateTime}=nothing,
              ledger::Union{Nothing,AuditLedger}=nothing, actor::Union{Nothing,Actor}=nothing,
              context::AbstractString="incremental sync")
    headers = c.token == "" ? ["Accept" => "application/json"] :
              ["Accept" => "application/json", "Authorization" => "Bearer " * c.token]
    rows, hashes, page, attempts = Dict{String,String}[], String[], 1, 0
    τ = something(since, DateTime(1970, 1, 1))
    while page <= c.max_pages
        url = string(c.base_url, c.endpoint, "?", c.page_param, "=", page,
                     "&size=", c.page_size,
                     since === nothing ? "" : string("&", c.since_param, "=",
                         Dates.format(since, dateformat"yyyy-mm-dd\THH:MM:SS")))
        local resp
        ((resp, a) = retry_with_backoff(() -> HTTP.get(url, headers; readtimeout=10),
            attempts=4, retry_on=e -> true))
        attempts += a
        payload = String(resp.body)
        push!(hashes, sha256_hex(payload))
        data = JSON.parse(payload)
        recs = get(data, "records", Any[])
        for r in recs
            d = Dict{String,String}()
            for (k, v) in r
                d[string(k)] = v === nothing ? "" : string(v)
            end
            push!(rows, d)
        end
        length(recs) < c.page_size && break
        page += 1
    end
    res = PullResult(rows, τ, c.name, sha256_hex(join(hashes, "|")), now(UTC),
                     length(rows), page, max(attempts, 1))
    if ledger !== nothing && actor !== nothing
        record!(ledger, actor, "ingest", c.name; why=context, source_hash=res.source_hash,
                after="$(length(rows)) row(s) over $(page - 1) page(s), $(res.attempts) attempt(s)")
    end
    res
end

# ── a mock ERP/SCADA API, so the connectivity demo runs fully offline ─────────
"""
    MockERP(rows; port=8855, page_size=50, transients=1)

In-process HTTP API that serves `rows` (string dictionaries) with pagination and
`since` filtering exactly like the real extract service, plus `transients`
deliberate `503` responses so the retry logic can be demonstrated without
touching a network.
"""
mutable struct MockERP
    rows::Vector{Dict{String,String}}
    port::Int
    page_size::Int
    transients::Int
    hits::Int
    server::Any
end
function MockERP(rows::AbstractVector; port::Integer=8855, page_size::Integer=50, transients::Integer=1)
    rs = [Dict{String,String}(string(k) => string(v) for (k, v) in r) for r in rows]
    MockERP(rs, Int(port), Int(page_size), Int(transients), 0, nothing)
end

Base.show(io::IO, m::MockERP) =
    print(io, "MockERP(port=", m.port, ", rows=", length(m.rows), ", ", m.hits, " hit(s), ",
          m.server === nothing ? "stopped" : "running", ")")

function _mock_handler(m::MockERP, req::HTTP.Request)
    m.hits += 1
    path = String(req.target)
    if startswith(path, "/health")
        return HTTP.Response(200, "ok")
    end
    if m.transients > 0
        m.transients -= 1
        return HTTP.Response(503, "temporarily unavailable (injected)")
    end
    q = HTTP.queryparams(HTTP.URI(path))
    page = parse(Int, get(q, "page", "1"))
    size = parse(Int, get(q, "size", string(m.page_size)))
    since = get(q, "since", "")
    rows = isempty(since) ? m.rows :
           [r for r in m.rows if get(r, "timestamp", "") >= since]
    slice = rows[min(end, (page - 1) * size + 1):min(length(rows), page * size)]
    body = JSON.json(Dict("records" => isempty(slice) ? Any[] : slice,
                          "total" => length(rows), "page" => page))
    HTTP.Response(200, ["Content-Type" => "application/json"], body)
end

"Start the mock API (idempotent) and wait until it answers `/health`."
function start!(m::MockERP)
    m.server === nothing || return m
    m.server = HTTP.serve!(m.port; verbose=-1) do req
        _mock_handler(m, req)
    end
    for _ in 1:60
        try
            HTTP.get("http://127.0.0.1:$(m.port)/health"; readtimeout=2, connect_timeout=1)
            return m
        catch
            sleep(0.1)
        end
    end
    error("MockERP did not become ready on port $(m.port)")
end

"Stop the mock API and release the port."
function stop!(m::MockERP)
    if m.server !== nothing
        HTTP.forceclose(m.server)
        m.server = nothing
    end
    m
end

"URL of the mock API, for a [`RESTConnector`](@ref)."
erp_url(m::MockERP) = "http://127.0.0.1:$(m.port)"

# ── connector 3: in-memory rows (unit tests, notebooks, replays) ─────────────
"Trivial connector over rows already in memory — used by the test suite."
struct InMemoryConnector <: Connector
    rows::Vector{Dict{String,String}}
    name::String
end
InMemoryConnector(rows::AbstractVector; name::AbstractString="memory:fixture") =
    InMemoryConnector([Dict{String,String}(string(k) => string(v) for (k, v) in r) for r in rows], String(name))

function pull(c::InMemoryConnector; since::Union{Nothing,DateTime}=nothing,
              ledger::Union{Nothing,AuditLedger}=nothing, actor::Union{Nothing,Actor}=nothing,
              context::AbstractString="replay")
    res = PullResult(c.rows, now(UTC), c.name,
                     sha256_hex(join([idempotency_key(c.name, r) for r in c.rows], "|")),
                     now(UTC), length(c.rows), 1, 1)
    if ledger !== nothing && actor !== nothing
        record!(ledger, actor, "ingest", c.name; why=context, source_hash=res.source_hash,
                after="$(length(c.rows)) row(s)")
    end
    res
end

# ── incremental, idempotent synchronisation ──────────────────────────────────
"Current high-water mark of a sync state (τ of the last successful pull)."
watermark(s::SyncState) = s.watermark

"""
    sync!(connector, state; kwargs...) -> (records, quarantined, pull_result)

One incremental run of a connector:
  1. pull rows newer than the watermark,
  2. drop rows whose idempotency key was already accepted (replay safety),
  3. normalise the rest through [`normalize_row`](@ref), quarantining what fails,
  4. advance the watermark and append the run to the synchronisation history.
"""
function sync!(c::Connector, state::SyncState;
               field_map::AbstractDict=DEFAULT_FIELD_MAP,
               dayfirst::Bool=true,
               ledger::Union{Nothing,AuditLedger}=nothing,
               actor::Union{Nothing,Actor}=nothing,
               context::AbstractString="scheduled sync")
    res = pull(c; since=state.watermark, ledger=ledger, actor=actor, context=context)
    records, quarantined, skipped = ActivityRecord[], QuarantinedRow[], 0
    for (i, raw) in enumerate(res.rows)
        k = idempotency_key(res.source, raw)
        if k in state.seen_keys
            skipped += 1
            continue
        end
        push!(state.seen_keys, k)
        try
            push!(records, normalize_row(raw; field_map=field_map, dayfirst=dayfirst,
                                         source=res.source, source_hash=res.source_hash,
                                         lineno=i))
        catch e
            push!(quarantined, QuarantinedRow(res.source, i,
                e isa ArgumentError ? e.msg : sprint(showerror, e), raw))
        end
    end
    state.watermark = max(state.watermark, res.watermark)
    push!(state.history, (ts=now(UTC), source=res.source, rows=length(records),
                          attempts=res.attempts, hash=res.source_hash))
    if ledger !== nothing && actor !== nothing && skipped > 0
        record!(ledger, actor, "ingest", res.source;
                why="idempotency filter of the sync: $skipped already-seen row(s) ignored",
                after="$(length(records)) accepted")
    end
    (records, quarantined, res)
end

"Synchronisation history as a DataFrame (one row per pull)."
function sync_history(s::SyncState)
    isempty(s.history) && return DataFrame(ts=DateTime[], source=String[], rows=Int[],
                                           attempts=Int[], hash=String[])
    DataFrame(ts=[h.ts for h in s.history], source=[h.source for h in s.history],
              rows=[h.rows for h in s.history], attempts=[h.attempts for h in s.history],
              hash=[first(h.hash, 16) for h in s.history])
end

"Persist a sync state as JSON next to the ledger, so restarts stay incremental."
function save_sync_state(s::SyncState, path::AbstractString)
    mkpath(dirname(path))
    write(path, JSON.json(Dict("watermark" => string(s.watermark),
                               "seen_keys" => collect(s.seen_keys),
                               "history" => [Dict("ts" => string(h.ts), "source" => h.source,
                                                  "rows" => h.rows, "attempts" => h.attempts,
                                                  "hash" => h.hash) for h in s.history])))
    path
end
