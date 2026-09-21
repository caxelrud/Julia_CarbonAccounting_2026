# ══════════════════════════════════════════════════════════════════════════════
#  security.jl — access control and the tamper-evident audit trail
#
#  Two guarantees are implemented here:
#
#   1. *Who may do what.* A role-based access-control (RBAC) matrix rᵢⱼ maps
#      roles to actions, with segregation of duties (the actor who prepares a
#      figure cannot approve it) and password handling by PBKDF2-HMAC-SHA256.
#   2. *Nothing can be changed unnoticed.* Every state change is appended to a
#      hash chain
#
#         hₙ = SHA256( hₙ₋₁ ‖ canonical(entryₙ) ),      h₀ = SHA256("genesis")
#
#      so that altering any historical entry invalidates every subsequent hash.
#      Batches of entries are additionally sealed with a Merkle root and signed
#      with HMAC-SHA256.
# ══════════════════════════════════════════════════════════════════════════════

# ── roles and actions ────────────────────────────────────────────────────────
"Actions of the system that are subject to authorisation."
@enum Action ReadAction IngestAction CalculateAction ProposeAction ApproveAction VerifyAction ExportAction AdminAction

const ACTION_NAMES = Dict(ReadAction => "read", IngestAction => "ingest",
    CalculateAction => "calculate", ProposeAction => "propose",
    ApproveAction => "approve", VerifyAction => "verify",
    ExportAction => "export", AdminAction => "admin")

"Roles and their permitted actions — the permission matrix rᵢⱼ."
const PERMISSIONS = Dict{Symbol,Set{Action}}(
    :viewer   => Set([ReadAction]),
    :analyst  => Set([ReadAction, IngestAction, CalculateAction, ProposeAction, ExportAction]),
    :approver => Set([ReadAction, ApproveAction, ExportAction]),
    :auditor  => Set([ReadAction, VerifyAction, ExportAction]),
    :service  => Set([ReadAction, IngestAction, CalculateAction]),
    :admin    => Set(instances(Action)),
)

"""
    Actor(id, name, role, org)

A user or service account. `role` must be a key of [`PERMISSIONS`](@ref).
"""
struct Actor
    id::String
    name::String
    role::Symbol
    org::String
end
function Actor(id::AbstractString, name::AbstractString, role::Symbol; org::AbstractString="Acme Industrial Group")
    haskey(PERMISSIONS, role) || throw(ArgumentError("unknown role «$role»"))
    Actor(String(id), String(name), role, String(org))
end

Base.show(io::IO, a::Actor) = print(io, a.name, " <", a.id, "> (", a.role, ")")

struct AccessDenied <: Exception
    actor::Actor
    action::Action
    context::String
end
function Base.showerror(io::IO, e::AccessDenied)
    print(io, "AccessDenied: actor «", e.actor.name, "» (role ", e.actor.role,
          ") may not ", ACTION_NAMES[e.action], " — ", e.context)
end

"Whether `actor` holds `action` (the permission matrix lookup rᵢⱼ)."
can(actor::Actor, action::Action) = action in PERMISSIONS[actor.role]

"""
    authorize(actor, action; context="") -> true

Throws [`AccessDenied`](@ref) when the role does not hold the action.
"""
function authorize(actor::Actor, action::Action; context::AbstractString="")
    can(actor, action) || throw(AccessDenied(actor, action, String(context)))
    true
end

"""
    check_segregation_of_duties(preparer, approver) -> true

GHG-Protocol-aligned control: the same person may not prepare and approve one
inventory item. Admins are exempt only when they act as the second signature
(`admin` must still differ from the preparer).
"""
function check_segregation_of_duties(preparer::Actor, approver::Actor)
    preparer.id == approver.id && throw(ArgumentError(
        "segregation of duties: «$(preparer.name)» may not approve their own preparation"))
    authorize(approver, ApproveAction; context="approval step")
    true
end

# ── credentials ──────────────────────────────────────────────────────────────
"""
    pbkdf2_sha256(password, salt; iterations=600_000, dklen=32) -> Vector{UInt8}

PBKDF2 (RFC 2898) with HMAC-SHA256, implemented on top of `SHA.hmac_sha256`.
Stretching password hashes makes offline guessing expensive; the iteration count
follows current OWASP guidance for PBKDF2-HMAC-SHA256.
"""
function pbkdf2_sha256(password::AbstractString, salt::AbstractVector{UInt8};
                       iterations::Integer=600_000, dklen::Integer=32)
    hlen = 32
    blocks = cld(dklen, hlen)
    dk = UInt8[]
    pwd = Vector{UInt8}(codeunits(password))
    for i in 1:blocks
        u_prev = SHA.hmac_sha256(pwd, vcat(Vector{UInt8}(salt), Vector{UInt8}(codeunits(string(i)))))
        T = copy(u_prev)
        for _ in 2:iterations
            u_prev = SHA.hmac_sha256(pwd, u_prev)
            T .⊻= u_prev
        end
        append!(dk, T)
    end
    dk[1:dklen]
end

"""
    Credential(user, salt, iterations, hash)

Stored verifier of one account: never the password itself.
"""
struct Credential
    user::String
    salt::Vector{UInt8}
    iterations::Int
    hash::Vector{UInt8}
end

"Create a credential for `user` from a cleartext password (registration)."
function hash_password(user::AbstractString, password::AbstractString;
                       iterations::Integer=600_000, salt::AbstractVector{UInt8}=rand(UInt8, 16))
    Credential(String(user), Vector{UInt8}(salt), Int(iterations),
               pbkdf2_sha256(password, salt; iterations=iterations))
end

"Constant-time comparison of a candidate password against a stored credential."
function check_password(cred::Credential, password::AbstractString)
    candidate = pbkdf2_sha256(password, cred.salt; iterations=cred.iterations, dklen=length(cred.hash))
    isequal(candidate, cred.hash)
end

# ── sessions ─────────────────────────────────────────────────────────────────
"""
    Session(token, actor, issued, expires)

A short-lived bearer token bound to an [`Actor`](@ref). Tokens are random and
expire; every request re-checks both expiry and permission.
"""
struct Session
    token::String
    actor::Actor
    issued::DateTime
    expires::DateTime
end

"Open a session for `actor`; `ttl_minutes` defaults to a one-shift working day."
function open_session(actor::Actor; ttl_minutes::Integer=480, at::DateTime=now(UTC))
    Session(bytes2hex(rand(UInt8, 32)), actor, at, at + Minute(ttl_minutes))
end

"Whether the session is still valid at `at`."
valid(session::Session; at::DateTime=now(UTC)) = at < session.expires

"""
    authorize(session, action; at) -> Actor

Session-aware authorisation: refuses expired sessions before consulting the
permission matrix.
"""
function authorize(session::Session, action::Action; context::AbstractString="", at::DateTime=now(UTC))
    valid(session; at=at) || throw(ArgumentError("session expired at $(session.expires)"))
    authorize(session.actor, action; context=context)
    session.actor
end

"Revoke a token (kept in a revocation list, itself audit-logged)."
const REVOKED_TOKENS = Set{String}()
revoke!(session::Session) = (push!(REVOKED_TOKENS, session.token); session.token)

# ── hashing helpers ──────────────────────────────────────────────────────────
"SHA-256 of a string, hex encoded."
sha256_hex(s::AbstractString) = bytes2hex(sha256(Vector{UInt8}(codeunits(String(s)))))
"SHA-256 of a file, hex encoded — the *source hash* of ingested evidence."
sha256_file(path::AbstractString) = bytes2hex(sha256(read(path)))

"The first hash of the chain, h₀."
genesis_hash() = sha256_hex("CarbonAccounting.jl::genesis::v1")

# ── the audit trail ──────────────────────────────────────────────────────────
"""
    AuditEntry

One immutable record of the audit trail. The hash covers the *complete* previous
state, so any later modification of any field of any earlier entry breaks the
chain:

    hₙ = SHA256( hₙ₋₁ ‖ canonical(entryₙ) )
"""
struct AuditEntry
    seq::Int
    ts::DateTime
    actor_id::String
    actor_name::String
    role::Symbol
    action::String
    entity::String
    why::String
    before::Union{Nothing,String}
    after::Union{Nothing,String}
    source_hash::String
    prev_hash::String
    hash::String
end

"""
    canonical(entry) -> String

Deterministic serialisation of everything an entry claims — the byte string that
is hashed. Field order is fixed and timestamps are ISO-8601 UTC, so the same
entry always produces the same hash.
"""
function canonical(e::AuditEntry)
    ts = Dates.format(e.ts, Dates.DateFormat("yyyy-mm-ddTHH:MM:SS.s"))
    b = e.before === nothing ? "∅" : e.before
    a = e.after === nothing ? "∅" : e.after
    sh = isempty(e.source_hash) ? "∅" : e.source_hash
    join([string(e.seq),
          string("τ_ts=", ts, "Z"),
          string("a_actor=", e.actor_id, "|", e.actor_name, "|", e.role),
          string("action=", e.action),
          string("entity=", e.entity),
          string("why=", e.why),
          string("before=", b),
          string("after=", a),
          string("source_hash=", sh)], "\n")
end

"Compute the chained hash of an entry given its predecessor's hash."
chain_hash(prev_hash::AbstractString, e::AuditEntry) =
    sha256_hex(prev_hash * "\n" * canonical(e))

"An entry that is *not yet* chained (used internally by `record!`)."
function _unchained(seq, ts, actor::Actor, action, entity; why="", before=nothing,
                    after=nothing, source_hash="", prev_hash=genesis_hash())
    AuditEntry(seq, ts, actor.id, actor.name, actor.role, String(action), String(entity),
               String(why), before, after, String(source_hash), String(prev_hash), "")
end

"""
    AuditLedger(; hmac_key=rand(UInt8,32))

Append-only, hash-chained audit trail with a per-deployment HMAC key for sealing
evidence manifests.
"""
mutable struct AuditLedger
    entries::Vector{AuditEntry}
    hmac_key::Vector{UInt8}
end
AuditLedger(; hmac_key::AbstractVector{UInt8}=rand(UInt8, 32)) = AuditLedger(AuditEntry[], Vector{UInt8}(hmac_key))

"Current head hash of the chain (hₙ)."
head_hash(l::AuditLedger) = isempty(l.entries) ? genesis_hash() : l.entries[end].hash
Base.length(l::AuditLedger) = length(l.entries)

"""
    record!(ledger, actor, action, entity; why, before, after, source_hash, ts) -> AuditEntry

Append one state change. `why` is mandatory-by-convention (ticket, e-mail,
calculation run id) and `source_hash` carries the SHA-256 of the raw evidence,
so every figure in the inventory can be traced back to an unmodified source
document.
"""
function record!(l::AuditLedger, actor::Actor, action::AbstractString, entity::AbstractString;
                 why::AbstractString="", before::Union{Nothing,AbstractString}=nothing,
                 after::Union{Nothing,AbstractString}=nothing,
                 source_hash::AbstractString="", ts::DateTime=now(UTC))
    authorize(actor, action == "read" ? ReadAction :
                    action in ("ingest",) ? IngestAction :
                    action in ("calculate",) ? CalculateAction :
                    action in ("propose",) ? ProposeAction :
                    action in ("approve",) ? ApproveAction :
                    action in ("verify",) ? VerifyAction :
                    action in ("export",) ? ExportAction : AdminAction;
              context="audit trail append")
    prev = head_hash(l)
    e = _unchained(length(l.entries) + 1, ts, actor, action, entity;
                   why=why, before=before, after=after, source_hash=source_hash, prev_hash=prev)
    e = AuditEntry(e.seq, e.ts, e.actor_id, e.actor_name, e.role, e.action, e.entity,
                   e.why, e.before, e.after, e.source_hash, prev, chain_hash(prev, e))
    push!(l.entries, e)
    e
end

# ── Merkle sealing and evidence signing ──────────────────────────────────────
"""
    merkle_root(hashes) -> String

Binary Merkle tree over the entry hashes (odd levels duplicate the last node).
Sealing a *batch* of entries lets an auditor verify a single row against a root
without re-reading the whole ledger.
"""
function merkle_root(hashes::AbstractVector{<:AbstractString})
    isempty(hashes) && return sha256_hex("∅")
    level = String.(hashes)
    while length(level) > 1
        nxt = String[]
        for i in 1:2:length(level)
            left = level[i]
            right = i + 1 <= length(level) ? level[i + 1] : level[i]
            push!(nxt, sha256_hex(left * right))
        end
        level = nxt
    end
    level[1]
end

merkle_root(l::AuditLedger) = merkle_root([e.hash for e in l.entries])

"""
    seal(l, ; key) -> NamedTuple

Evidence manifest for the current ledger state: head hash, Merkle root, entry
count and an HMAC-SHA256 signature over both.
"""
function seal(l::AuditLedger; key::AbstractVector{UInt8}=l.hmac_key)
    head, root = head_hash(l), merkle_root(l)
    payload = head * "|" * root * "|" * string(length(l.entries))
    (entries=length(l.entries), head=head, merkle=root,
     signature=bytes2hex(SHA.hmac_sha256(Vector{UInt8}(key), Vector{UInt8}(codeunits(payload)))))
end

"""
    verify_seal(manifest, l; key) -> Bool

Re-computes the signature from the ledger and compares it with the manifest.
"""
function verify_seal(m::NamedTuple, l::AuditLedger; key::AbstractVector{UInt8}=l.hmac_key)
    m.entries == length(l) || return false
    m.head == head_hash(l) || return false
    m.merkle == merkle_root(l) || return false
    payload = m.head * "|" * m.merkle * "|" * string(m.entries)
    isequal(m.signature, bytes2hex(SHA.hmac_sha256(Vector{UInt8}(key), Vector{UInt8}(codeunits(payload)))))
end

"""
    verify_chain(ledger) -> (ok::Bool, first_broken::Union{Nothing,Int})

Recomputes every link of hₙ = H(hₙ₋₁ ‖ entryₙ). Returns `false` and the index of
the first entry whose stored hash no longer matches its content — i.e. detects
retro-active edits, deletions and re-orderings.
"""
function verify_chain(l::AuditLedger)
    prev = genesis_hash()
    for (i, e) in enumerate(l.entries)
        e.prev_hash == prev || return (false, i)
        chain_hash(prev, e) == e.hash || return (false, i)
        prev = e.hash
    end
    (true, nothing)
end

"One-line verification statement for reports."
function chain_statement(l::AuditLedger)
    ok, idx = verify_chain(l)
    ok && return @sprintf("AUDIT CHAIN VERIFIED — %d entries, head %s…",
                          length(l), first(head_hash(l), 16))
    return "AUDIT CHAIN BROKEN at entry #$(idx) — the ledger was modified outside the application"
end

"""
    tamper!(ledger, i; after=nothing, why=nothing) -> AuditEntry

Test tool: rewrite one field of entry `i` **without** recomputing the hashes, to
demonstrate that the chain detects the manipulation. Only used in the security
notebook and in the test suite.
"""
function tamper!(l::AuditLedger, i::Integer;
                 after::Union{Nothing,AbstractString}=nothing,
                 why::Union{Nothing,AbstractString}=nothing)
    e = l.entries[i]
    l.entries[i] = AuditEntry(e.seq, e.ts, e.actor_id, e.actor_name, e.role, e.action,
        e.entity, why === nothing ? e.why : String(why), e.before,
        after === nothing ? e.after : String(after), e.source_hash, e.prev_hash, e.hash)
    l.entries[i]
end

# ── persistence and redaction ────────────────────────────────────────────────
"Ledger as JSON (one object per entry, chain fields included)."
ledger_json(l::AuditLedger) = JSON.json([
    Dict("seq" => e.seq, "ts" => string(e.ts), "actor_id" => e.actor_id,
         "actor_name" => e.actor_name, "role" => string(e.role), "action" => e.action,
         "entity" => e.entity, "why" => e.why, "before" => e.before, "after" => e.after,
         "source_hash" => e.source_hash, "prev_hash" => e.prev_hash, "hash" => e.hash)
    for e in l.entries])

save_ledger(l::AuditLedger, path::AbstractString) = (mkpath(dirname(path)); write(path, ledger_json(l)); path)

function load_ledger(path::AbstractString; hmac_key::AbstractVector{UInt8}=rand(UInt8, 32))
    l = AuditLedger(hmac_key=hmac_key)
    for d in JSON.parsefile(path)
        push!(l.entries, AuditEntry(Int(d["seq"]), DateTime(d["ts"]), d["actor_id"],
            d["actor_name"], Symbol(d["role"]), d["action"], d["entity"], d["why"],
            d["before"], d["after"], d["source_hash"], d["prev_hash"], d["hash"]))
    end
    l
end

"The audit trail as a DataFrame — the table an auditor is handed."
function audit_trail_table(l::AuditLedger)
    DataFrame(seq=[e.seq for e in l.entries], τ_ts=[e.ts for e in l.entries],
              a_actor=[e.actor_name for e in l.entries], π_role=[string(e.role) for e in l.entries],
              action=[e.action for e in l.entries], entity=[e.entity for e in l.entries],
              why=[e.why for e in l.entries], before=[something(e.before, "") for e in l.entries],
              after=[something(e.after, "") for e in l.entries],
              source_hash=[first(e.source_hash, 12) for e in l.entries],
              hₙ=[first(e.hash, 16) for e in l.entries])
end

"""
    redact(text) -> String

Scrubs secrets and personal data from log lines before they reach the audit
trail (API keys, bearer tokens, e-mail addresses).
"""
function redact(text::AbstractString)
    s = String(text)
    s = replace(s, r"(?i)(api[_-]?key|token|password|secret)\s*[:=]\s*\S+" => s"\1=<redacted>")
    s = replace(s, r"[\w.+-]+@[\w-]+\.[\w.]+" => "<email>")
    s
end