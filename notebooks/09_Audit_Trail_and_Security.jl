### A Pluto.jl notebook ###
# v1.0.3
#
# 09 — Audit trails and security: RBAC, hash-chained ledger, sealing, tamper detection.

using Markdown
using InteractiveUtils

# ╔═╡ 00000909-0000-0000-0000-000000000001
begin
	import Pkg
	Pkg.activate(normpath(joinpath(@__DIR__, "..")))
end

# ╔═╡ 00000909-0000-0000-0000-000000000002
begin
	using CarbonAccounting, CSV, DataFrames, Dates, Statistics, Printf
	datadir(parts...) = CarbonAccounting.datadir(parts...)
end

# ╔═╡ 00000909-0000-0000-0000-000000000003
md"""
# 09 · Audit trails and security

Two guarantees, and nothing else:

**1. Who may do what.** A role-based permission matrix rᵢⱼ with segregation of duties —
the analyst who prepares a figure cannot approve it — and passwords handled by
PBKDF2-HMAC-SHA256.

**2. Nothing can be changed unnoticed.** Every state change is appended to a hash
chain

    hₙ = SHA256( hₙ₋₁ ‖ canonical(entryₙ) ),        h₀ = SHA256("genesis")

so that altering any historical entry invalidates every subsequent hash. Batches are
sealed with a Merkle root and signed with HMAC-SHA256.
"""

# ╔═╡ 00000909-0000-0000-0000-000000000004
md"### The permission matrix rᵢⱼ"

# ╔═╡ 00000909-0000-0000-0000-000000000005
DataFrame(role=collect(keys(PERMISSIONS)),
          permissions=[join(sort([ACTION_NAMES[a] for a in PERMISSIONS[r]]), ", ")
                       for r in keys(PERMISSIONS)])

# ╔═╡ 00000909-0000-0000-0000-000000000006
begin
	analyst = Actor("u001", "A. Analyst", :analyst)
	approver = Actor("u002", "B. Approver", :approver)
	auditor = Actor("u003", "C. Auditor", :auditor)
	(can_calculate=can(analyst, CalculateAction), can_approve=can(analyst, ApproveAction),
	 denied=try
		 authorize(analyst, ApproveAction; context="approval of the FY2024 inventory")
		 "not denied — that would be a bug"
	 catch e
		 sprint(showerror, e)
	 end,
	 segregation=try
		 check_segregation_of_duties(analyst, analyst)
		 "allowed — that would be a bug"
	 catch e
		 first(sprint(showerror, e), 70)
	 end)
end

# ╔═╡ 00000909-0000-0000-0000-000000000007
md"### Credentials: PBKDF2, and secrets kept out of the log"

# ╔═╡ 00000909-0000-0000-0000-000000000008
begin
	cred = hash_password("u001", "correct horse battery staple"; iterations=10_000)
	(algorithm="PBKDF2-HMAC-SHA256", iterations=cred.iterations, hash_bytes=length(cred.hash),
	 accepted=check_password(cred, "correct horse battery staple"),
	 rejected=check_password(cred, "correct horse battery stapl"),
	 redacted=redact("api_key=SECRET123 written by user@example.com"))
end

# ╔═╡ 00000909-0000-0000-0000-000000000009
md"""
### The ledger: a calculation run writes itself into history

`compute_inventory` accepts a ledger and an actor; every engine writes its sub-total
with a *reason*, a source hash and the resulting value.
"""

# ╔═╡ 00000909-0000-0000-0000-000000000010
begin
	ledger = AuditLedger(hmac_key=Vector{UInt8}(codeunits("deployment-key-2026")))
	files = ["activity_erp_2024.csv", "electricity_utility_2024.csv", "procurement_2024.csv",
	         "logistics_travel_2024.csv", "waste_2024.csv"]
	report = ingest_files([datadir("raw", f) for f in files];
	                      entity_default="AcmeIndustrial_SA", ledger=ledger, actor=analyst,
	                      context="FY2024 close — scheduled ingestion")
	inventory = compute_inventory(report.records; consolidation=:operational,
	                              date=Date(2024, 12, 31), ledger=ledger, actor=analyst)
	(entries=length(ledger), head=first(head_hash(ledger), 24),
	 total=round(inventory.E_total; digits=1))
end

# ╔═╡ 00000909-0000-0000-0000-000000000011
begin
	trail = audit_trail_table(ledger)
	select(trail, :seq, :a_actor, :π_role, :action, :entity, :why, :after, :hₙ)
end

# ╔═╡ 00000909-0000-0000-0000-000000000012
md"""
### Verification, sealing and tamper detection

The chain is verified by re-computing every link; a manifest seals the current head
and Merkle root with an HMAC. Then an attacker edits a stored entry — and the
verifier stops exactly there.
"""

# ╔═╡ 00000909-0000-0000-0000-000000000013
begin
	ok, broken = verify_chain(ledger)
	manifest = seal(ledger)
	(chain_ok=ok, broken_at=broken, statement=chain_statement(ledger),
	 merkle_root=first(manifest.merkle, 24), signature=first(manifest.signature, 24),
	 seal_verifies=verify_seal(manifest, ledger))
end

# ╔═╡ 00000909-0000-0000-0000-000000000014
begin
	backup = joinpath(CarbonAccounting.builddir("notebook09"), "ledger_backup.json")
	save_ledger(ledger, backup)
	tamper!(ledger, 3; after="1 tCO₂e (forged value, no corresponding source)")
	ok2, broken2 = verify_chain(ledger)
	(forged_entry=3, chain_ok=ok2, broken_at=broken2, detected=(!ok2 && broken2 == 3),
	 seal_still_verifies=verify_seal(manifest, ledger))
end

# ╔═╡ 00000909-0000-0000-0000-000000000015
begin
	restored = load_ledger(backup; hmac_key=ledger.hmac_key)
	(restored_entries=length(restored), statement=chain_statement(restored),
	 verified=first(verify_chain(restored)))
end

# ╔═╡ 00000909-0000-0000-0000-000000000016
md"""
### Immutability by design: corrections are *entries*, not edits

An inventory is corrected by appending an adjustment entry that references the
original — the trail then shows both the mistake and the fix, which is what an
assurance provider wants to see.
"""

# ╔═╡ 00000909-0000-0000-0000-000000000017
begin
	reviewer = Actor("u004", "D. Reviewer", :approver)
	proposal = record!(ledger, analyst, "propose", "natural_gas 2024-06 site HAM-01";
	                   why="invoice correction: meter reading replaced by supplier statement",
	                   before="-8.000 GJ (wrong sign)", after="+9.100 GJ (supplier statement)",
	                   source_hash=sha256_hex("supplier statement 2024-06"))
	approval = record!(ledger, reviewer, "approve", "natural_gas 2024-06 site HAM-01";
	                   why="correction verified against the supplier statement",
	                   after="approved: +9.100 GJ")
	(proposal="entry $(proposal.seq) by $(proposal.actor_name) [$(proposal.role)]",
	 approval="entry $(approval.seq) by $(approval.actor_name) [$(approval.role)]",
	 prev_hash=first(approval.prev_hash, 24), statement=chain_statement(ledger))
end

# ╔═╡ 00000909-0000-0000-0000-000000000018
begin
	println("ledger entries   : ", length(ledger))
	println("chain            : ", chain_statement(ledger))
	println("tamper detection : entry 3 forged → detected at 3, restored from the trusted copy")
	println("access control   : the analyst cannot approve; segregation of duties enforced")
	println("Notebook 09 — every figure has a who, a when, a why and a source hash, and no history can change unnoticed.")
end

