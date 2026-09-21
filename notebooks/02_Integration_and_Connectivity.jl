### A Pluto.jl notebook ###
# v1.0.3
#
# 02 — Integration and connectivity: connectors, incremental sync, idempotency.

using Markdown
using InteractiveUtils

# ╔═╡ 00000202-0000-0000-0000-000000000001
begin
	import Pkg
	Pkg.activate(normpath(joinpath(@__DIR__, "..")))
end

# ╔═╡ 00000202-0000-0000-0000-000000000002
begin
	using CarbonAccounting, CSV, DataFrames, Dates, Statistics, Printf
	datadir(parts...) = CarbonAccounting.datadir(parts...)
end

# ╔═╡ 00000202-0000-0000-0000-000000000003
md"""
# 02 · Integration and connectivity

Emissions data does not appear in a spreadsheet by magic: it is pulled from
meters, historians, ERPs, utility portals and supplier APIs. `src/connect.jl`
defines a small connector contract with the properties an audited inventory needs:

| property | mechanism |
|---|---|
| incremental sync | only rows newer than the stored high-water mark τ return (`since=…`) |
| idempotency | every row carries a deterministic key — a replayed page cannot double-count |
| resilience | transient failures retried with exponential back-off and jitter |
| provenance | the connector returns a `source_hash` and the next watermark |

Three connectors are implemented: a **folder** of exports, a **REST/JSON API**, and
in-memory rows. A mock ERP serves the demo API in-process, so this notebook runs
without a network.
"""

# ╔═╡ 00000202-0000-0000-0000-000000000004
md"### A folder connector — the 'our utility sends a CSV every month' interface"

# ╔═╡ 00000202-0000-0000-0000-000000000005
begin
	ledger = AuditLedger(hmac_key=Vector{UInt8}(codeunits("notebook-02")))
	service = Actor("svc-etl", "ETL Service", :service)
	folder = CSVFolderConnector(datadir("raw"), r"electricity.*\.csv$"i)
	pull(folder; ledger=ledger, actor=service, context="nightly folder sync")
end

# ╔═╡ 00000202-0000-0000-0000-000000000006
md"""
### A REST connector against an API that misbehaves

`MockERP` serves paginated JSON with `since` filtering and injects two `503`s, so the
retry logic is demonstrated rather than asserted. The pull reports how many HTTP
attempts it needed — evidence for the audit trail.
"""

# ╔═╡ 00000202-0000-0000-0000-000000000007
begin
	function erp_rows()
	    rows = Dict{String,String}[]
	    for r in CSV.File(datadir("raw", "activity_erp_2024.csv"); delim=';',
	                      types=String, missingstring=String[])
	        d = Dict{String,String}(string(k) => string(v) for (k, v) in pairs(r))
	        d["timestamp"] = "2024-03-01T00:00:00"
	        push!(rows, d)
	    end
	    rows
	end
	erp = MockERP(erp_rows(); port=8877, page_size=25, transients=2)
	start!(erp)
	erp
end

# ╔═╡ 00000202-0000-0000-0000-000000000008
begin
	conn = RESTConnector(erp_url(erp); page_size=25, token="demo-token")
	state = SyncState()
	records, quarantined, pulled = sync!(conn, state; ledger=ledger, actor=service,
	                                     context="first incremental sync")
	(pull=pulled, accepted=length(records), quarantined=length(quarantined),
	 watermark=state.watermark)
end

# ╔═╡ 00000202-0000-0000-0000-000000000009
md"""
### Idempotency: run the same sync again

The second run returns **zero new records**: every row's idempotency key is already
in the sync state. This is what protects the inventory from a replayed page after a
timeout, the most common way a connector silently doubles emissions.
"""

# ╔═╡ 00000202-0000-0000-0000-000000000010
begin
	again, _, pulled2 = sync!(conn, state; ledger=ledger, actor=service,
	                          context="replay after timeout")
	(replayed_rows_accepted=length(again), attempts=pulled2.attempts,
	 seen_keys=length(state.seen_keys))
end

# ╔═╡ 00000202-0000-0000-0000-000000000011
begin
	# the idempotency key is a hash of the row content: stable across runs, machines and order
	k1 = idempotency_key("rest:127.0.0.1", Dict("Kategorie" => "Erdgas", "Verbrauch" => "18000"))
	k2 = idempotency_key("rest:127.0.0.1", Dict("Verbrauch" => "18000", "Kategorie" => "Erdgas"))
	(description="key is content-addressed, so field order does not matter",
	 key=first(k1, 24), equal=(k1 == k2))
end

# ╔═╡ 00000202-0000-0000-0000-000000000012
md"### The synchronisation history — one row per pull, with attempts and payload hash"

# ╔═╡ 00000202-0000-0000-0000-000000000013
sync_history(state)

# ╔═╡ 00000202-0000-0000-0000-000000000014
begin
	stop!(erp)
	println("mock API hits            : ", erp.hits)
	println("records accepted         : ", length(records))
	println("watermark                : ", state.watermark)
	println("audit trail              : ", chain_statement(ledger))
	println("Notebook 02 — ingestion is incremental, replayed safely and logged with its payload hash.")
end

