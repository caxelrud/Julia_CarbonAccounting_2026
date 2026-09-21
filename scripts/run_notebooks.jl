# ══════════════════════════════════════════════════════════════════════════════
#  run_notebooks.jl — execute every Pluto notebook headlessly and dump its cells
#
#  Run:  julia --project=. scripts/run_notebooks.jl [notebook-name …]
#
#  For each notebook this writes `build/printout/<name>.json` containing every
#  cell in document order with its rendered output (HTML, plain text or base64
#  image data), which `scripts/build_pdf.jl` then turns into the PDF printout.
#
#  Implementation notes
#  --------------------
#  * `Pluto.SessionActions.open(session, path; run_async=false)` runs the notebook
#    to completion before returning, and saves it back to disk — so the `.jl`
#    files in `notebooks/` end up with their cell UUIDs and the Pluto header
#    written by Pluto itself;
#  * notebooks are opened through a path without special characters: Pluto (and
#    Julia's LOAD_PATH) cannot handle a `;` in the active project path, which is
#    why the build scripts resolve the repository root first (see
#    `scripts/build_pdf.jl` and the README section "The semicolon in the path");
#  * the cell *output* is captured, not re-executed: what you see in the PDF is
#    what the notebook produced.
# ══════════════════════════════════════════════════════════════════════════════
using Pluto, Dates, JSON, SHA, Base64

const ROOT = normpath(joinpath(@__DIR__, ".."))
const NB_DIR = joinpath(ROOT, "notebooks")
const OUT_DIR = joinpath(ROOT, "build", "printout")
mkpath(OUT_DIR)

"All notebooks, in the order of their numeric prefix."
function notebook_paths(filter_names::Vector{String}=String[])
    files = sort(filter(f -> endswith(f, ".jl"), readdir(NB_DIR)))
    isempty(filter_names) && return [joinpath(NB_DIR, f) for f in files]
    [joinpath(NB_DIR, f) for f in files if any(n -> occursin(n, f), filter_names)]
end

"Run one notebook and return its cells as dictionaries."
function run_notebook(path::AbstractString)
    session = Pluto.ServerSession()
    # Keep the repository's notebook files exactly as they are: without this,
    # Pluto rewrites them (adding its own cell-order footer) after every run,
    # which makes the files churn in git for no reason.
    session.options.server.disable_writing_notebook_files = true
    t0 = time()
    nb = Pluto.SessionActions.open(session, path; run_async=false)
    elapsed = time() - t0
    cells = Dict{String,Any}[]
    for c in nb.cells
        body = c.output.body
        payload = if body isa Vector{UInt8}
            "base64:" * base64encode(body)
        elseif body === nothing
            ""
        else
            string(body)
        end
        push!(cells, Dict{String,Any}(
            "id" => string(c.cell_id),
            "code" => c.code,
            "mime" => string(c.output.mime),
            "body" => payload,
            "errored" => c.errored,
            "runtime_ms" => c.runtime === nothing ? 0 : round(Int, c.runtime / 1e6)))
    end
    Pluto.SessionActions.shutdown(session, nb)
    (cells=cells, seconds=elapsed, path=path)
end

"Write one notebook's printout data as JSON."
function write_printout(name::AbstractString, res)
    out = joinpath(OUT_DIR, string(name, ".json"))
    payload = Dict{String,Any}(
        "notebook" => name,
        "source" => relpath(res.path, ROOT),
        "generated_at" => string(now(UTC)),
        "seconds" => round(res.seconds; digits=2),
        "cells" => res.cells)
    write(out, JSON.json(payload, 2))
    out
end

function main(names::Vector{String})
    paths = notebook_paths(names)
    isempty(paths) && error("no notebooks matched $(names)")
    results = Dict{String,Any}()
    for path in paths
        name = splitext(basename(path))[1]
        print(rpad(name, 46), " … ")
        flush(stdout)
        res = try
            run_notebook(path)
        catch err
            println("FAILED: ", sprint(showerror, err))
            continue
        end
        out = write_printout(name, res)
        nerr = count(c -> c["errored"], res.cells)
        println(length(res.cells), " cells, ", nerr, " errored, ",
                round(res.seconds; digits=1), " s → ", relpath(out, ROOT))
        # report every failing cell so that a build log is enough to debug it
        for (i, c) in enumerate(res.cells)
            c["errored"] || continue
            snippet = first(replace(string(c["code"]), r"\s+" => " "), 70)
            msg = first(replace(string(c["body"]), r"\s+" => " "), 240)
            println("   ✗ cell ", i, ": ", snippet)
            println("       ", msg)
        end
        results[name] = Dict("seconds" => res.seconds, "cells" => length(res.cells),
                             "errored" => nerr)
    end
    write(joinpath(OUT_DIR, "index.json"), JSON.json(results, 2))
    total_err = sum(v["errored"] for v in values(results); init=0)
    println("\nprintouts in ", relpath(OUT_DIR, ROOT), "  ·  ",
            length(results), " notebook(s), ", total_err, " errored cell(s)")
    results
end

main(isempty(ARGS) ? String[] : String.(ARGS))
