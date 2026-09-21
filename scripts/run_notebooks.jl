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

# ── rendering Pluto's structured outputs to HTML ─────────────────────────────
# Pluto does not hand the *value* back for its own display MIMEs, it hands back the
# payload it would send to the browser (`Dict(:rows => …)`, `Dict(:elements => …)`).
# For a static printout we turn that payload back into HTML tables: a `Table` becomes
# a table of rows, a `Tree` a two-column key → value table, and the leaves are
# `(text, MIME)` pairs.
escape_html(s) = replace(string(s), '&' => "&amp;", '<' => "&lt;", '>' => "&gt;")

function render_pluto_value(x)
    if x isa AbstractDict
        render_pluto_object(x)
    elseif x isa Tuple && length(x) == 2 && x[2] isa MIME
        # Pluto wraps a value together with the MIME it was formatted for; when that
        # value is itself a table or a tree, keep rendering it as such.
        render_pluto_value(x[1])
    elseif x isa Pair
        string(escape_html(x.first), " → ", render_pluto_value(x.second))
    else
        s = string(x)
        # Pluto hands strings back in their display form, quotes included; a reader of a
        # printout wants the text, not the literal
        length(s) > 1 && startswith(s, '"') && endswith(s, '"') && (s = chop(s; head=1, tail=1))
        escape_html(s)
    end
end

"Pluto sometimes formats a key as `(name, MIME)` as well — take the name out of the pair."
function pluto_key(e)
    key = e isa Pair ? e.first : (e isa Tuple ? e[1] : e)
    key isa Tuple && length(key) == 2 && key[2] isa MIME ? key[1] : key
end

"Pluto's counterpart of a key: the payload, be it a pair, a tuple or a bare value."
pluto_value(e) = e isa Pair ? e.second : (e isa Tuple && length(e) == 2 ? e[2] : e)

function render_pluto_object(x)
    if x isa AbstractDict && haskey(x, :rows)
        io = IOBuffer()
        println(io, "<table><tbody>")
        for row in x[:rows]
            index, values = row isa Tuple && length(row) == 2 ? (row[1], row[2]) : ("", row)
            print(io, "<tr><th>", escape_html(index), "</th>")
            for v in (values isa AbstractVector || values isa Tuple ? values : [values])
                print(io, "<td>", render_pluto_value(v), "</td>")
            end
            println(io, "</tr>")
        end
        println(io, "</tbody></table>")
        return String(take!(io))
    elseif x isa AbstractDict && haskey(x, :elements)
        io = IOBuffer()
        println(io, "<table class=\"kv\"><tbody>")
        for e in x[:elements]
            println(io, "<tr><th>", escape_html(pluto_key(e)), "</th><td>",
                    render_pluto_value(pluto_value(e)), "</td></tr>")
        end
        println(io, "</tbody></table>")
        return String(take!(io))
    end
    escape_html(string(x))
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
        mime = string(c.output.mime)
        payload = if body isa Vector{UInt8}
            "base64:" * base64encode(body)          # 1. binary output (images)
        elseif startswith(mime, "application/vnd.pluto")
            # 2. Pluto's own structured outputs → HTML tables (see render_pluto_object)
            mime = "text/html"
            render_pluto_object(body)
        elseif body === nothing
            ""
        else
            string(body)
        end
        push!(cells, Dict{String,Any}(
            "id" => string(c.cell_id),
            "code" => c.code,
            "mime" => mime,
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
    # A notebook that produced no printout must not be quietly absent from the PDF: the
    # document claims to be the record of an execution, so a missing record is an error.
    failed = setdiff([splitext(basename(p))[1] for p in notebook_paths(names)], collect(keys(results)))
    isempty(failed) || error("no printout was written for: ", join(failed, ", "))
    results
end

main(isempty(ARGS) ? String[] : String.(ARGS))
