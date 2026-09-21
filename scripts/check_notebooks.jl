# ══════════════════════════════════════════════════════════════════════════════
#  check_notebooks.jl — inspect the printouts produced by run_notebooks.jl
#
#  Run:  julia --project=. scripts/check_notebooks.jl [notebook-name …]
#
#  Prints, per notebook, every cell with its output MIME type and — for failing
#  cells — the error message, so a build log is enough to debug a notebook run.
# ══════════════════════════════════════════════════════════════════════════════
using JSON

const ROOT = normpath(joinpath(@__DIR__, ".."))
const OUT_DIR = joinpath(ROOT, "build", "printout")
const filter_names = String.(ARGS)

function check(path::AbstractString)
    d = JSON.parsefile(path)
    cells = d["cells"]
    nerr = count(c -> c["errored"], cells)
    println("── ", d["notebook"], "  (", length(cells), " cells, ", nerr, " errored, ",
            d["seconds"], " s, ", d["generated_at"], ")")
    for (i, c) in enumerate(cells)
        code = first(replace(string(c["code"]), r"\s+" => " "), 64)
        if c["errored"]
            msg = first(replace(string(c["body"]), r"\s+" => " "), 300)
            println("   ✗ ", lpad(i, 3), "  ", code)
            println("        ", msg)
        else
            body = string(c["body"])
            println("     ", lpad(i, 3), "  ", rpad(string(c["mime"]), 26),
                    rpad(string(length(body), " chars"), 14), code)
        end
    end
    println()
    (notebook=d["notebook"], cells=length(cells), errored=nerr)
end

files = sort(filter(f -> endswith(f, ".json") && f != "index.json", readdir(OUT_DIR)))
isempty(filter_names) || (files = filter(f -> any(n -> occursin(n, f), filter_names), files))
isempty(files) && error("no printouts in $(OUT_DIR) — run scripts/run_notebooks.jl first")

results = [check(joinpath(OUT_DIR, f)) for f in files]
println("="^72)
println("notebooks: ", length(results), "   cells: ", sum(r.cells for r in results),
        "   errored: ", sum(r.errored for r in results))
for r in results
    r.errored == 0 || println("   FAILING: ", r.notebook, " (", r.errored, ")")
end
