# ══════════════════════════════════════════════════════════════════════════════
#  build_pdf.jl — render the notebook printouts into one PDF
#
#  Run:  julia --project=. scripts/build_pdf.jl [--no-run] [--html-only]
#
#  Pipeline
#  --------
#   1. (unless --no-run) execute every notebook headlessly via run_notebooks.jl,
#      which writes `build/printout/<notebook>.json` with each cell's rendered output;
#   2. render one self-contained HTML document: cover page, table of contents, then
#      each notebook with its markdown, its code and its *actual* output;
#   3. print that HTML to PDF with headless Chrome/Edge — no network needed, MathJax
#      is embedded from `docs/assets/mathjax/` — and stamp page numbers with
#      `scripts/stamp_page_numbers.py` when Python, pypdf and reportlab are present.
#
#  Result: `docs/CarbonAccounting_Notebooks.pdf`.
# ══════════════════════════════════════════════════════════════════════════════
using JSON, Dates, Printf, Base64

const ROOT = normpath(joinpath(@__DIR__, ".."))
const OUT_DIR = joinpath(ROOT, "build", "printout")
const DOCS = joinpath(ROOT, "docs")
const HTML_OUT = joinpath(ROOT, "build", "CarbonAccounting_Notebooks.html")
const PDF_OUT = joinpath(DOCS, "CarbonAccounting_Notebooks.pdf")
const MATHJAX = joinpath(DOCS, "assets", "mathjax", "tex-svg.js")
const CHROME_CANDIDATES = [
    "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
    "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe",
    "/usr/bin/google-chrome", "/usr/bin/chromium", "/usr/bin/chromium-browser",
]

"Notebook printouts in presentation order."
function printouts()
    files = sort(filter(f -> endswith(f, ".json") && f != "index.json", readdir(OUT_DIR)))
    [(name=replace(f, ".json" => ""), data=JSON.parsefile(joinpath(OUT_DIR, f))) for f in files]
end

"Download MathJax once, so the PDF build works offline ever after."
function ensure_mathjax()
    isfile(MATHJAX) && return MATHJAX
    mkpath(dirname(MATHJAX))
    try
        run(`curl -sL -o $MATHJAX https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js`)
        @info "downloaded MathJax (formulas render offline from now on)" MATHJAX
    catch err
        @warn "could not download MathJax — LaTeX will appear as source text" err
    end
    MATHJAX
end

"Escape text that is inserted as HTML text content."
escape_html(s) = replace(string(s), '&' => "&amp;", '<' => "&lt;", '>' => "&gt;")

"""
    render_cell(cell) -> String

One cell as HTML: its code in a `<pre>`, then its output — raw HTML when the cell
rendered HTML (markdown cells, tables), **inline SVG** for plots, a data-URI `<img>`
for raster images, plain text otherwise. A *failing* cell is printed as a failure:
a printout that hides errors would be a lie.

Plots are inlined as SVG rather than referenced as an `<img>`: the vector graphics
survive the PDF print at full quality, need no decoding step, and cannot end up
mis-labelled (a data URI whose declared MIME does not match its payload renders as
nothing at all).
"""
function render_cell(cell)
    code = string(cell["code"])
    # skip the environment-activation plumbing: it carries nothing for a reader and is
    # documented in the README
    (occursin("Pkg.activate", code) && ncodeunits(code) < 400) && return ""
    io = IOBuffer()
    println(io, "<div class=\"cell\"><pre class=\"code\"><code>", escape_html(code), "</code></pre>")
    body, mime = string(cell["body"]), lowercase(string(cell["mime"]))
    println(io, "<div class=\"output\">")
    if cell["errored"]
        println(io, "<div class=\"error\"><strong>✗ this cell failed during the build</strong><br>",
                escape_html(first(body, 600)), "</div>")
    elseif startswith(body, "base64:") && occursin("svg", mime)
        svg = replace(String(base64decode(body[8:end])), r"<\?xml[^>]*\?>" => "")
        println(io, "<div class=\"fig\">", svg, "</div>")
    elseif startswith(body, "base64:")
        println(io, "<img class=\"fig\" src=\"data:", mime, ";base64,", body[8:end], "\">")
    elseif occursin("svg", mime) && occursin("<svg", body)
        println(io, "<div class=\"fig\">", replace(body, r"<\?xml[^>]*\?>" => ""), "</div>")
    elseif startswith(mime, "image/")
        println(io, "<p class=\"muted\">[image output ", mime, "]</p>")
    elseif startswith(mime, "text/html") || startswith(body, "<svg") ||
           startswith(body, "<div") || startswith(body, "<table")
        println(io, body)
    else
        println(io, "<pre class=\"text\">", escape_html(body), "</pre>")
    end
    println(io, "</div></div>")
    String(take!(io))
end

const CSS = """
@page { size: A4; margin: 14mm 12mm 16mm 12mm; }
body { font-family: "Segoe UI", Helvetica, Arial, sans-serif; font-size: 10.5pt;
       color: #16181d; margin: 0; line-height: 1.45; }
h1 { font-size: 17pt; border-bottom: 2px solid #2b6cb0; padding-bottom: 3px; }
h2 { font-size: 13pt; color: #2b6cb0; }
h3 { font-size: 11.5pt; color: #2c5282; }
section.cover { page-break-after: always; padding-top: 28mm; }
section.cover h2 { font-size: 12pt; color: #444; font-weight: normal; }
section.toc { page-break-after: always; }
section.notebook { page-break-before: always; }
.cell { margin: 0 0 9px 0; page-break-inside: avoid; }
pre.code { background: #f6f8fa; border: 1px solid #dfe3e8; border-left: 3px solid #2b6cb0;
           padding: 6px 8px; margin: 0 0 2px 0; font-size: 8.7pt; white-space: pre-wrap;
           font-family: "Cascadia Mono", Consolas, "DejaVu Sans Mono", monospace; }
pre.text { background: #fbfbfc; border: 1px solid #eceff3; padding: 5px 8px; margin: 0;
           font-size: 9pt; white-space: pre-wrap;
           font-family: "Cascadia Mono", Consolas, monospace; }
.output { margin: 0 0 4px 0; }
.output table { border-collapse: collapse; font-size: 8.6pt; margin: 2px 0; }
.output th, .output td { border: 1px solid #d5dae0; padding: 2px 5px; }
.output th { background: #eef2f7; }
.output table.kv th { background: #f8fafc; text-align: left; font-weight: 600;
                      white-space: nowrap; }
.output img, .output svg { max-width: 100%; }
.fig { margin: 4px 0 8px 0; page-break-inside: avoid; }
.fig svg { width: 100% !important; height: auto !important; }
.error { background: #fff5f5; border: 1px solid #fc8181; color: #822727; padding: 6px 8px;
         font-size: 9pt; }
.meta { color: #666; font-size: 9pt; }
.muted { color: #888; }
code { font-family: "Cascadia Mono", Consolas, monospace; font-size: 9pt; }
"""

"The whole printout as one HTML document: cover, contents, then one section per notebook."
function render_html(pages)
    io = IOBuffer()
    println(io, "<!DOCTYPE html><html><head><meta charset=\"utf-8\">")
    println(io, "<title>Carbon Accounting in Julia — Pluto notebook printout</title>")
    if isfile(MATHJAX)
        println(io, "<script src=\"data:text/javascript;base64,",
                base64encode(read(MATHJAX)), "\"></script>")
    else
        println(io, "<script src=\"https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js\"></script>")
    end
    println(io, "<style>", CSS, "</style></head><body>")
    println(io, """
    <section class="cover">
      <h1>Carbon Accounting in Julia</h1>
      <h2>Scope 1, 2 and 3 · audit trails and security · data parsing and normalisation ·
      calculation engines and emission-factor libraries · integration and connectivity ·
      a neural soft sensor for the annual analyzer campaign · anomaly detection</h2>
      <p class="meta">Pluto notebook printout · $(length(pages)) notebooks ·
      generated $(Dates.format(now(UTC), "yyyy-mm-dd HH:MM")) UTC</p>
      <p class="meta">Everything in this document is the *output of the notebooks as they
      were executed*: the build runs each notebook headlessly, captures its cells and
      renders them here.</p>
      <p class="meta">All data is synthetic and generated deterministically by
      <code>scripts/make_data.jl</code>. Emission factors are published defaults — replace
      them with a licensed factor set before any real disclosure.</p>
    </section>
    """)
    println(io, "<section class=\"toc\"><h1>Contents</h1><ol>")
    for p in pages
        println(io, "<li><strong>", escape_html(p.name), "</strong></li>")
    end
    println(io, "</ol></section>")
    for p in pages
        cells = p.data["cells"]
        nerr = count(c -> c["errored"], cells)
        println(io, "<section class=\"notebook\">")
        println(io, "<h1>", escape_html(p.name), "</h1>")
        @printf(io, "<p class=\"meta\">%d cells · executed in %s s%s</p>\n", length(cells),
                p.data["seconds"],
                nerr == 0 ? "" : string(" · ", nerr, " failing cell(s) printed below"))
        for cell in cells
            print(io, render_cell(cell))
        end
        println(io, "</section>")
    end
    println(io, "</body></html>")
    String(take!(io))
end

"Locate a Chromium-based browser for the headless print."
function find_browser()
    for c in CHROME_CANDIDATES
        isfile(c) && return c
    end
    nothing
end

"Print `html_path` to `pdf_path` with headless Chrome/Edge."
function print_to_pdf(html_path, pdf_path)
    browser = find_browser()
    browser === nothing && error("no Chrome/Edge found — open $html_path and print it manually")
    mkpath(dirname(pdf_path))
    run(`$browser --headless=new --disable-gpu --no-sandbox --no-pdf-header-footer
         --run-all-compositor-stages-before-draw --virtual-time-budget=180000
         --print-to-pdf=$pdf_path file:///$html_path`)
    pdf_path
end

"Add 'Page X of Y' footers, replacing the PDF in place when the helper succeeds."
function stamp_page_numbers(pdf_path)
    script = joinpath(ROOT, "scripts", "stamp_page_numbers.py")
    isfile(script) || return false
    numbered = joinpath(dirname(pdf_path), "CarbonAccounting_Notebooks_numbered.pdf")
    try
        run(`python $script $pdf_path $numbered`)
        mv(numbered, pdf_path; force=true)      # the numbered copy becomes the deliverable
        true
    catch err
        @warn "page numbering skipped (needs python + pypdf + reportlab)" err
        isfile(numbered) && rm(numbered; force=true)
        false
    end
end

"""
    check_render(html, pages)

Verify the rendered document before it is printed: every cell that produced an image MIME
must have produced a figure in the HTML, and no raw Pluto payload (`Dict{Symbol, Any}`)
may survive into the output.

This guard exists because the first release of this document was wrong in a way nobody
could see: the plot payloads were written as `data:image/png;base64,<SVG>` — an image
whose declared type did not match its content renders as *nothing*, so the PDF looked
complete, with all the code and all the tables, and had no figures in it at all. A silent
failure in the record of an execution is the worst kind; the build now refuses to print it.
"""
function check_render(html, pages)
    figures = length(findall("<svg", html)) + length(findall("<img class=\"fig\"", html))
    tables = length(findall("<table", html))
    leaks = length(findall("Dict{Symbol, Any}", html))
    expected = length([c for p in pages for c in p.data["cells"]
                       if startswith(string(c["mime"]), "image/")])
    n_notebooks = length(filter(f -> endswith(f, ".jl"), readdir(joinpath(ROOT, "notebooks"))))
    length(pages) == n_notebooks || error("$(length(pages)) printout(s) for $(n_notebooks) " *
                                          "notebook(s) — the document would be missing one")
    leaks == 0 || error("$(leaks) raw Pluto payload(s) leaked into the printout — " *
                        "render_pluto_object should have turned them into tables")
    figures >= expected || error("$(expected) cell(s) produced an image but only " *
                                 "$(figures) figure(s) were rendered")
    (; figures, tables, expected)
end

function main(args)
    if !("--no-run" in args)
        run(`julia --project=$ROOT --startup-file=no $(joinpath(ROOT, "scripts", "run_notebooks.jl"))`)
    end
    pages = printouts()
    isempty(pages) && error("no printouts in $(OUT_DIR) — run scripts/run_notebooks.jl first")
    ensure_mathjax()
    html = render_html(pages)
    st = check_render(html, pages)
    @info "printout verified — no silent losses" figures=st.figures tables=st.tables
    mkpath(dirname(HTML_OUT))
    write(HTML_OUT, html)
    @info "rendered HTML" file=HTML_OUT kb=round(Int, length(html) / 1024)
    "--html-only" in args && return HTML_OUT
    print_to_pdf(HTML_OUT, PDF_OUT)
    stamp_page_numbers(PDF_OUT)
    @info "wrote PDF" file=PDF_OUT kb=round(Int, filesize(PDF_OUT) / 1024) notebooks=length(pages)
    PDF_OUT
end

main(collect(String, ARGS))
