# CarbonAccounting.jl — Scope 1, 2 and 3 carbon accounting in Julia

A complete, auditable carbon-accounting platform in Julia: **emission-factor
libraries, Scope 1/2/3 calculation engines, data parsing and normalisation,
integration and connectivity, a neural soft sensor for the annual analyzer campaign,
anomaly detection, and a hash-chained audit trail** — with Pluto notebooks that
explain the mathematics, and a build pipeline that prints those notebooks to PDF.

```julia
julia --project=. -e 'using Pkg; Pkg.instantiate()'   # once
julia --project=. scripts/make_data.jl                # generate the synthetic data set
julia --project=. scripts/run_notebooks.jl            # run every notebook headlessly
julia --project=. scripts/build_pdf.jl --no-run       # → docs/CarbonAccounting_Notebooks.pdf
julia --project=. -e 'using Pkg; Pkg.test()'          # test suite
```

| notebook | what it shows |
|---|---|
| `00_Overview_and_Notation.jl` | the map of the platform and the symbolic legend |
| `01_Data_Parsing_and_Normalization.jl` | messy exports → typed, dimension-checked, traceable rows |
| `02_Integration_and_Connectivity.jl` | connectors, incremental sync, idempotency, retries |
| `03_Emission_Factor_Libraries.jl` | provenance, versions, validity windows, dimensional fit |
| `04_Scope1_Direct_Emissions.jl` | combustion, flaring, fugitive, LDAR, process emissions |
| `05_Scope2_Purchased_Energy.jl` | location **and** market based reporting, instruments, λ_cov |
| `06_Scope3_Value_Chain.jl` | the 15 categories, hybrid methods, EEIO, hotspots |
| `07_SoftSensor_Neural_Net.jl` | 168 campaign hours → the year, with honest uncertainty |
| `08_Anomaly_Detection.jl` | EWMA, CUSUM, Hotelling T², PCA SPE, autoencoder |
| `09_Audit_Trail_and_Security.jl` | RBAC, PBKDF2, hash chain, sealing, tamper detection |
| `10_Inventory_Rollup_Uncertainty_Disclosure.jl` | consolidation, GUM + Monte Carlo, DQI, disclosure, export |

## Three design rules

1. **The code carries the equation.** Every quantity keeps its mathematical symbol —
   `Q̇ᵢ`, `EFᵢ`, `σ̂`, `W⁽ˡ⁾`, `r̂ₜ`, `hₙ` — so a cell can be read next to the formula it
   implements. `CarbonAccounting.NOTATION` *is* the legend, and `check_notation()`
   asserts that every symbol in it is a legal Julia identifier, so documentation and
   source cannot drift apart.
2. **A number is never reported without its qualification.** Each `EmissionRow`
   carries the factor id, factor version and library, the consolidation share ©ᵢ, the
   data-quality class DQ, both uncertainty components (`u_act`, `u_fac`), the source
   hash of the file it came from and the line number.
3. **Nothing is silently repaired.** Rows that cannot be interpreted are quarantined
   with the reason (`report.quarantined`); every state change — ingestion,
   calculation, correction, approval — is appended to a hash chain
   `hₙ = SHA256(hₙ₋₁ ‖ entryₙ)`, which makes retro-active edits detectable.

## What is inside

```
src/
  notation.jl        symbolic legend, GWP sets (AR4/AR5/AR6), Σᵢ operator
  units.jl           dimensions, Quantity, q"…" macro, E = Q̇·EF with unit checking
  ingest.jl          parsing, normalisation, quarantine, validation, lineage
  connect.jl         CSV-folder / REST / in-memory connectors, sync state, mock ERP
  factors.jl         216-record versioned factor library + resolution rules
  engines.jl         Scope 1/2/3 engines, consolidation approaches, emission rows
  softsensor.jl      MLP from scratch (Adam, NLL), campaign design, annualisation
  anomaly.jl         EWMA, CUSUM, T², PCA SPE, autoencoder, case management
  security.jl        RBAC, sessions, PBKDF2, audit ledger, Merkle sealing
  uncertainty.jl     GUM propagation + Monte Carlo with correlated factor groups
  report.jl          consolidation, DQI, intensity, disclosure table, packages
scripts/
  make_data.jl       deterministic synthetic data (messy on purpose)
  run_notebooks.jl   execute the notebooks headlessly, capture every cell
  check_notebooks.jl inspect the captured cells (debugging builds)
  build_pdf.jl       printouts → HTML → PDF (headless Chrome, bundled MathJax)
  stamp_page_numbers.py  adds "page X of Y" footers (pypdf + reportlab)
data/                raw exports, reference data, the dumped factor library
notebooks/           the eleven Pluto notebooks
docs/                the generated PDF, figures, bundled MathJax
test/runtests.jl     unit tests (units, ledger, ingest, engines, NN, detectors)
```

## The mathematics in one page

| quantity | formula | function |
|---|---|---|
| combustion | `E = Σᵢ Q̇ᵢ/η_b·(EF_CO₂,ᵢ·OFᵢ + GWP_CH₄·EF_CH₄,ᵢ + GWP_N₂O·EF_N₂O,ᵢ)` | `combustion_emissions` |
| flaring | `E = V̇·EF·φ + GWP_CH₄·V̇·ρ_CH₄·x_CH₄·(1−φ)`, `φ = DRE·η_comb` | `flare_emissions` |
| refrigerant | `E = (C_start + C_purchased − C_end − C_recovered − C_disposed)·GWP₁₀₀` | `fugitive_refrigerant` |
| LDAR | `E = N · leak_rate · GWP_CH₄` | `fugitive_ldar` |
| process | `E = P · EF` (stoichiometry) | `process_emissions` |
| Scope 2 | `E_loc = Σᵢ(Cᵢ·EF_grid,y)`, `E_mkt = Σᵢ(Cᵢ·EF_contract) + residual` | `compute_scope2` |
| Scope 3 | `E = Σ_cat Q̇·EF`, spend / average / supplier-specific data | `compute_scope3` |
| soft sensor | `ŷ = f_θ⃗(x⃗)`, `L = ½[(μ−y)²e^{−s}+s]`, Adam, early stopping | `train_soft_sensor` |
| uncertainty | `u_c² = Σ_g[Σ_{i∈g}(Eᵢu_act,ᵢ)² + (A_g u_fac,g)²] + 2ρΣ A_g A_h u_fac,g u_fac,h` | `gum_uncertainty` |
| anomalies | `zₜ = λr̂ₜ+(1−λ)z_{t−1}`, `Sₜ⁺ = max(0,S+r̂ₜ−kσ̂)`, `T² = (x⃗−μ⃗)ᵀΣ⁻¹(x⃗−μ⃗)`, `SPE = ‖x⃗−PPᵀx⃗‖²` | `detect_anomalies` |
| audit | `hₙ = SHA256(hₙ₋₁ ‖ canonical(entryₙ))` | `record!`, `verify_chain` |

## About the data

Everything in `data/` is **synthetic**, generated by `scripts/make_data.jl` from a
fixed seed: six sites in five countries, one reporting year, deliberately messy
exports (semicolon delimiters, decimal commas, units glued to values, multi-language
headers, three unparseable rows), a 168-hour analyzer campaign and a year of hourly
DCS data with four injected anomalies. The emission factors are **published defaults**
(IPCC 2006, DEFRA 2023, US EPA, IEA/Ember, EPA USEEIO), so the repository runs
without any licensed dataset — replace them with your factor set through
`load_factors` before any real disclosure. The library is versioned, so changing a
factor never rewrites history.

## The semicolon in the path

This folder's name contains a semicolon (`Ju;ia_…`, a typo for `Julia`), which has two
consequences:

* **Julia** cannot use such a path as its active project on Windows — `LOAD_PATH`
  entries are separated by `;` there, so precompilation fails with *"LOAD_PATH
  entries cannot contain ';'"*. The build scripts therefore expect the repository
  reachable through a semicolon-free path:

  ```powershell
  New-Item -ItemType Junction -Path C:\ca_acct_repo -Target '<this folder>'
  julia --project=C:\ca_acct_repo scripts/run_notebooks.jl
  ```

  Renaming the folder to `Julia_CarbonAccounting_2026` removes the problem entirely —
  and that is also the GitHub repository name, because GitHub allows only letters,
  digits, `.`, `-` and `_` in a repository name.
* **GitHub** rejects `;`, so the remote is `Julia_CarbonAccounting_2026`.

## Reproducibility notes

* The notebooks activate the repository's single `Project.toml`/`Manifest.toml`
  (`Pkg.activate("../")`), so library, notebooks and scripts share one pinned
  environment — no per-notebook resolver run, no version drift.
* Code cells are written as `begin … end` blocks (one expression per cell), which
  keeps them portable across Pluto releases, and every notebook file ends with a
  blank line, because Pluto's loader expects a `\n\n` cell suffix.
* `docs/assets/mathjax/tex-svg.js` is bundled, so the PDF build needs no network:
  formulas are rendered as self-contained SVG.
* The synthetic data set is regenerated identically on every machine
  (`MersenneTwister(2026)`), and the PDF is built from the *captured* notebook
  outputs, never from re-executed code.

## Licence

MIT — see `LICENSE`. Copyright (c) 2026 caxelrud.
