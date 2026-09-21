"""
    CarbonAccounting

Scope 1, 2 and 3 carbon accounting in Julia, written so that the code can be read
next to the equations: every quantity keeps its mathematical symbol
(`Q̇ᵢ`, `EFᵢ`, `E_Scope1`, `σ̂`, `W⁽ˡ⁾`, `r̂ₜ`, `hₙ`), and the legend of those symbols
is part of the package itself (`NOTATION`, `notation_markdown()`, `check_notation()`).

Modules
-------
| file               | responsibility |
|--------------------|----------------|
| `notation.jl`      | symbolic legend, GWP sets, Σᵢ operator |
| `units.jl`         | dimensions, quantities, `E = Q̇·EF` with unit checking |
| `security.jl`      | RBAC, sessions, hash-chained audit trail, Merkle sealing |
| `ingest.jl`        | parsing and normalisation of raw activity data |
| `connect.jl`       | integration: connectors, incremental sync, idempotency |
| `factors.jl`       | emission-factor library (GHG, DEFRA, IPCC, EPA, EEIO) with versioning |
| `engines.jl`       | calculation engines for Scope 1, 2 and 3 |
| `uncertainty.jl`   | GUM sensitivity propagation + Monte-Carlo inventory uncertainty |
| `softsensor.jl`    | neural soft sensor for the annual analyzer campaign |
| `anomaly.jl`       | EWMA/CUSUM/Hotelling-T²/PCA/autoencoder anomaly detection |
| `report.jl`        | consolidation, data-quality index, disclosure tables, exports |

Notebooks (Pluto) in `notebooks/` walk through each module; the generated PDF
`docs/CarbonAccounting_Notebooks.pdf` is a printout of those notebooks.
"""
module CarbonAccounting

using Dates
using Statistics
using LinearAlgebra
using Random
using Printf
using Markdown
using Logging
using Base64
using SHA
using JSON
using HTTP
using DataFrames
using CSV
using Tables
import StatsBase

# ── source includes (order matters: notation → units → everything else) ──────
include("notation.jl")
include("units.jl")
include("security.jl")
include("ingest.jl")
include("connect.jl")
include("factors.jl")
include("engines.jl")
include("uncertainty.jl")
include("softsensor.jl")
include("anomaly.jl")
include("report.jl")

# ── package paths ────────────────────────────────────────────────────────────
"Root folder of the package (the repository)."
pkgroot() = normpath(joinpath(@__DIR__, ".."))

"`root/data/…` — raw and reference data."
datadir(parts...) = joinpath(pkgroot(), "data", parts...)

"`root/docs/figures/…` — figures used by the notebooks and the PDF."
figdir(parts...) = joinpath(pkgroot(), "docs", "figures", parts...)

"`root/build/…` — intermediate build artefacts."
builddir(parts...) = joinpath(pkgroot(), "build", parts...)

"""
    save_figure(fig, name; dir, ext, saver) -> path

Save a plot into `docs/figures/` as SVG **and** PNG, so that the HTML/PDF build can
embed a vector image and the README can show a raster one.

The plotting package is deliberately *not* a dependency of this package: the saver
is looked up in `Main` (`using Plots` brings `savefig` into `Main`), or passed
explicitly through the `saver` keyword. This keeps a heavy plotting stack out of
the calculation engines and out of the precompilation path.
"""
function save_figure(fig, name::AbstractString; dir::AbstractString=figdir(),
                     ext=("svg", "png"), saver=nothing)
    mkpath(dir)
    save = saver === nothing ? (isdefined(Main, :savefig) ? Main.savefig : nothing) : saver
    save === nothing && throw(ArgumentError(
        "no figure saver available: load Plots (`using Plots`) in the notebook, or pass saver=savefig"))
    for e in ext
        save(fig, joinpath(dir, string(name, ".", e)))
    end
    joinpath(dir, string(name, ".svg"))
end

"Register the mathematical symbols used by the notebooks, e.g. `@symbols scope1`."
macro symbols(area)
    return :(symbol_list($(string(area))))
end

# ── public API ───────────────────────────────────────────────────────────────
export SymbolDef, NOTATION, NOTATION_AREAS, notation_table, notation_markdown, notation_latex,
    symbol_list, check_notation, GWPSet, GWP_AR4, GWP_AR5, GWP_AR6, co2e, Σᵢ, Vᵢⱼ,
    M_CO₂, M_CH₄, M_N₂O, M_C, ρ_CH₄, ρ_CO₂

export UnitDef, Quantity, CompoundUnit, UNITS, CANONICAL, UNIT_ALIASES, unit, has_unit,
    parse_quantity, parse_compound, adjust_ef, emission, emission_gas, value_in, val_canon,
    canonical, is_canonical, dimension, unitdef, PriceIndex, deflate, @q_str, @u_str

export Action, ACTION_NAMES, ReadAction, IngestAction, CalculateAction, ProposeAction,
    ApproveAction, VerifyAction, ExportAction, AdminAction, Actor, PERMISSIONS, can,
    authorize, AccessDenied,
    check_segregation_of_duties, Credential, hash_password, check_password, pbkdf2_sha256,
    Session, open_session, valid, revoke!, sha256_hex, sha256_file, genesis_hash, AuditEntry,
    AuditLedger, record!, verify_chain, chain_statement, tamper!, seal, verify_seal,
    merkle_root, save_ledger, load_ledger, ledger_json, redact, head_hash, audit_trail_table

export ActivityRecord, QuarantinedRow, IngestReport, ingest_file, ingest_files, normalize_row,
    read_delimited, map_fields, parse_number, parse_date, parse_period, header_key,
    lookup_source, iso_country, deduplicate, natural_key, validate_records, records_dataframe,
    quarantine_dataframe, ingest_summary, SOURCE_RULES, SourceRule, DEFAULT_FIELD_MAP,
    DATA_QUALITY_CLASSES, DEFAULT_ITEM_BY_RULE, EEIO_CATEGORIES, sector_hint

export Connector, RESTConnector, CSVFolderConnector, InMemoryConnector, MockERP, PullResult,
    pull, sync!, sync_history, save_sync_state, SyncState, watermark, idempotency_key,
    retry_with_backoff, start!, stop!, erp_url

export EFRecord, FactorLibrary, default_library, load_factors, export_factors,
    factors_dataframe, lookup_factor, lookup_factor_cat, resolve_factor, describe_factor,
    factor_uncertainty, compatibility, applies_on

export EmissionRow, emissions_dataframe, Scope1Result, Scope2Result, Scope3Result,
    InventoryResult, Scope1Result, combustion_emissions, flare_emissions, fugitive_refrigerant,
    fugitive_ldar, process_emissions, scope1_row, compute_scope1, compute_scope2, compute_scope3,
    compute_inventory, scope1_detail, scope2_detail, scope3_detail, contribution_table,
    scope1_breakdown, inventory_rows, total_rows, consolidate, equity_share, financial_control,
    operational_control, CONSOLIDATION_METHODS, SCOPE3_CATEGORY_NAMES, scope3_method,
    pick_factor, emission_row, row_u_rel, u_activity, DQ_UNCERTAINTY, DEFAULT_OXIDATION,
    DEFAULT_EFFICIENCY, SCOPE1_BUCKET, STATIONARY_FUELS, SINGLE_FACTOR_RULES,
    BIOGENIC_CO₂_PER_GJ

export UncertaintySpec, gaussian, lognormal, triangular, uniform_unc, sample, gum_uncertainty,
    monte_carlo_uncertainty, UncertaintySummary, uncertainty_table, uncertainty_statement

export MLP, forward, loss_and_grad, adam_step!, train_soft_sensor, train_ensemble,
    train_autoencoder, nparams, predict, predict_interval, predict_std, ensemble_interval,
    softsensor_metrics, SoftSensorMetrics, metrics_table, cross_validate_softsensor,
    training_history, standardize!, apply_standardisation, campaign_design,
    min_campaign_samples, annualize_with_softsensor, annualize_ratio, annualization_statement,
    quantile_normal, chi2_quantile, act, dact, reconstruct, reconstruction_error,
    feature_attribution

export ewma_chart, cusum_chart, calibrate_threshold, ARL0, fit_covariance,
    hotelling_t2, pca_spe, Autoencoder, AnomalyEvent, AnomalyReport, detect_anomalies,
    severity_of, summarize_events, events_dataframe, residual_frame, anomaly_emission_impact,
    log_anomalies!, anomaly_report

export DQI_CRITERIA, DQI_WEIGHTS, dqi_score, dqi_label, dqi_table, intensity_metrics,
    disclosure_table, compare_years, inventory_totals, export_inventory, report_markdown,
    pkgroot, datadir, figdir, builddir, save_figure

end # module
