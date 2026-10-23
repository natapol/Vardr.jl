# Build the documentation website:  julia --project=docs docs/make.jl
# Build a PDF manual instead:       julia --project=docs docs/make.jl pdf
using Vardr
using Documenter
using DocumenterCitations

const BIB = CitationBibliography(joinpath(@__DIR__, "src", "refs.bib"); style = :authoryear)

DocMeta.setdocmeta!(Vardr, :DocTestSetup, :(using Vardr); recursive=true)

const PAGES = [
    "Home" => "index.md",
    "Tutorial" => "guide/tutorial.md",
    "Step-by-step workflow" => "guide/workflow.md",
    "User guide" => [
        "guide/models.md",
        "guide/estimation.md",
        "guide/whatif.md",
        "guide/calibration.md",
        "guide/live.md",
        "guide/troubleshooting.md",
    ],
    "Case study: bioprocess twin" => "cases/bioreactor.md",
    "Theory" => [
        "theory/concepts.md",
        "theory/filtering.md",
        "theory/prediction.md",
    ],
    "Examples" => "examples.md",
    "References" => "references.md",
    "API" => "api.md",
]

if "pdf" in ARGS
    # Tectonic is a self-contained LaTeX engine, so no TeX installation is needed.
    using tectonic_jll: tectonic
    makedocs(;
        modules=[Vardr],
        authors="Natapol Pornputtapong",
        sitename="Vardr.jl",
        format=Documenter.LaTeX(; platform="tectonic", tectonic=tectonic()),
        build="build-pdf",
        pages=PAGES,
        plugins=[BIB],
    )
else
    makedocs(;
        modules=[Vardr],
        authors="Natapol Pornputtapong <natapol.p@chula.ac.th>",
        sitename="Vardr.jl",
        format=Documenter.HTML(;
            canonical="https://natapol.github.io/Vardr.jl",
            edit_link="main",
            assets=String["assets/citations.css"],
        ),
        pages=PAGES,
        plugins=[BIB],
    )

    deploydocs(;
        repo="github.com/natapol/Vardr.jl",
        devbranch="main",
    )
end
