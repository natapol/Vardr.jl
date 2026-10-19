# Build the documentation website:  julia --project=docs docs/make.jl
# Build a PDF manual instead:       julia --project=docs docs/make.jl pdf
using Vardr
using Documenter

DocMeta.setdocmeta!(Vardr, :DocTestSetup, :(using Vardr); recursive=true)

const PAGES = [
    "Home" => "index.md",
    "Tutorial" => "guide/tutorial.md",
    "User guide" => [
        "guide/models.md",
        "guide/estimation.md",
        "guide/whatif.md",
        "guide/calibration.md",
        "guide/live.md",
        "guide/troubleshooting.md",
    ],
    "Examples" => "examples.md",
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
    )
else
    makedocs(;
        modules=[Vardr],
        authors="Natapol Pornputtapong <natapol.p@chula.ac.th>",
        sitename="Vardr.jl",
        format=Documenter.HTML(;
            canonical="https://natapol.github.io/Vardr.jl",
            edit_link="main",
            assets=String[],
        ),
        pages=PAGES,
    )

    deploydocs(;
        repo="github.com/natapol/Vardr.jl",
        devbranch="main",
    )
end
