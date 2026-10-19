using Vardr
using Documenter

DocMeta.setdocmeta!(Vardr, :DocTestSetup, :(using Vardr); recursive=true)

makedocs(;
    modules=[Vardr],
    authors="Natapol Pornputtapong <natapol.p@chula.ac.th>",
    sitename="Vardr.jl",
    format=Documenter.HTML(;
        canonical="https://natapol.github.io/Vardr.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/natapol/Vardr.jl",
    devbranch="main",
)
