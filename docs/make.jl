using OptimKit
using Documenter

mathengine = MathJax3(
    Dict(
        :loader => Dict("load" => ["[tex]/physics"]),
        :tex => Dict(
            "inlineMath" => [["\$", "\$"], ["\\(", "\\)"]],
            "tags" => "ams",
            "packages" => ["base", "ams", "autoload", "physics"]
        )
    )
)
makedocs(;
    sitename = "OptimKit.jl",
    format = Documenter.HTML(;
        prettyurls = true,
        mathengine,
    ),
    pages = [
        "Home" => "index.md",
        "Library" => "lib.md",
    ],
    checkdocs = :exports,
)

deploydocs(; repo = "github.com/Jutho/OptimKit.jl.git")
