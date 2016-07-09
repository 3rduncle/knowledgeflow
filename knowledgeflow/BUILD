# Description: Tensorflow Serving.

package(
    default_visibility = ["//knowledgeflow:internal"],
)

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

# open source marker; do not remove

package_group(
    name = "internal",
    packages = [
        "//knowledgeflow/...",
    ],
)

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
            "g3doc/sitemap.md",
        ],
    ),
)
