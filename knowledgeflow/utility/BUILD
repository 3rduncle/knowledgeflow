package(
	default_visibility = ["//knowledgeflow:internal"],
	features = [
		"-parse_headers",
		"no_layering_check",
	],
)

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

filegroup(
	name = "all_files",
	srcs = glob(
		["**/*"],
		exclude = [
			"**/METADATA",
			"**/OWNERS",
		],
	),
)

py_library(
	name = "utility",
	srcs = [
		"utility.py",
		"qa_utils.py",
		"ws_utils.py",
		"letor_metrics.py",
	],
)
