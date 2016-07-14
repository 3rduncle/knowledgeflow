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
	name = "exporter",
	deps = [
		"@tf_serving//tensorflow_serving/session_bundle:exporter",
		"@tf_serving//tensorflow_serving/session_bundle:session_bundle_py",
		"@org_tensorflow//tensorflow:tensorflow_py",
	],
)

py_binary(
	name = "apn",
	srcs = [
		"apn.py",
	],
	deps = [
		":exporter",
	],
)

py_binary(
	name = "sentences_similarity",
	srcs = [
		"sentences_similarity.py",
	],
	deps = [
		":exporter",
		"//knowledgeflow/utility:utility"
	],
)

