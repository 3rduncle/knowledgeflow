# Description: Tensorflow Serving examples.

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
		"@org_tensorflow//tensorflow:tensorflow_py",
	],
)

cc_library(
    name = "serving",
    deps = [
        "@tf_serving//tensorflow_serving/servables/tensorflow:session_bundle_config_proto",
        "@tf_serving//tensorflow_serving/servables/tensorflow:session_bundle_factory",
        "@tf_serving//tensorflow_serving/session_bundle",
        "@tf_serving//tensorflow_serving/session_bundle:manifest_proto",
        "@tf_serving//tensorflow_serving/session_bundle:signature",
    ],
)

cc_library(
    name = "tensorflow",
    linkopts = ["-lm"],
    deps = [
        "@org_tensorflow//tensorflow/core:core_cpu",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "@org_tensorflow//tensorflow/core:tensorflow",
    ],
)

py_library(
    name = "mnist_input_data",
    srcs = ["mnist_input_data.py"],
)

py_binary(
    name = "mnist_export",
    srcs = [
        "mnist_export.py",
    ],
    deps = [
        ":mnist_input_data",
		":exporter",
    ],
)

py_binary(
    name = "partition_export",
    srcs = [
        "partition_export.py",
    ],
    deps = [
		":exporter",
    ],
)

cc_binary(
    name = "partition_inference",
    srcs = [
        "partition_inference.cc",
    ],
    linkopts = ["-lm"],
    deps = [
		":serving",
		":tensorflow",
    ],
)
