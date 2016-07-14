/* Copyright 2016 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// y = x * w
// 这个例子演示了tensorflow如何只导出部分模型作为serving使用。
// 导出的的例子参考partition_export.py
// 整个模型为y = (x + 1) * w
// 从模型中间变量开始导出
// 导出模型为y = x * w
// service服务时可以对中间变量feed，完成剩余计算

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <iostream>

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/util/command_line_flags.h"
#include "tensorflow_serving/servables/tensorflow/session_bundle_config.pb.h"
#include "tensorflow_serving/servables/tensorflow/session_bundle_factory.h"
#include "tensorflow_serving/session_bundle/manifest.pb.h"
#include "tensorflow_serving/session_bundle/session_bundle.h"
#include "tensorflow_serving/session_bundle/signature.h"

using tensorflow::serving::ClassificationSignature;
using tensorflow::serving::GenericSignature;
using tensorflow::serving::BatchingParameters;
using tensorflow::serving::SessionBundle;
using tensorflow::serving::SessionBundleConfig;
using tensorflow::serving::SessionBundleFactory;
using tensorflow::string;
using tensorflow::Tensor;
using tensorflow::TensorShape;
using tensorflow::Status;

class ServiceImpl final {
public:
    explicit ServiceImpl(std::unique_ptr<SessionBundle> bundle)
        : bundle_(std::move(bundle)) {
        signature_status_ = tensorflow::serving::GetGenericSignature(
			"generic",
            bundle_->meta_graph_def, 
			&signature_
		);
    }

    Status Classify() {
        // Transform protobuf input to inference input tensor and create
        // output tensor placeholder.
        // See partition_export.py for details.
        Tensor q_input(tensorflow::DT_FLOAT, {3, 5, 300});
		Tensor a_input(tensorflow::DT_FLOAT, {3, 7, 300});
        std::vector<Tensor> outputs;

        // Run inference.
        if (!signature_status_.ok()) {
            return signature_status_;
        }
        // WARNING(break-tutorial-inline-code): The following code snippet is
        // in-lined in tutorials, please update tutorial documents accordingly
        // whenever code changes.
        const tensorflow::Status status = bundle_->session->Run(
        {
            {signature_.map().at("input_q").tensor_name(), q_input},
            {signature_.map().at("input_a").tensor_name(), a_input}
        },
        {
            signature_.map().at("output_y").tensor_name()
        },
        {}, &outputs
        );
        if (!status.ok()) {
            return status;
        }

        // Transform inference output tensor to protobuf output.
        // See mnist_export.py for details.
        if (outputs.size() != 1) {
            std::cerr << "WRONG OUPUT SIZE" << std::endl;
            return tensorflow::errors::Internal("WRONG OUPUT SIZE");
        }
        const Tensor& score_tensor = outputs[0];
        const auto score_flat = outputs[0].flat<float>();
		std::cout << score_flat << std::endl;
		/*
        const TensorShape expected_shape({3, 1});
        if (!score_tensor.shape().IsSameSize(expected_shape)) {
            std::cerr << "WRONG OUTPUT SHAPE" << std::endl;
            return tensorflow::errors::Internal("WRONG OUTPUT SHAPE");;
        }
        const auto score_flat = outputs[0].flat<float>();
		std::cout << score_flat << std::endl;
		*/
        return Status::OK();
    }

private:
    std::unique_ptr<SessionBundle> bundle_;
    tensorflow::Status signature_status_;
    GenericSignature signature_;
};

int main(int argc, char** argv) {
    if (argc != 2) {
        LOG(FATAL) << "Usage: mnist_inference /path/to/export";
    }
    const string bundle_path(argv[1]);

    tensorflow::port::InitMain(argv[0], &argc, &argv);

    // WARNING(break-tutorial-inline-code): The following code snippet is
    // in-lined in tutorials, please update tutorial documents accordingly
    // whenever code changes.

    SessionBundleConfig session_bundle_config;

    //////
    // Request batching, keeping default values for the tuning parameters.
    //
    // (If you prefer to disable batching, simply omit the following lines of code
    // such that session_bundle_config.batching_parameters remains unset.)
    BatchingParameters* batching_parameters =
        session_bundle_config.mutable_batching_parameters();
    batching_parameters->mutable_thread_pool_name()->set_value(
        "service_batch_threads");
    // Use a very large queue, to avoid rejecting requests. (Note: a production
    // server with load balancing may want to use the default, much smaller,
    // value.)
    batching_parameters->mutable_max_enqueued_batches()->set_value(1000);
    //////

    std::unique_ptr<SessionBundleFactory> bundle_factory;
    TF_QCHECK_OK(
        SessionBundleFactory::Create(session_bundle_config, &bundle_factory));
    std::unique_ptr<SessionBundle> bundle(new SessionBundle);
    TF_QCHECK_OK(bundle_factory->CreateSessionBundle(bundle_path, &bundle));

    // END WARNING(break-tutorial-inline-code)
    ServiceImpl service(std::move(bundle));
    auto s = service.Classify();
	std::cerr << s.error_message() << std::endl;
    return 0;
}
