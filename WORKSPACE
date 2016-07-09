local_repository(
  name = "org_tensorflow",
  path = __workspace_dir__ + "/tensorflow",
)

local_repository(
  name = "tf_serving",
  path = __workspace_dir__ + "/serving",
)

load('//knowledgeflow_serving:tensorflow_local_workspace.bzl', 'tf_workspace')
tf_workspace("tensorflow/", "@org_tensorflow")
