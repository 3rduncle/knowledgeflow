local_repository(
  name = "org_tensorflow",
  path = __workspace_dir__ + "/tensorflow",
)

local_repository(
  name = "tf_serving",
  path = __workspace_dir__ + "/serving",
)

load('//tensorflow/tensorflow:workspace.bzl', 'tf_workspace')
tf_workspace("tensorflow/", "@org_tensorflow")

new_git_repository(
	name = "tinytoml",
	remote = "https://github.com/mayah/tinytoml.git",
	commit = "3559856002eee57693349b8a2d8a0cf6250d269c",
	build_file = "tinytoml.BUILD"
)

new_git_repository(
	name = "cppjieba",
	remote = "https://github.com/yanyiwu/cppjieba.git",
	commit = "ec84858",
	build_file = "cppjieba.BUILD",
)

new_git_repository(
	name = "rapidjson",
	remote = "https://github.com/miloyip/rapidjson.git",
	commit = "5e8a382",
	build_file = "rapidjson.BUILD",
)
