add_library(model_qwen3 qwen3op.cc)
target_link_libraries(model_qwen3 layer kernel util model_gpt "${TORCH_LIBRARIES}")
set_property(TARGET model_qwen3 PROPERTY POSITION_INDEPENDENT_CODE ON)