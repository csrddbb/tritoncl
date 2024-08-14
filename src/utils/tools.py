import triton
def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"