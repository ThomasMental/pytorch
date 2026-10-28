#if USE_DISTRIBUTED && defined(USE_C10D_GLOO)

#include <c10/util/Registry.h>
#include <torch/csrc/distributed/c10d/gloo/ProcessGroupGlooWork.hpp>

#include <utility>

namespace c10d {

TORCH_DECLARE_TYPED_REGISTRY(
    GlooAllreduceRegistry,
    c10::DeviceType,
    ProcessGroupGloo::AsyncWork,
    c10::intrusive_ptr,
    std::shared_ptr<gloo::Context>,
    std::vector<at::Tensor>&,
    ReduceOp,
    uint32_t,
    uint64_t,
    std::chrono::milliseconds);

namespace {

class OpenRegAllreduceWork : public AsyncAllreduceWork {
 public:
  OpenRegAllreduceWork(
      std::shared_ptr<gloo::Context> context,
      std::vector<at::Tensor>& tensors,
      ReduceOp reduceOp,
      uint32_t tag,
      uint64_t seq,
      std::chrono::milliseconds timeout)
      : AsyncAllreduceWork(
            std::move(context), tensors, reduceOp, tag, seq, timeout) {}

  void run() override {}

  void synchronize() override {}
};

c10::intrusive_ptr<ProcessGroupGloo::AsyncWork> createOpenRegAllreduceWork(
    std::shared_ptr<gloo::Context> context,
    std::vector<at::Tensor>& tensors,
    ReduceOp reduceOp,
    uint32_t tag,
    uint64_t seq,
    std::chrono::milliseconds timeout) {
  return c10::make_intrusive<OpenRegAllreduceWork>(
      std::move(context), tensors, reduceOp, tag, seq, timeout);
}

C10_REGISTER_TYPED_CREATOR(
    GlooAllreduceRegistry,
    at::kPrivateUse1,
    createOpenRegAllreduceWork)

} // namespace
} // namespace c10d

#endif // USE_DISTRIBUTED && defined(USE_C10D_GLOO)
