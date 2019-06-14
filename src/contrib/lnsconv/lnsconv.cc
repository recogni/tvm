#include <dmlc/logging.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/util.h>
#include <mutex>

namespace tvm {
namespace contrib {
using namespace runtime;

TVM_REGISTER_GLOBAL("tvm.contrib.lnsconv.conv3x3").set_body([](TVMArgs args, TVMRetValue* ret) {
  // Maybe check for one-time initialization issues
  static std::once_flag flag;

  // inputs (3x3)
  DLTensor* a = args[0];
  DLTensor* b = args[1];

  CHECK_EQ(a->ndim, 2);
  CHECK_EQ(b->ndim, 2);
  CHECK_EQ(a->shape[0], 3);
  CHECK_EQ(a->shape[1], 3);
  CHECK_EQ(b->shape[0], 3);
  CHECK_EQ(b->shape[1], 3);

  // Should these be float or uint
  CHECK(TypeMatch(a->dtype, kDLFloat, 32));
  CHECK(TypeMatch(b->dtype, kDLFloat, 32));

  // results (single value)
  DLTensor* z = args[2];
  CHECK_EQ(z->ndim, 1);
  CHECK_EQ(z->shape[0], 1);

  CHECK(TypeMatch(z->dtype, kDLFloat, 32));
});
}  // namespace contrib

}  // namespace tvm
