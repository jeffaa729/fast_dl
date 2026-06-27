#include <dl/autograd/GradMode.hpp>

namespace dl::autograd {

namespace {

thread_local bool grad_enabled = true;

}  // namespace

bool is_grad_enabled() {
    return grad_enabled;
}

void set_grad_enabled(bool enabled) {
    grad_enabled = enabled;
}

NoGradGuard::NoGradGuard()
    : previous_(is_grad_enabled()) {
    set_grad_enabled(false);
}

NoGradGuard::~NoGradGuard() {
    set_grad_enabled(previous_);
}

}  // namespace dl::autograd
