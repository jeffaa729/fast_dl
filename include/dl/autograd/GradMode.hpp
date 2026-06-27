#pragma once

namespace dl::autograd {

bool is_grad_enabled();
void set_grad_enabled(bool enabled);

class NoGradGuard {
public:
    NoGradGuard();
    ~NoGradGuard();

    NoGradGuard(const NoGradGuard&) = delete;
    NoGradGuard& operator=(const NoGradGuard&) = delete;

private:
    bool previous_;
};

}  // namespace dl::autograd
