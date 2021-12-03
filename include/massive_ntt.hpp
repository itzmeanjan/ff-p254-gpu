#pragma once
#include <ntt.hpp>
#include <utils.hpp>

void massive_ntt(sycl::queue &q, const uint64_t dim, const uint64_t wg_size);
