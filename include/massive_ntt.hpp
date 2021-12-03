#pragma once
#include <ntt.hpp>
#include <utils.hpp>

int64_t massive_ntt(sycl::queue &q, const uint64_t round_size,
                    const uint64_t dim, const uint64_t wg_size);
