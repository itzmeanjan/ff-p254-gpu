#include <massive_ntt.hpp>

int64_t massive_ntt(sycl::queue &q, const uint64_t round_size,
                    const uint64_t dim, const uint64_t wg_size) {
  uint64_t log_2_dim = (uint64_t)sycl::log2((float)dim);
  uint64_t n1 = 1 << (log_2_dim / 2);
  uint64_t n2 = dim / n1;
  uint64_t n = sycl::max(n1, n2);

  ff_p254_t *vec_h = static_cast<ff_p254_t *>(
      sycl::malloc_host(sizeof(ff_p254_t) * dim * round_size, q));
  prepare_random_vector(vec_h, dim * round_size);

  std::vector<ff_p254_t *> mem;
  std::vector<sycl::event> evts;

  mem.reserve(round_size);
  evts.reserve(round_size);

  tp start = std::chrono::system_clock::now();

  for (uint64_t i = 0; i < round_size; i++) {
    mem[i] = static_cast<ff_p254_t *>(
        sycl::malloc_device(sizeof(ff_p254_t) * (n1 * n2 + n2 * n2 + 3), q));

    sycl::event evt_0_a =
        q.memcpy(mem[i], vec_h + dim * i, sizeof(ff_p254_t) * dim);
    sycl::event evt_0_b =
        six_step_fft(q, mem[i], mem[i] + dim, mem[i] + dim + n2 * n2 + 0,
                     mem[i] + dim + n2 * n2 + 1, mem[i] + dim + n2 * n2 + 2,
                     dim, wg_size, {evt_0_a});
    sycl::event evt_0_c =
        q.memcpy(vec_h + dim * i, mem[i], sizeof(ff_p254_t) * dim);

    evts[i] = evt_0_c;
  }

  q.submit_barrier(evts);

  tp end = std::chrono::system_clock::now();

  sycl::free(vec_h, q);
  for (uint64_t i = 0; i < round_size; i++) {
    sycl::free(mem[i], q);
  }

  return std::chrono::duration_cast<std::chrono::microseconds>(end - start)
      .count();
}
