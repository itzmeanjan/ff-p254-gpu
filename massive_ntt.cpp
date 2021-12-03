#include <massive_ntt.hpp>

int64_t massive_ntt(sycl::queue &q, const uint64_t round_size,
                    const uint64_t dim, const uint64_t wg_size) {
  uint64_t log_2_dim = (uint64_t)sycl::log2((float)dim);
  uint64_t n1 = 1 << (log_2_dim / 2);
  uint64_t n2 = dim / n1;
  uint64_t n = sycl::max(n1, n2);

  ff_p254_t *vec_h = static_cast<ff_p254_t *>(
      sycl::malloc_host(sizeof(ff_p254_t) * dim * round_size, q));
  // I just populate input of single round with random elements
  // All rounds compute on same input, after they copy same input
  // data from host to device
  // But output doesn't end up modifying same memory locations
  // for all rounds
  prepare_random_vector(vec_h, dim);

  tp start = std::chrono::system_clock::now();

  ff_p254_t *mem = static_cast<ff_p254_t *>(
      sycl::malloc_device(sizeof(ff_p254_t) * (n1 * n2 + n * n + 3) * 4, q));

  sycl::event evt_0 =
      q.memcpy(mem + (n1 * n2 + n * n + 3) * 0, vec_h, sizeof(ff_p254_t) * dim);
  sycl::event evt_1 =
      q.memcpy(mem + (n1 * n2 + n * n + 3) * 1, vec_h, sizeof(ff_p254_t) * dim);

  sycl::event evt_2 =
      six_step_fft(q, mem + (n1 * n2 + n * n + 3) * 0,
                   mem + (n1 * n2 + n * n + 3) * 0 + n1 * n2,
                   mem + (n1 * n2 + n * n + 3) * 0 + n1 * n2 + n * n + 0,
                   mem + (n1 * n2 + n * n + 3) * 0 + n1 * n2 + n * n + 1,
                   mem + (n1 * n2 + n * n + 3) * 0 + n1 * n2 + n * n + 2, dim,
                   wg_size, {evt_0});

  sycl::event evt_3 =
      six_step_fft(q, mem + (n1 * n2 + n * n + 3) * 1,
                   mem + (n1 * n2 + n * n + 3) * 1 + n1 * n2,
                   mem + (n1 * n2 + n * n + 3) * 1 + n1 * n2 + n * n + 0,
                   mem + (n1 * n2 + n * n + 3) * 1 + n1 * n2 + n * n + 1,
                   mem + (n1 * n2 + n * n + 3) * 1 + n1 * n2 + n * n + 2, dim,
                   wg_size, {evt_1});

  sycl::event evt_4 =
      q.memcpy(mem + (n1 * n2 + n * n + 3) * 2, vec_h, sizeof(ff_p254_t) * dim);
  sycl::event evt_5 =
      q.memcpy(mem + (n1 * n2 + n * n + 3) * 3, vec_h, sizeof(ff_p254_t) * dim);

  sycl::event evt_6 =
      six_step_fft(q, mem + (n1 * n2 + n * n + 3) * 2,
                   mem + (n1 * n2 + n * n + 3) * 2 + n1 * n2,
                   mem + (n1 * n2 + n * n + 3) * 2 + n1 * n2 + n * n + 0,
                   mem + (n1 * n2 + n * n + 3) * 2 + n1 * n2 + n * n + 1,
                   mem + (n1 * n2 + n * n + 3) * 2 + n1 * n2 + n * n + 2, dim,
                   wg_size, {evt_4});

  sycl::event evt_7 =
      six_step_fft(q, mem + (n1 * n2 + n * n + 3) * 3,
                   mem + (n1 * n2 + n * n + 3) * 3 + n1 * n2,
                   mem + (n1 * n2 + n * n + 3) * 3 + n1 * n2 + n * n + 0,
                   mem + (n1 * n2 + n * n + 3) * 3 + n1 * n2 + n * n + 1,
                   mem + (n1 * n2 + n * n + 3) * 3 + n1 * n2 + n * n + 2, dim,
                   wg_size, {evt_5});

  sycl::event evt_8 = q.submit([&](sycl::handler &h) {
    h.depends_on(evt_2);
    h.memcpy(vec_h + dim * 0, mem + (n1 * n2 + n * n + 3) * 0,
             sizeof(ff_p254_t) * dim);
  });

  sycl::event evt_9 = q.submit([&](sycl::handler &h) {
    h.depends_on(evt_3);
    h.memcpy(vec_h + dim * 1, mem + (n1 * n2 + n * n + 3) * 1,
             sizeof(ff_p254_t) * dim);
  });

  sycl::event evt_10 = q.submit([&](sycl::handler &h) {
    h.depends_on(evt_6);
    h.memcpy(vec_h + dim * 2, mem + (n1 * n2 + n * n + 3) * 2,
             sizeof(ff_p254_t) * dim);
  });

  sycl::event evt_11 = q.submit([&](sycl::handler &h) {
    h.depends_on(evt_7);
    h.memcpy(vec_h + dim * 3, mem + (n1 * n2 + n * n + 3) * 3,
             sizeof(ff_p254_t) * dim);
  });

  q.submit_barrier({evt_8, evt_9, evt_10, evt_11}).wait();

  tp end = std::chrono::system_clock::now();

  sycl::free(vec_h, q);
  sycl::free(mem, q);

  return std::chrono::duration_cast<std::chrono::microseconds>(end - start)
      .count();
}
