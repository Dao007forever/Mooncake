#include "master_service.h"

#include <algorithm>
#include <atomic>
#include <barrier>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <thread>
#include <vector>

#include <gflags/gflags.h>
#include <glog/logging.h>

#include "random.h"

DEFINE_uint64(num_objects, 10000000, "Number of unique random keys to create");
DEFINE_uint64(prefill_threads, 8, "Concurrent PutStart/PutEnd threads");
DEFINE_uint64(lookup_threads, 8, "Concurrent BatchExistKey threads");
DEFINE_uint64(batch_size, 200, "Random keys per existence request");
DEFINE_uint64(baseline_sec, 10, "Measured lookup-only baseline duration");
DEFINE_uint64(duration_sec, 60, "Measured duration with periodic eviction");
DEFINE_uint64(eviction_interval_sec, 10, "Seconds between eviction starts");
DEFINE_double(eviction_ratio, 0.05, "Target fraction of objects to evict");
DEFINE_double(eviction_lowerbound, 0.025, "Fallback eviction fraction");
DEFINE_uint64(lease_ms, 1000,
              "Read lease granted by successful existence calls");
DEFINE_uint64(seed, 1, "Reproducible key permutation and lookup RNG seed");

namespace {
using Clock = std::chrono::steady_clock;
using Microseconds = std::chrono::microseconds;

// SplitMix64's invertible permutation guarantees unique, randomized key IDs.
// The compact base64 representation stays inside std::string's small buffer.
std::string Key(uint64_t index) {
    uint64_t value = index + FLAGS_seed + 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    value ^= value >> 31;
    constexpr char alphabet[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
    std::string key(12, 'k');
    for (size_t i = 1; i < key.size(); ++i) {
        key[i] = alphabet[value & 63];
        value >>= 6;
    }
    return key;
}

uint64_t Us(Clock::duration duration) {
    return std::chrono::duration_cast<Microseconds>(duration).count();
}

struct Sample {
    uint64_t begin_us;
    uint64_t end_us;
    uint64_t hits;
    uint64_t errors;
};

struct alignas(64) ReaderStats {
    std::vector<Sample> samples;
};

struct Eviction {
    uint64_t begin_us;
    uint64_t end_us;
};

struct Summary {
    std::vector<uint64_t> latencies;
    uint64_t hits{0};
    uint64_t errors{0};

    void Add(const Sample& sample) {
        latencies.push_back(sample.end_us - sample.begin_us);
        hits += sample.hits;
        errors += sample.errors;
    }

    void Print(const char* phase, double seconds) {
        std::sort(latencies.begin(), latencies.end());
        auto percentile = [&](double fraction) -> uint64_t {
            if (latencies.empty()) return 0;
            const size_t rank =
                static_cast<size_t>(std::ceil(fraction * latencies.size()));
            return latencies[rank - 1];
        };
        const uint64_t keys = latencies.size() * FLAGS_batch_size;
        std::cout << "RESULT phase=" << phase
                  << " requests=" << latencies.size()
                  << " seconds=" << std::fixed << std::setprecision(3)
                  << seconds << " batches_per_sec="
                  << (seconds > 0 ? latencies.size() / seconds : 0)
                  << " p50_us=" << percentile(0.50)
                  << " p95_us=" << percentile(0.95)
                  << " p99_us=" << percentile(0.99)
                  << " max_us=" << percentile(1.0)
                  << " hit_percent=" << (keys > 0 ? 100.0 * hits / keys : 0)
                  << " errors=" << errors << std::endl;
    }
};

bool Prefill(mooncake::MasterService& service,
             const mooncake::UUID& client_id) {
    std::atomic<uint64_t> completed{0};
    std::atomic<bool> failed{false};
    std::vector<std::thread> writers;
    const auto begin = Clock::now();
    for (uint64_t thread = 0; thread < FLAGS_prefill_threads; ++thread) {
        writers.emplace_back([&, thread] {
            mooncake::ReplicateConfig config;
            config.replica_num = 1;
            config.preferred_segment = "exists_evict_bench_segment";
            for (uint64_t index = thread;
                 index < FLAGS_num_objects && !failed.load();
                 index += FLAGS_prefill_threads) {
                const auto key = Key(index);
                auto start = service.PutStart(client_id, key,
                                              mooncake::TenantId::Default(),
                                              1024, config);
                if (!start || start->size() != 1) {
                    LOG(ERROR) << "PutStart failed at index=" << index;
                    failed.store(true);
                    return;
                }
                auto end = service.PutEnd(client_id, key,
                                          mooncake::TenantId::Default(),
                                          mooncake::ReplicaType::MEMORY);
                if (!end) {
                    LOG(ERROR) << "PutEnd failed at index=" << index;
                    failed.store(true);
                    return;
                }
                const uint64_t count = completed.fetch_add(1) + 1;
                if (count % 1000000 == 0) {
                    std::cout
                        << "PREFILL completed=" << count
                        << " elapsed_sec=" << Us(Clock::now() - begin) / 1e6
                        << std::endl;
                }
            }
        });
    }
    for (auto& writer : writers) writer.join();
    std::cout << "PREFILL completed=" << completed.load()
              << " live_keys=" << service.GetKeyCount()
              << " elapsed_sec=" << Us(Clock::now() - begin) / 1e6 << std::endl;
    return !failed.load() && service.GetKeyCount() == FLAGS_num_objects;
}

bool Run() {
    auto config =
        mooncake::MasterServiceConfig::builder()
            .set_memory_allocator(mooncake::BufferAllocatorType::OFFSET)
            .set_eviction_ratio(0.0)
            .set_eviction_high_watermark_ratio(1.0)
            .set_default_kv_lease_ttl(FLAGS_lease_ms)
            .set_client_live_ttl_sec(86400)
            .set_enable_ha(false)
            .set_enable_oplog(false)
            .build();
    mooncake::MasterService service(config);
    const auto client_id = mooncake::generate_uuid();
    mooncake::Segment segment;
    segment.id = mooncake::generate_uuid();
    segment.name = "exists_evict_bench_segment";
    segment.te_endpoint = segment.name;
    segment.base = 0x300000000ULL;
    segment.size = FLAGS_num_objects * 1024 + FLAGS_num_objects * 128 + 1048576;
    // Like batch_evict_bench, the segment is a synthetic address range. Real
    // metadata/allocator handles are created; payload bytes and RPC are absent.
    if (!service.MountSegment(segment, client_id)) return false;
    if (!Prefill(service, client_id)) return false;

    std::cout << "CONFIG mode=in_process_metadata_no_payload_io"
              << " objects=" << FLAGS_num_objects
              << " threads=" << FLAGS_lookup_threads
              << " batch_size=" << FLAGS_batch_size
              << " baseline_sec=" << FLAGS_baseline_sec
              << " duration_sec=" << FLAGS_duration_sec
              << " eviction_interval_sec=" << FLAGS_eviction_interval_sec
              << " eviction_ratio=" << FLAGS_eviction_ratio
              << " eviction_lowerbound=" << FLAGS_eviction_lowerbound
              << " lease_ms=" << FLAGS_lease_ms << " seed=" << FLAGS_seed
              << std::endl;

    std::atomic<bool> stop{false};
    std::vector<ReaderStats> stats(FLAGS_lookup_threads);
    std::vector<std::thread> readers;
    std::barrier gate(static_cast<std::ptrdiff_t>(FLAGS_lookup_threads + 1));
    Clock::time_point origin;
    for (uint64_t thread = 0; thread < FLAGS_lookup_threads; ++thread) {
        readers.emplace_back([&, thread] {
            mooncake::RandomEngine rng(FLAGS_seed + thread + 1);
            std::vector<std::string> keys(FLAGS_batch_size);
            auto& samples = stats[thread].samples;
            samples.reserve(65536);
            gate.arrive_and_wait();
            while (!stop.load(std::memory_order_relaxed)) {
                for (auto& key : keys) {
                    key = Key(mooncake::randomIndex(FLAGS_num_objects, rng));
                }
                const auto begin = Clock::now();
                auto results =
                    service.BatchExistKey(keys, mooncake::TenantId::Default());
                const auto end = Clock::now();
                uint64_t hits = 0;
                uint64_t errors = 0;
                if (results.size() != keys.size()) {
                    errors = keys.size();
                } else {
                    for (const auto& result : results) {
                        if (!result) {
                            ++errors;
                        } else {
                            hits += result.value();
                        }
                    }
                }
                samples.push_back(
                    {Us(begin - origin), Us(end - origin), hits, errors});
            }
        });
    }
    origin = Clock::now();
    gate.arrive_and_wait();
    const auto baseline_end = origin + std::chrono::seconds(FLAGS_baseline_sec);
    const auto deadline =
        baseline_end + std::chrono::seconds(FLAGS_duration_sec);
    const auto interval = std::chrono::seconds(FLAGS_eviction_interval_sec);
    auto next = baseline_end + interval;
    std::vector<Eviction> evictions;
    while (next < deadline) {
        std::this_thread::sleep_until(next);
        if (Clock::now() >= deadline) break;
        const auto before = service.GetKeyCount();
        const auto begin = Clock::now();
        service.RunBatchEvictForTesting(FLAGS_eviction_ratio,
                                        FLAGS_eviction_lowerbound);
        const auto end = Clock::now();
        evictions.push_back({Us(begin - origin), Us(end - origin)});
        std::cout << "EVICTION cycle=" << evictions.size()
                  << " begin_sec=" << Us(begin - origin) / 1e6
                  << " duration_ms=" << Us(end - begin) / 1000.0
                  << " keys_before=" << before
                  << " keys_after=" << service.GetKeyCount() << std::endl;
        // Skip missed ticks instead of issuing overlapping/catch-up evictions.
        do {
            next += interval;
        } while (next <= Clock::now());
    }
    std::this_thread::sleep_until(deadline);
    stop.store(true, std::memory_order_relaxed);
    for (auto& reader : readers) reader.join();
    const double measured_sec = Us(Clock::now() - baseline_end) / 1e6;

    Summary baseline, all, during, between;
    uint64_t eviction_us = 0;
    for (const auto& eviction : evictions) {
        eviction_us += eviction.end_us - eviction.begin_us;
    }
    for (const auto& reader : stats) {
        for (const auto& sample : reader.samples) {
            if (sample.begin_us < FLAGS_baseline_sec * 1000000) {
                baseline.Add(sample);
                continue;
            }
            all.Add(sample);
            const bool overlaps = std::any_of(
                evictions.begin(), evictions.end(), [&](const Eviction& e) {
                    return sample.begin_us < e.end_us &&
                           sample.end_us > e.begin_us;
                });
            (overlaps ? during : between).Add(sample);
        }
    }
    std::cout << "FINAL live_keys=" << service.GetKeyCount()
              << " eviction_cycles=" << evictions.size() << std::endl;
    baseline.Print("baseline", FLAGS_baseline_sec);
    all.Print("periodic_eviction", measured_sec);
    during.Print("overlapping_eviction", eviction_us / 1e6);
    between.Print("between_evictions", measured_sec - eviction_us / 1e6);
    return baseline.errors == 0 && all.errors == 0 &&
           !baseline.latencies.empty() && !during.latencies.empty() &&
           !evictions.empty() && service.GetKeyCount() < FLAGS_num_objects;
}
}  // namespace

int main(int argc, char** argv) {
    google::InitGoogleLogging("ExistsEvictBench");
    FLAGS_logtostderr = true;
    FLAGS_minloglevel = 1;
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    if (FLAGS_num_objects == 0 || FLAGS_prefill_threads == 0 ||
        FLAGS_lookup_threads == 0 || FLAGS_batch_size == 0 ||
        FLAGS_baseline_sec == 0 || FLAGS_eviction_interval_sec == 0 ||
        FLAGS_duration_sec <= FLAGS_eviction_interval_sec ||
        FLAGS_num_objects >
            std::numeric_limits<uint64_t>::max() / 1152 - 1024 ||
        !(FLAGS_eviction_lowerbound > 0 &&
          FLAGS_eviction_lowerbound <= FLAGS_eviction_ratio &&
          FLAGS_eviction_ratio <= 1)) {
        LOG(ERROR) << "Invalid benchmark dimensions, durations, or ratios";
        return 1;
    }
    const bool ok = Run();
    google::ShutdownGoogleLogging();
    return ok ? 0 : 1;
}
