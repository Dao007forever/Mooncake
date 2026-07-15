// Copyright 2024 KVCache.AI
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef WORKER_H
#define WORKER_H

#include <queue>
#include <unordered_set>

#include "config.h"
#include "rdma_context.h"
#include "transport/rdma_transport/context_health_tracker.h"

namespace mooncake {
class WorkerPool {
   public:
    WorkerPool(RdmaContext &context, int numa_socket_id = 0);

    ~WorkerPool();

    // Add slices to queue, called by Transport
    int submitPostSend(const std::vector<Transport::Slice *> &slice_list);

    void trackPostedSlices(const std::vector<Transport::Slice *> &slice_list,
                           size_t first, size_t count);
    void untrackPostedSlices(const std::vector<Transport::Slice *> &slice_list,
                             size_t first, size_t count);

   private:
    void performPostSend(int thread_id);

    void performPollCq(int thread_id);

    void redispatch(std::vector<Transport::Slice *> &slice_list, int thread_id);

    void transferWorker(int thread_id);

    bool hasOutstandingCq(int thread_id);

    void monitorWorker();

    int doProcessContextEvents();

    // Simplified rail monitor: pause problematic paths for a cooldown period
    struct RailState {
        int error_count = 0;
        uint64_t pause_until_ns = 0;  // Timestamp (ns) when pause expires
    };

    void markRailFailed(const std::string &peer_nic_path);
    bool isRailAvailable(const std::string &peer_nic_path);

    // Retry helper: increment retry count and return whether retry is allowed
    static bool shouldRetrySlice(Transport::Slice *slice);

    // Unified path failure handler: marks rail failed, notifies other workers,
    // and optionally deletes the endpoint
    void handlePathFailure(const std::string &peer_nic_path,
                           RdmaEndPoint *endpoint = nullptr);
    void refreshPublishedLocalTopology();
    GidRefreshResult refreshPublishedLocalGid();

    // Context-level circuit breaker for catastrophic local RNIC failure.
    // State lives in ContextHealthTracker. The breaker trips only when the
    // consecutive all-rails-failed streak spans at least
    // MC_CONTEXT_FAILURE_MIN_PEERS distinct peer servers (a streak against a
    // single dead peer must never deactivate the local context), and it
    // auto-reactivates (half-open) after MC_CONTEXT_PAUSE_TTL_MS from the
    // monitor tick. Fatal async events reset the tracker so the TTL can never
    // resurrect an event-deactivated context.
    //
    // Every (tripped, context-active) transition -- submitter-thread trips,
    // monitor-thread TTL reactivation, and the async-event handlers below --
    // updates context_.set_active() inside the tracker mutex, so the flag can
    // never diverge from the trip state. Without that, a trip's
    // set_active(false) interleaving with PORT_ACTIVE's set_active(true) +
    // reset() could leave the context inactive with no armed trip: TTL
    // recovery would never run and the context would be skipped forever.
    bool contextHealthy() const { return !health_tracker_.tripped(); }
    void markContextSuccess() { health_tracker_.recordSuccess(); }
    void markContextFailure(
        const std::unordered_set<std::string> &failed_peers);
    void maybeReactivateContext();
    // Fatal async event owns the inactive state: clear the breaker (cancels
    // any pending TTL reactivation) and deactivate, atomically.
    void onFatalEventDeactivate() {
        health_tracker_.reset([this] { context_.set_active(false); });
    }
    // PORT_ACTIVE recovery: clear the breaker (streak AND any armed trip)
    // and reactivate, atomically.
    void onPortActiveReactivate() {
        health_tracker_.reset([this] { context_.set_active(true); });
    }

   private:
    RdmaContext &context_;
    const int numa_socket_id_;

    std::vector<std::thread> worker_thread_;
    std::atomic<bool> workers_running_;

    std::atomic<int> parked_worker_count_;

    // The poll worker updates these on every poll pass. The monitor worker
    // reads them when CQ entries stay outstanding, so a transfer timeout can
    // be distinguished from a stalled poller.
    std::atomic<uint64_t> last_poll_ts_ns_{0};
    std::atomic<uint64_t> last_poll_interval_ns_{0};
    std::atomic<uint64_t> max_poll_interval_ns_{0};

    std::mutex posted_slices_mutex_;
    std::unordered_set<Transport::Slice *> posted_slices_;

    std::atomic<int> redispatch_counter_;

    std::mutex cond_mutex_;
    std::condition_variable cond_var_;

    using SliceList = std::vector<Transport::Slice *>;

    const static int kShardCount = 8;
    std::unordered_map<std::string, SliceList> slice_queue_[kShardCount];
    std::atomic<uint64_t> slice_queue_count_[kShardCount];
    TicketLock slice_queue_lock_[kShardCount];

    std::vector<std::unordered_map<std::string, SliceList>>
        collective_slice_queue_;

    std::atomic<uint64_t> submitted_slice_count_, processed_slice_count_;

    // Rail state management: peer_nic_path -> RailState
    std::unordered_map<std::string, RailState> rail_states_;
    std::mutex rail_state_lock_;

    // Rail monitor configuration
    const static int kRailErrorThreshold = 5;            // Errors before pause
    const static uint64_t kRailPauseNs = 1000000000ull;  // 1 second pause

    // Context-level circuit breaker (see markContextFailure above)
    ContextHealthTracker health_tracker_;
    const static int kContextFailureThreshold =
        32;  // consecutive all-rails-failed
};
}  // namespace mooncake

#endif  // WORKER_H
