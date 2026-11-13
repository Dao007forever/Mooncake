// ---------------------------------------------------------------------------
// correctness_repro_tests.rs — reproductions for route/allocator safety gaps
// ---------------------------------------------------------------------------
//
// Each `repro_*` test asserts the invariant the store should provide and
// FAILS on the current code. Each `control_*` test pins behaviour that already
// holds, so the repros can be read against a passing baseline. The whole module
// is `#[ignore]`d so it adds no load or failures to the default suite. Run it
// with:
//
//   cargo test -p mooncake-store-client --lib correctness_repro -- --include-ignored
//
// Cases:
//   1. Eviction frees a replica immediately after the route CAS; an in-flight
//      reader holding the old route reads whatever the next writer put there.
//   2. Delete/overwrite reclaim waits only `reclaim_grace_ms`; a reader slower
//      than the grace window has the same problem.
//   3. The storage owner treats a miss from the bounded (top-k) route lookup as
//      "route gone" and frees a live allocation while the route still exists
//      on a lower-ranked authority.
//   4. Under authority failover, two writers both replace the same route
//      version; both release its replicas, and the second release frees an
//      unrelated key's live allocation.
//   5. When a new authority joins and the primary then blips, a writer that
//      fails over finds no route and publishes a different route at the same
//      version; readers return the older write.
//   6. `batch_put_from` from a registered buffer to remote-only replicas
//      publishes a route without a checksum, so cases 1-2 return wrong bytes
//      silently instead of failing.
//   7. A writer that dies between reserve and publish leaves a remote
//      allocation that nothing ever frees until a drain (the drain recovery
//      is pinned by a control test).
//   8. During a join, a writer whose membership view predates the join CASes
//      a superseded version on the old authorities; the two authority sets
//      hold different routes at the same version.
//   9. An authority whose lease lapsed rejoins with the same epoch and its old
//      route table, and is primary again for keys overwritten meanwhile.
//  10. Looking up a missing key reads every authority, not route_topk.
//  11. The placement label bonus saturates, so ties go to the lowest runtime.
//  12. (control) A drain recovers the allocation abandoned in case 7.
//  13. A deferred release whose span was reused at a different length fails
//      the releasing writer's next, unrelated put.

use std::collections::BTreeMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread::sleep;
use std::time::{Duration, Instant};

use mooncake_store_core::{
    CasResult, ClientLifecycleState, ClientRuntimeId, ClientStableId, MetadataBackend, ObjectKey,
    ObjectRoute, RouteVersion,
};
use mooncake_store_route::RouteAuthorityService;
use mooncake_store_rs_metadata::InMemoryMetadataBackend;
use mooncake_store_test_utils::transport::TestTransport;

use crate::{
    LocalMemoryConfig, MooncakeCompatibilityFacade, ObjectRef, PlacementPlanner, PutFromRequest,
    ReplicationPolicy, RouteControlMode, StoreClient, StoreClientBuilder,
};

use super::{
    fast_live_client_sync_interval, resolved_object_for_route, rw_only_config,
    storage_config_with_bytes, test_future_expiry_ms, wait_for_membership_convergence,
    wait_for_storage_clock_route,
};

// ===========================================================================
// Helpers
// ===========================================================================

const POOL: &str = "repro-pool";

/// Common builder: active client in one pool with explicit storage/route roles.
fn node<M: MetadataBackend + 'static>(
    metadata: &Arc<M>,
    transport: Arc<TestTransport>,
    stable_id: &str,
    storage: bool,
    route: bool,
) -> StoreClientBuilder {
    let metadata: Arc<dyn MetadataBackend> = metadata.clone();
    StoreClientBuilder::new(metadata, stable_id)
        .state(ClientLifecycleState::Active)
        .label("pool", POOL)
        .label("storage", if storage { "true" } else { "false" })
        .label("route", if route { "true" } else { "false" })
        .live_client_sync_interval(fast_live_client_sync_interval())
        .transport(transport)
}

fn build(builder: StoreClientBuilder) -> StoreClient {
    let client = builder
        .build(test_future_expiry_ms())
        .expect("client build should succeed");
    client
        .register_local_memory()
        .expect("client memory should register");
    client
}

fn writer_planner<M: MetadataBackend + 'static>(metadata: &Arc<M>) -> PlacementPlanner {
    let metadata: Arc<dyn MetadataBackend> = metadata.clone();
    PlacementPlanner::new(metadata).require_label("storage", "true")
}

fn remote_only() -> ReplicationPolicy {
    ReplicationPolicy::new()
        .replica_count(1)
        .prefer_local(false)
}

/// Writes `payload` through the zero-copy registered-buffer path.
fn put_registered(writer: &StoreClient, key: &str, payload: &[u8]) -> ObjectRoute {
    writer
        .register_buffer(payload.as_ptr() as *mut c_void, payload.len())
        .expect("source buffer should register");
    let mut routes = writer
        .batch_put_from(&[
            PutFromRequest::new(key, payload.as_ptr().cast(), payload.len())
                .replication(remote_only()),
        ])
        .expect("registered batch_put_from should succeed");
    assert_eq!(routes.len(), 1);
    routes.remove(0)
}

fn put_bytes(writer: &StoreClient, key: &str, payload: &[u8]) -> ObjectRoute {
    writer
        .put_with_policy(key, payload, &remote_only())
        .expect("put should succeed")
}

/// Issues the data-plane read for `route` the way `get` does after it has
/// resolved the route. Holding the route across other operations models a
/// reader whose RDMA read is still in flight.
fn read_with_route(
    reader: &StoreClient,
    key: &str,
    route: ObjectRoute,
) -> mooncake_store_core::Result<Vec<u8>> {
    let length = route.replicas[0].length as usize;
    let mut resolved = resolved_object_for_route(reader, "default", key, route);
    let mut buffer = vec![0u8; length];
    let mut checked_runtimes = std::collections::BTreeSet::new();
    let local_segments = reader.local_storage_segments();
    let deadline = reader.request_deadline_for_transfer(length as u64, 1);
    let transport = reader.transport().expect("reader transport should exist");
    reader.read_single_object_with_failover(
        transport,
        &mut resolved,
        &mut buffer,
        &mut checked_runtimes,
        &local_segments,
        deadline,
    )?;
    Ok(buffer)
}

/// Marks `runtime` unreachable in the namespace-wide suspect cache, the same
/// state a transport failure to that authority produces.
fn mark_unreachable(client: &StoreClient, target: &StoreClient, for_duration: Duration) {
    client.suspect_runtime_cache.lock().mark(
        target.runtime_id().clone(),
        Instant::now() + for_duration,
        None,
    );
}

fn scoped(key: &str) -> ObjectKey {
    ObjectKey::new(format!("default::{key}"))
}

/// Single storage node (`capacity` bytes) plus a compute-only writer and
/// reader, with routes kept in metadata so the route layer is not a factor.
struct SingleStoreCluster {
    store: StoreClient,
    writer: StoreClient,
    reader: StoreClient,
}

fn single_store_cluster(
    tag: &str,
    capacity: usize,
    writer_memory: LocalMemoryConfig,
) -> SingleStoreCluster {
    let metadata = Arc::new(InMemoryMetadataBackend::new());
    let store_transport = Arc::new(TestTransport::new(&format!("{tag}-store-segment")));
    let writer_transport = Arc::new(store_transport.peer(&format!("{tag}-writer-segment")));
    let reader_transport = Arc::new(store_transport.peer(&format!("{tag}-reader-segment")));
    let store = build(
        node(
            &metadata,
            store_transport,
            &format!("{tag}-store"),
            true,
            false,
        )
        .route_control(RouteControlMode::MetadataOnly)
        .local_memory(storage_config_with_bytes(capacity)),
    );
    let writer = build(
        node(
            &metadata,
            writer_transport,
            &format!("{tag}-writer"),
            false,
            false,
        )
        .route_control(RouteControlMode::MetadataOnly)
        .local_memory(writer_memory)
        .routed_writes(writer_planner(&metadata), 1),
    );
    let reader = build(
        node(
            &metadata,
            reader_transport,
            &format!("{tag}-reader"),
            false,
            false,
        )
        .route_control(RouteControlMode::MetadataOnly)
        .local_memory(rw_only_config()),
    );
    wait_for_membership_convergence(&[&store, &writer, &reader]);
    SingleStoreCluster {
        store,
        writer,
        reader,
    }
}

/// Embedded-WRH cluster: one storage node, two route authorities where
/// `primary` outranks `mirror` for every key, and compute-only clients.
struct AuthorityCluster {
    metadata: Arc<InMemoryMetadataBackend>,
    /// Kept alive so its storage segment stays registered.
    _store: StoreClient,
    primary: StoreClient,
    mirror: StoreClient,
    writer_a: StoreClient,
    writer_b: StoreClient,
    reader: StoreClient,
}

fn authority_cluster(
    tag: &str,
    capacity: usize,
    writer_b_memory: LocalMemoryConfig,
) -> AuthorityCluster {
    let metadata = Arc::new(InMemoryMetadataBackend::new());
    let base = Arc::new(TestTransport::new(&format!("{tag}-store-segment")));
    let peer = |name: &str| Arc::new(base.peer(&format!("{tag}-{name}-segment")));
    let store = build(
        node(
            &metadata,
            base.clone(),
            &format!("{tag}-store"),
            true,
            false,
        )
        .local_memory(storage_config_with_bytes(capacity)),
    );
    let primary = build(
        node(
            &metadata,
            peer("primary"),
            &format!("{tag}-primary"),
            false,
            true,
        )
        .label("route_weight", "1000000")
        .local_memory(rw_only_config()),
    );
    let mirror = build(
        node(
            &metadata,
            peer("mirror"),
            &format!("{tag}-mirror"),
            false,
            true,
        )
        .label("route_weight", "1")
        .local_memory(rw_only_config()),
    );
    let writer = |name: &str, memory: LocalMemoryConfig| {
        build(
            node(
                &metadata,
                peer(name),
                &format!("{tag}-{name}"),
                false,
                false,
            )
            .local_memory(memory)
            .routed_writes(writer_planner(&metadata), 1),
        )
    };
    let writer_a = writer("writer-a", rw_only_config());
    let writer_b = writer("writer-b", writer_b_memory);
    let reader = build(
        node(
            &metadata,
            peer("reader"),
            &format!("{tag}-reader"),
            false,
            false,
        )
        .local_memory(rw_only_config()),
    );
    wait_for_membership_convergence(&[&store, &primary, &mirror, &writer_a, &writer_b, &reader]);
    AuthorityCluster {
        metadata,
        _store: store,
        primary,
        mirror,
        writer_a,
        writer_b,
        reader,
    }
}

impl AuthorityCluster {
    fn namespace(&self) -> String {
        self.metadata.route_namespace()
    }

    fn authority_route(&self, authority: &StoreClient, key: &str) -> Option<ObjectRoute> {
        crate::route_directory::authority_get(
            &self.namespace(),
            &authority.runtime_id().stable_id,
            &scoped(key),
        )
        .expect("authority route read should succeed")
    }

    /// Models the best-effort mirror publication being dropped (async queue
    /// full or the primary dying before the mirror is sent): the mirror's copy
    /// is put back to what it held before the write.
    fn lose_mirror_update(&self, key: &str, previous: Option<&ObjectRoute>) {
        self.writer_a.route_directory.flush_pending_route_mirrors();
        self.writer_b.route_directory.flush_pending_route_mirrors();
        crate::route_directory::authority_replace(
            &self.namespace(),
            &self.mirror.runtime_id().stable_id,
            &scoped(key),
            previous,
        )
        .expect("mirror rollback should succeed");
    }
}

// ===========================================================================
// Case 1: eviction frees a replica under an in-flight reader
// ===========================================================================

#[test]
#[ignore = "control for the correctness repro suite; run with --include-ignored"]
fn control_eviction_reuse_is_detected_when_route_has_checksum() {
    let cluster = single_store_cluster("repro-evict-ck", 32, rw_only_config());
    let k1 = [b'A'; 32];
    let k2 = [b'B'; 32];

    let route_k1 = put_bytes(&cluster.writer, "k1", &k1);
    assert!(route_k1.replicas[0].checksum.is_some());
    wait_for_storage_clock_route(&cluster.store, &route_k1, false);

    // Store is full: writing k2 evicts k1 and reuses its span.
    let route_k2 = put_bytes(&cluster.writer, "k2", &k2);
    assert_eq!(
        route_k2.replicas[0].segment_offset,
        route_k1.replicas[0].segment_offset
    );

    // A reader that resolved k1 before the eviction must not get k2's bytes.
    // With a checksum the stale read is rejected.
    match read_with_route(&cluster.reader, "k1", route_k1) {
        Ok(bytes) => assert_ne!(bytes, k2.to_vec(), "stale reader returned k2's bytes"),
        Err(_) => {}
    }
}

#[test]
#[ignore = "repro case 1: eviction releases a replica with no grace; an in-flight reader returns another key's bytes"]
fn repro_eviction_reuse_returns_other_keys_bytes_to_in_flight_reader() {
    let cluster = single_store_cluster("repro-evict", 32, rw_only_config());
    let k1 = [b'A'; 32];
    let k2 = [b'B'; 32];

    // Zero-copy path used by KV-cache writers: no checksum on the route.
    let route_k1 = put_registered(&cluster.writer, "k1", &k1);
    wait_for_storage_clock_route(&cluster.store, &route_k1, false);

    // Reader resolved k1's route; its RDMA read has not landed yet.
    let stale_route = route_k1.clone();

    // Store is full: k2's reservation evicts k1 (route CAS, then immediate
    // release) and first-fit hands the same span to k2.
    let route_k2 = put_registered(&cluster.writer, "k2", &k2);
    assert_eq!(
        route_k2.replicas[0].segment_offset, route_k1.replicas[0].segment_offset,
        "precondition: k2 reuses k1's freed span"
    );

    // The reader's transfer lands now.
    let result = read_with_route(&cluster.reader, "k1", stale_route);
    if let Ok(bytes) = result {
        assert_ne!(
            bytes,
            k2.to_vec(),
            "read of k1 returned k2's bytes: eviction freed the span under an in-flight reader"
        );
    }
}

// ===========================================================================
// Case 2: delete reclaim waits only reclaim_grace_ms
// ===========================================================================

#[test]
#[ignore = "repro case 2: a reader slower than reclaim_grace_ms returns another key's bytes after delete + reuse"]
fn repro_delete_reclaim_grace_does_not_cover_slow_reader() {
    let writer_memory = rw_only_config().reclaim_grace_ms(50);
    let cluster = single_store_cluster("repro-grace", 32, writer_memory);
    let k1 = [b'A'; 32];
    let k3 = [b'C'; 32];

    let route_k1 = put_registered(&cluster.writer, "k1", &k1);
    let stale_route = route_k1.clone();

    cluster
        .writer
        .remove("k1", false)
        .expect("remove should succeed");
    // Reader is slower than the 50 ms grace window (large value, stall, GC).
    sleep(Duration::from_millis(120));

    // The writer's next put flushes due reclaims and first-fit reuses the span.
    let route_k3 = put_registered(&cluster.writer, "k3", &k3);
    assert_eq!(
        route_k3.replicas[0].segment_offset, route_k1.replicas[0].segment_offset,
        "precondition: k3 reuses k1's reclaimed span"
    );

    let result = read_with_route(&cluster.reader, "k1", stale_route);
    if let Ok(bytes) = result {
        assert_ne!(
            bytes,
            k3.to_vec(),
            "read of deleted k1 returned k3's bytes after the grace window expired"
        );
    }
}

// ===========================================================================
// Case 3: a bounded (top-k) route miss frees a live allocation
// ===========================================================================

#[test]
#[ignore = "repro case 3: storage owner frees a live replica when only the top-k authorities miss the route"]
fn repro_bounded_route_miss_frees_live_allocation_after_authorities_join() {
    let tag = "repro-join";
    let metadata = Arc::new(InMemoryMetadataBackend::new());
    let base = Arc::new(TestTransport::new(&format!("{tag}-store-segment")));
    let peer = |name: &str| Arc::new(base.peer(&format!("{tag}-{name}-segment")));

    let store = build(
        node(
            &metadata,
            base.clone(),
            &format!("{tag}-store"),
            true,
            false,
        )
        .local_memory(storage_config_with_bytes(32)),
    );
    let old_authority = build(
        node(
            &metadata,
            peer("old-auth"),
            &format!("{tag}-old-auth"),
            false,
            true,
        )
        .local_memory(rw_only_config()),
    );
    let writer = build(
        node(
            &metadata,
            peer("writer"),
            &format!("{tag}-writer"),
            false,
            false,
        )
        .local_memory(rw_only_config())
        .routed_writes(writer_planner(&metadata), 1),
    );
    let reader = build(
        node(
            &metadata,
            peer("reader"),
            &format!("{tag}-reader"),
            false,
            false,
        )
        .local_memory(rw_only_config()),
    );
    wait_for_membership_convergence(&[&store, &old_authority, &writer, &reader]);

    let k1 = [b'A'; 32];
    let route_k1 = put_bytes(&writer, "k1", &k1);
    wait_for_storage_clock_route(&store, &route_k1, false);

    // Scale-out: two route-capable nodes join and outrank the old authority
    // for every key (route_topk = 2). Their route tables start empty.
    let joined_a = build(
        node(
            &metadata,
            peer("new-auth-a"),
            &format!("{tag}-new-auth-a"),
            false,
            true,
        )
        .label("route_weight", "1000000")
        .local_memory(rw_only_config()),
    );
    let joined_b = build(
        node(
            &metadata,
            peer("new-auth-b"),
            &format!("{tag}-new-auth-b"),
            false,
            true,
        )
        .label("route_weight", "1000000")
        .local_memory(rw_only_config()),
    );
    wait_for_membership_convergence(&[
        &store,
        &old_authority,
        &writer,
        &reader,
        &joined_a,
        &joined_b,
    ]);

    // Memory pressure on the store: the stale/orphan pre-pass looks k1 up on
    // the new top-2 authorities only, gets "missing", and releases k1.
    // (A legitimate CLOCK eviction would instead remove k1's route.)
    let k2 = [b'B'; 32];
    let k2_result = writer.put_with_policy("k2", &k2, &remote_only());

    // Invariant: while k1's route is published, its bytes are k1's.
    let still_routed = reader
        .query_route("k1")
        .expect("route query should succeed");
    if let Some(route) = still_routed {
        assert_eq!(route.version, route_k1.version);
        if let Ok(route_k2) = &k2_result {
            assert_ne!(
                route_k2.replicas[0].segment_offset, route_k1.replicas[0].segment_offset,
                "store gave live object k1's span to k2 while k1's route is still published"
            );
        }
        assert_eq!(
            reader
                .get("k1")
                .expect("k1's route is still published, so k1 must be readable"),
            k1.to_vec()
        );
    }
}

// ===========================================================================
// Case 4: double release under authority failover frees another key's span
// ===========================================================================

#[test]
#[ignore = "repro case 4: two writers replace the same route version; the second release of its replicas frees an unrelated key"]
fn repro_failover_double_release_frees_unrelated_live_allocation() {
    let cluster = authority_cluster("repro-dbl", 64, rw_only_config());
    let v1 = [b'1'; 16];
    let k9 = [b'9'; 16];
    let k10 = [b'X'; 16];

    // k1 v1 at offset 0; mirrored to the secondary authority.
    let route_v1 = put_bytes(&cluster.writer_a, "k1", &v1);
    cluster
        .writer_a
        .route_directory
        .flush_pending_route_mirrors();
    assert_eq!(
        cluster.authority_route(&cluster.mirror, "k1"),
        Some(route_v1.clone())
    );

    // Writer A: v1 -> v2 at the primary. v1's span is released (grace 0).
    let _route_v2a = put_bytes(&cluster.writer_a, "k1", &[b'2'; 16]);
    // The mirror update for v2 is lost; the secondary still holds v1.
    cluster.lose_mirror_update("k1", Some(&route_v1));

    // Unrelated key k9 takes v1's freed span.
    let route_k9 = put_bytes(&cluster.writer_a, "k9", &k9);
    cluster
        .writer_a
        .route_directory
        .flush_pending_route_mirrors();
    assert_eq!(
        route_k9.replicas[0].segment_offset, route_v1.replicas[0].segment_offset,
        "precondition: k9 reuses v1's span"
    );

    // Writer B can't reach the primary and fails over to the secondary, which
    // still has v1. Its CAS v1 -> v2' applies, and it schedules v1's release.
    mark_unreachable(&cluster.writer_b, &cluster.primary, Duration::from_secs(10));
    let _route_v2b = put_bytes(&cluster.writer_b, "k1", &[b'3'; 16]);

    // That second release freed k9's allocation; the next write lands on it.
    let route_k10 = put_bytes(&cluster.writer_a, "k10", &k10);
    assert_ne!(
        route_k10.replicas[0].segment_offset, route_k9.replicas[0].segment_offset,
        "k10 was placed on live object k9's span"
    );
    assert_eq!(
        cluster.reader.get("k9").expect("k9 should be readable"),
        k9.to_vec()
    );
}

// ===========================================================================
// Case 5: same-version divergence after authority churn + failover
// ===========================================================================

#[test]
#[ignore = "repro case 5: failover to an authority that never saw the route publishes a second route at the same version; readers see the older write"]
fn repro_failover_to_empty_authority_publishes_same_version_and_loses_newer_write() {
    let tag = "repro-split";
    let metadata = Arc::new(InMemoryMetadataBackend::new());
    let base = Arc::new(TestTransport::new(&format!("{tag}-store-segment")));
    let peer = |name: &str| Arc::new(base.peer(&format!("{tag}-{name}-segment")));
    let store = build(
        node(
            &metadata,
            base.clone(),
            &format!("{tag}-store"),
            true,
            false,
        )
        .local_memory(storage_config_with_bytes(64)),
    );
    let primary = build(
        node(
            &metadata,
            peer("primary"),
            &format!("{tag}-primary"),
            false,
            true,
        )
        .label("route_weight", "1000000")
        .local_memory(rw_only_config()),
    );
    let writer = |name: &str| {
        build(
            node(
                &metadata,
                peer(name),
                &format!("{tag}-{name}"),
                false,
                false,
            )
            .local_memory(rw_only_config())
            .routed_writes(writer_planner(&metadata), 1),
        )
    };
    let writer_a = writer("writer-a");
    let writer_b = writer("writer-b");
    let reader = build(
        node(
            &metadata,
            peer("reader"),
            &format!("{tag}-reader"),
            false,
            false,
        )
        .local_memory(rw_only_config()),
    );
    wait_for_membership_convergence(&[&store, &primary, &writer_a, &writer_b, &reader]);

    // k1 is written while the primary is the only authority.
    let first = [b'A'; 16];
    let route_first = put_bytes(&writer_a, "k1", &first);

    // A second route-capable node joins; its table starts empty and nothing
    // has read k1 since, so read repair has not backfilled it.
    let joined = build(
        node(
            &metadata,
            peer("joined"),
            &format!("{tag}-joined"),
            false,
            true,
        )
        .label("route_weight", "1")
        .local_memory(rw_only_config()),
    );
    wait_for_membership_convergence(&[&store, &primary, &joined, &writer_a, &writer_b, &reader]);

    // The primary blips for writer B. B's load finds no route on any
    // reachable authority, so it publishes k1 as a fresh object.
    mark_unreachable(&writer_b, &primary, Duration::from_millis(500));
    let second = [b'B'; 16];
    let route_second = put_bytes(&writer_b, "k1", &second);
    sleep(Duration::from_millis(700)); // primary reachable again

    let namespace = metadata.route_namespace();
    let read_authority = |authority: &StoreClient| {
        crate::route_directory::authority_get(
            &namespace,
            &authority.runtime_id().stable_id,
            &scoped("k1"),
        )
        .expect("authority route read should succeed")
    };
    let on_primary = read_authority(&primary).expect("primary keeps the first write");
    let on_joined = read_authority(&joined).expect("joined authority holds the second write");
    assert_eq!(on_primary.replicas, route_first.replicas);
    assert_eq!(on_joined.replicas, route_second.replicas);
    assert_ne!(
        on_primary.version, on_joined.version,
        "two authorities hold different routes for k1 at the same version"
    );
    assert_eq!(
        reader.get("k1").expect("k1 should be readable"),
        second.to_vec(),
        "reader returned the older write after writer B's put succeeded"
    );
}

// ===========================================================================
// Case 6: registered remote-only writes publish routes without a checksum
// ===========================================================================

#[test]
#[ignore = "repro case 6: batch_put_from to remote-only replicas publishes no checksum, disabling read validation"]
fn repro_registered_remote_put_route_has_no_checksum() {
    let cluster = single_store_cluster("repro-cksum", 64, rw_only_config());
    let route = put_registered(&cluster.writer, "k1", &[b'A'; 32]);
    assert!(
        route
            .replicas
            .iter()
            .all(|replica| replica.checksum.is_some()),
        "registered remote-only write published replicas without a checksum: {:?}",
        route.replicas
    );
}

// ===========================================================================
// Case 7: a writer that dies between reserve and publish leaks the span
// ===========================================================================

#[test]
#[ignore = "repro case 7: a reservation abandoned before route publish is never freed, so the store stays full"]
fn repro_abandoned_reservation_is_never_reclaimed() {
    let tag = "repro-orphan";
    let metadata = Arc::new(InMemoryMetadataBackend::new());
    let store_transport = Arc::new(TestTransport::new(&format!("{tag}-store-segment")));
    let crashed_transport = Arc::new(store_transport.peer(&format!("{tag}-crashed-segment")));
    let writer_transport = Arc::new(store_transport.peer(&format!("{tag}-writer-segment")));
    let store = build(
        node(
            &metadata,
            store_transport,
            &format!("{tag}-store"),
            true,
            false,
        )
        .route_control(RouteControlMode::MetadataOnly)
        .request_timeout(Duration::from_millis(20))
        .local_memory(storage_config_with_bytes(32)),
    );
    let crashed = build(
        node(
            &metadata,
            crashed_transport,
            &format!("{tag}-crashed"),
            false,
            false,
        )
        .route_control(RouteControlMode::MetadataOnly)
        .local_memory(rw_only_config())
        .routed_writes(writer_planner(&metadata), 1),
    );
    let writer = build(
        node(
            &metadata,
            writer_transport,
            &format!("{tag}-writer"),
            false,
            false,
        )
        .route_control(RouteControlMode::MetadataOnly)
        .local_memory(rw_only_config())
        .routed_writes(writer_planner(&metadata), 1),
    );
    wait_for_membership_convergence(&[&store, &crashed, &writer]);

    // The crashed writer reserved the whole store, then died before writing
    // the bytes or publishing a route.
    let (_target, reservation) = crashed
        .reserve_storage_runtime_segment(store.runtime_id(), 32, false)
        .expect("reservation should succeed");
    assert_eq!(reservation.length_bytes, 32);
    drop(crashed);

    // Well past the pending-publish deadline (20 ms + 250 ms slack).
    sleep(Duration::from_millis(600));

    // No object exists anywhere, so a 32-byte put must fit.
    let result = writer.put_with_policy("k1", &[b'A'; 32], &remote_only());
    assert!(
        result.is_ok(),
        "store is still full of an abandoned reservation: {:?}",
        result.err()
    );
}

// ===========================================================================
// Case 8: a writer with a pre-join membership view overwrites a superseded
// version (no fencing across authority-set changes)
// ===========================================================================

#[test]
#[ignore = "repro case 8: during a join, a writer whose membership view predates the join CASes a superseded version on the old authorities"]
fn repro_join_view_divergence_lets_stale_writer_overwrite_superseded_version() {
    let tag = "repro-view";
    // No background refresh: membership changes only when the test says so.
    let slow = Duration::from_secs(3600);
    let metadata = Arc::new(InMemoryMetadataBackend::new());
    let base = Arc::new(TestTransport::new(&format!("{tag}-store-segment")));
    let peer = |name: &str| Arc::new(base.peer(&format!("{tag}-{name}-segment")));
    let authority = |name: &str, weight: &str| {
        build(
            node(&metadata, peer(name), &format!("{tag}-{name}"), false, true)
                .live_client_sync_interval(slow)
                .label("route_weight", weight)
                .local_memory(rw_only_config()),
        )
    };
    let writer = |name: &str| {
        build(
            node(
                &metadata,
                peer(name),
                &format!("{tag}-{name}"),
                false,
                false,
            )
            .live_client_sync_interval(slow)
            .local_memory(rw_only_config())
            .routed_writes(writer_planner(&metadata), 1),
        )
    };
    let store = build(
        node(
            &metadata,
            base.clone(),
            &format!("{tag}-store"),
            true,
            false,
        )
        .live_client_sync_interval(slow)
        .local_memory(storage_config_with_bytes(64)),
    );
    let old_a = authority("old-a", "1");
    let old_b = authority("old-b", "1");
    let writer_a = writer("writer-a");
    let writer_b = writer("writer-b");
    let reader = build(
        node(
            &metadata,
            peer("reader"),
            &format!("{tag}-reader"),
            false,
            false,
        )
        .live_client_sync_interval(slow)
        .local_memory(rw_only_config()),
    );
    wait_for_membership_convergence(&[&store, &old_a, &old_b, &writer_a, &writer_b, &reader]);

    let v1 = put_bytes(&writer_a, "k1", &[b'1'; 16]);
    writer_a.route_directory.flush_pending_route_mirrors();
    let pre_join_view = writer_b
        .live_client_cache
        .lock()
        .snapshot()
        .expect("membership snapshot should exist");

    // Scale-out: two authorities join and become the top-2 for every key.
    let new_a = authority("new-a", "1000000");
    let new_b = authority("new-b", "1000000");
    wait_for_membership_convergence(&[
        &store, &old_a, &old_b, &writer_a, &writer_b, &reader, &new_a, &new_b,
    ]);

    // Writer A (current view) supersedes v1 on the new authorities.
    let a_v2 = put_bytes(&writer_a, "k1", &[b'A'; 16]);
    writer_a.route_directory.flush_pending_route_mirrors();
    assert!(a_v2.version > v1.version);

    // Writer B runs in another process whose membership cache has not refreshed
    // since before the join (refresh interval is 1 s in production). Every
    // client in this test shares one per-process cache, so the pre-join view is
    // put back for the duration of B's write.
    writer_b.live_client_cache.lock().store(pre_join_view);
    let b_result = writer_b.put_with_policy("k1", &[b'B'; 16], &remote_only());
    writer_b.route_directory.flush_pending_route_mirrors();
    super::super::refresh_live_client_cache(
        metadata.as_ref(),
        &writer_b.live_client_cache,
        "repro_restore_view",
    )
    .expect("membership refresh should succeed");

    let namespace = metadata.route_namespace();
    let on = |client: &StoreClient| {
        crate::route_directory::authority_get(
            &namespace,
            &client.runtime_id().stable_id,
            &scoped("k1"),
        )
        .expect("authority route read should succeed")
    };
    let on_new = on(&new_a).expect("new authority holds k1");
    let on_old = on(&old_a).expect("old authority holds k1");
    assert!(
        on_new.version != on_old.version || on_new.replicas == on_old.replicas,
        "new and old authorities hold different routes for k1 at version {:?} \
         (writer B's put returned {:?})",
        on_new.version,
        b_result.as_ref().map(|route| route.version)
    );
    if b_result.is_ok() {
        assert_eq!(
            reader.get("k1").expect("k1 should be readable"),
            vec![b'B'; 16],
            "writer B's put succeeded last, but readers see writer A's value"
        );
    }
}

// ===========================================================================
// Case 9: an authority that rejoins after its lease lapsed serves its old table
// ===========================================================================

#[test]
#[ignore = "repro case 9: an authority whose lease lapsed rejoins with the same epoch and its stale table, and readers get the superseded route"]
fn repro_authority_rejoining_after_lease_lapse_serves_stale_route() {
    let tag = "repro-rejoin";
    let metadata = Arc::new(InMemoryMetadataBackend::new());
    let base = Arc::new(TestTransport::new(&format!("{tag}-store-segment")));
    let peer = |name: &str| Arc::new(base.peer(&format!("{tag}-{name}-segment")));
    let store = build(
        node(
            &metadata,
            base.clone(),
            &format!("{tag}-store"),
            true,
            false,
        )
        .local_memory(storage_config_with_bytes(64)),
    );
    let primary = build(
        node(
            &metadata,
            peer("primary"),
            &format!("{tag}-primary"),
            false,
            true,
        )
        .label("route_weight", "1000000")
        .local_memory(rw_only_config()),
    );
    let mirror = build(
        node(
            &metadata,
            peer("mirror"),
            &format!("{tag}-mirror"),
            false,
            true,
        )
        .local_memory(rw_only_config()),
    );
    let writer = build(
        node(
            &metadata,
            peer("writer"),
            &format!("{tag}-writer"),
            false,
            false,
        )
        .local_memory(rw_only_config())
        .routed_writes(writer_planner(&metadata), 1),
    );
    let reader = build(
        node(
            &metadata,
            peer("reader"),
            &format!("{tag}-reader"),
            false,
            false,
        )
        .local_memory(rw_only_config()),
    );
    wait_for_membership_convergence(&[&store, &primary, &mirror, &writer, &reader]);

    let v1 = put_bytes(&writer, "k1", &[b'A'; 16]);
    writer.route_directory.flush_pending_route_mirrors();

    // The primary's heartbeats stop reaching metadata; its lease runs out
    // while the process (and its route table) stays up.
    let mut lease = metadata
        .list_live_clients()
        .expect("lease list should succeed")
        .into_iter()
        .find(|lease| lease.runtime == *primary.runtime_id())
        .expect("primary lease should exist");
    lease.expires_at_ms = super::now_ms() + 100;
    metadata
        .upsert_client_lease(&lease)
        .expect("lease update should succeed");
    sleep(Duration::from_millis(300));

    // Meanwhile k1 is overwritten through the remaining authority.
    let v2 = put_bytes(&writer, "k1", &[b'B'; 16]);
    writer.route_directory.flush_pending_route_mirrors();
    assert!(v2.version > v1.version);

    // Heartbeats recover: the same runtime (same epoch) refreshes its lease.
    lease.expires_at_ms = super::now_ms() + 30_000;
    metadata
        .upsert_client_lease(&lease)
        .expect("lease refresh should succeed");
    sleep(Duration::from_millis(200));

    let held_by_primary = crate::route_directory::authority_get(
        &metadata.route_namespace(),
        &primary.runtime_id().stable_id,
        &scoped("k1"),
    )
    .expect("authority route read should succeed")
    .expect("primary still holds a route for k1");
    assert_eq!(
        held_by_primary.version, v2.version,
        "rejoined primary still serves superseded version {:?}",
        held_by_primary.version
    );
    // In production a reader stops at the primary once it answers, so it would
    // return v1 here. This in-process harness registers every authority as
    // local, which makes the read path also consult and repair the mirror, so
    // the stale primary is asserted on directly above.
    assert_eq!(
        reader.get("k1").expect("k1 should be readable"),
        vec![b'B'; 16],
        "reader returned the superseded value"
    );
}

// ===========================================================================
// Case 10: a route miss queries every authority, not just the top-k
// ===========================================================================

/// Authority service that counts per-key route reads and otherwise behaves
/// like the client's local authority adapter.
struct CountingAuthority {
    reads: AtomicUsize,
}

impl RouteAuthorityService for CountingAuthority {
    fn get_route(
        &self,
        namespace: &str,
        authority: &ClientStableId,
        key: &ObjectKey,
    ) -> mooncake_store_core::Result<Option<ObjectRoute>> {
        self.reads.fetch_add(1, Ordering::Relaxed);
        crate::route_directory::authority_get(namespace, authority, key)
    }

    fn list_routes_by_replica_owner(
        &self,
        namespace: &str,
        authority: &ClientStableId,
        owner: &ClientRuntimeId,
    ) -> mooncake_store_core::Result<Vec<ObjectRoute>> {
        crate::route_directory::authority_list_routes_by_replica_owner(namespace, authority, owner)
    }

    fn compare_and_swap_route(
        &self,
        namespace: &str,
        authority: &ClientStableId,
        key: &ObjectKey,
        expected: Option<RouteVersion>,
        next: Option<&ObjectRoute>,
    ) -> mooncake_store_core::Result<CasResult> {
        crate::route_directory::authority_compare_and_swap(
            namespace, authority, key, expected, next,
        )
    }

    fn replace_route(
        &self,
        namespace: &str,
        authority: &ClientStableId,
        key: &ObjectKey,
        next: Option<&ObjectRoute>,
    ) -> mooncake_store_core::Result<()> {
        crate::route_directory::authority_replace(namespace, authority, key, next)
    }
}

#[test]
#[ignore = "repro case 10: looking up a missing key reads every route authority in turn instead of route_topk"]
fn repro_route_miss_reads_every_authority() {
    const AUTHORITIES: usize = 8;
    const ROUTE_TOPK: usize = 2;
    let tag = "repro-scan";
    let metadata = Arc::new(InMemoryMetadataBackend::new());
    let base = Arc::new(TestTransport::new(&format!("{tag}-store-segment")));
    let peer = |name: &str| Arc::new(base.peer(&format!("{tag}-{name}-segment")));
    let store = build(
        node(
            &metadata,
            base.clone(),
            &format!("{tag}-store"),
            true,
            false,
        )
        .local_memory(storage_config_with_bytes(64)),
    );
    let authorities = (0..AUTHORITIES)
        .map(|index| {
            build(
                node(
                    &metadata,
                    peer(&format!("auth-{index}")),
                    &format!("{tag}-auth-{index}"),
                    false,
                    true,
                )
                .local_memory(rw_only_config()),
            )
        })
        .collect::<Vec<_>>();
    let writer = build(
        node(
            &metadata,
            peer("writer"),
            &format!("{tag}-writer"),
            false,
            false,
        )
        .local_memory(rw_only_config())
        .routed_writes(writer_planner(&metadata), 1),
    );
    let mut all = vec![&store, &writer];
    all.extend(authorities.iter());
    wait_for_membership_convergence(&all);

    let namespace = metadata.route_namespace();
    let counters = authorities
        .iter()
        .map(|authority| {
            let counter = Arc::new(CountingAuthority {
                reads: AtomicUsize::new(0),
            });
            crate::route_directory::bind_local_authority_service(
                &namespace,
                &authority.runtime_id().stable_id,
                counter.clone(),
            );
            counter
        })
        .collect::<Vec<_>>();
    let total_reads = || -> usize {
        counters
            .iter()
            .map(|counter| counter.reads.swap(0, Ordering::Relaxed))
            .sum()
    };

    assert!(writer
        .query_route("missing-key")
        .expect("route query should succeed")
        .is_none());
    let query_reads = total_reads();

    put_bytes(&writer, "fresh-key", &[b'F'; 16]);
    let put_reads = total_reads();

    assert!(
        query_reads <= ROUTE_TOPK && put_reads <= ROUTE_TOPK,
        "a miss read {query_reads} (query_route) and {put_reads} (put of a new key) \
         authorities out of {AUTHORITIES}; expected at most route_topk = {ROUTE_TOPK}"
    );
}

// ===========================================================================
// Case 11: the placement label bonus saturates and skews placement
// ===========================================================================

fn placement_top_choice_shares(tag: &str, with_domain: bool) -> Vec<f64> {
    const NODES: usize = 8;
    const KEYS: usize = 4000;
    let metadata = Arc::new(InMemoryMetadataBackend::new());
    let transport = TestTransport::new(&format!("{tag}-observer-segment"));
    let nodes = (0..NODES)
        .map(|index| {
            super::publish_labeled_storage_node_with_capacity(
                &metadata,
                &transport,
                &format!("{tag}-node-{index}"),
                &format!("{tag}-node-{index}-segment"),
                POOL,
                64,
                1,
                Some("d1"),
                None,
                None,
            )
        })
        .collect::<Vec<_>>();
    let observer = build(
        node(
            &metadata,
            Arc::new(transport.peer(&format!("{tag}-observer-client-segment"))),
            &format!("{tag}-observer"),
            false,
            false,
        )
        .route_control(RouteControlMode::MetadataOnly)
        .local_memory(rw_only_config()),
    );
    let planner = writer_planner(&metadata);
    let mut wins = BTreeMap::<ClientRuntimeId, usize>::new();
    for index in 0..KEYS {
        let key = format!("key-{index}");
        let object = if with_domain {
            ObjectRef::new(&key).domain("d1")
        } else {
            ObjectRef::new(&key)
        };
        let ranked = planner
            .ranked_candidates_from_lease(&observer.lease(), "default", &object)
            .expect("ranking should succeed");
        assert_eq!(ranked.len(), NODES);
        *wins.entry(ranked[0].clone()).or_default() += 1;
    }
    nodes
        .iter()
        .map(|runtime| *wins.get(runtime).unwrap_or(&0) as f64 / KEYS as f64)
        .collect()
}

#[test]
#[ignore = "control for the correctness repro suite; run with --include-ignored"]
fn control_placement_without_labels_is_balanced() {
    let shares = placement_top_choice_shares("ctl-skew", false);
    let max = shares.iter().cloned().fold(0.0, f64::max);
    assert!(max < 1.5 / shares.len() as f64, "shares: {shares:?}");
}

#[test]
#[ignore = "repro case 11: saturating_add of the label bonus caps scores, so ties go to the lowest runtime id"]
fn repro_placement_label_bonus_skews_to_lowest_runtime() {
    let shares = placement_top_choice_shares("repro-skew", true);
    let max = shares.iter().cloned().fold(0.0, f64::max);
    assert!(
        max < 1.5 / shares.len() as f64,
        "with a matching domain label the top-choice shares are {shares:?} (fair share {:.3})",
        1.0 / shares.len() as f64
    );
}

// ===========================================================================
// Case 12: the abandoned reservation from case 7 is only recovered by a drain
// ===========================================================================

#[test]
#[ignore = "control for the correctness repro suite; run with --include-ignored"]
fn control_abandoned_reservation_is_recovered_by_drain() {
    let tag = "ctl-orphan-drain";
    let metadata = Arc::new(InMemoryMetadataBackend::new());
    let store_transport = Arc::new(TestTransport::new(&format!("{tag}-store-segment")));
    let crashed_transport = Arc::new(store_transport.peer(&format!("{tag}-crashed-segment")));
    let mut store = build(
        node(
            &metadata,
            store_transport,
            &format!("{tag}-store"),
            true,
            false,
        )
        .route_control(RouteControlMode::MetadataOnly)
        .request_timeout(Duration::from_millis(20))
        .local_memory(storage_config_with_bytes(32)),
    );
    let crashed = build(
        node(
            &metadata,
            crashed_transport,
            &format!("{tag}-crashed"),
            false,
            false,
        )
        .route_control(RouteControlMode::MetadataOnly)
        .local_memory(rw_only_config())
        .routed_writes(writer_planner(&metadata), 1),
    );
    wait_for_membership_convergence(&[&store, &crashed]);
    crashed
        .reserve_storage_runtime_segment(store.runtime_id(), 32, false)
        .expect("reservation should succeed");
    drop(crashed);
    sleep(Duration::from_millis(600));
    assert_eq!(
        store.allocator.lock().allocations().len(),
        1,
        "the abandoned reservation is still allocated before the drain"
    );

    store
        .evacuate_owned_replicas()
        .expect("drain should succeed");
    assert!(
        store.allocator.lock().allocations().is_empty(),
        "drain should release the abandoned reservation"
    );
}

// ===========================================================================
// Case 13: a stale release whose length doesn't match fails the next put
// ===========================================================================

#[test]
#[ignore = "repro case 13: a deferred release of a reused span with a different length fails the releasing writer's next put"]
fn repro_stale_release_length_mismatch_fails_next_put() {
    let cluster = authority_cluster("repro-len", 64, rw_only_config().reclaim_grace_ms(20));

    let route_v1 = put_bytes(&cluster.writer_a, "k1", &[b'1'; 16]);
    cluster
        .writer_a
        .route_directory
        .flush_pending_route_mirrors();
    put_bytes(&cluster.writer_a, "k1", &[b'2'; 16]);
    cluster.lose_mirror_update("k1", Some(&route_v1));

    // An 8-byte object takes the start of v1's freed 16-byte span.
    let route_k9 = put_bytes(&cluster.writer_a, "k9", &[b'9'; 8]);
    assert_eq!(
        route_k9.replicas[0].segment_offset,
        route_v1.replicas[0].segment_offset
    );

    // Writer B fails over, replaces v1 again, and defers v1's release by 20 ms.
    mark_unreachable(&cluster.writer_b, &cluster.primary, Duration::from_secs(10));
    put_bytes(&cluster.writer_b, "k1", &[b'3'; 16]);
    sleep(Duration::from_millis(60));

    // Writer B's next put flushes the due release: offset matches k9, length doesn't.
    let next = cluster
        .writer_b
        .put_with_policy("k11", &[b'Y'; 16], &remote_only());
    assert!(
        next.is_ok(),
        "unrelated put failed because of a stale deferred release: {:?}",
        next.err()
    );
}
