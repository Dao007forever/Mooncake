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

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdlib>
#include <string>
#include <vector>

#include "config.h"
#include "error.h"
#include "memory_location.h"
#include "transport/rdma_transport/rdma_transport.h"

namespace mooncake {
namespace {

TEST(RdmaLocalNicMapTest, PreferredOnlyWithReorderedOpenedDevices) {
    TopologyMatrix matrix;
    matrix["cuda:0"] = {"cuda:0", {"nic0", "nic1"}, {"nic2", "nic3"}};
    matrix["cuda:1"] = {"cuda:1", {"nic2", "nic3"}, {"nic0", "nic1"}};
    auto map = RdmaTransport::buildLocalNicMap(
        matrix, {"nic3", "nic1", "nic0", "nic2"});
    EXPECT_EQ(map.at("cuda:0"), (std::vector<size_t>{2, 1}));
    EXPECT_EQ(map.at("cuda:1"), (std::vector<size_t>{3, 0}));
}

TEST(RdmaLocalNicMapTest, MissingAndDuplicatePreferredDevices) {
    TopologyMatrix matrix;
    matrix["cuda:0"] = {"cuda:0", {"nic0", "nic0", "gone"}, {"nic1"}};
    matrix["cuda:1"] = {"cuda:1", {"gone"}, {"nic1"}};
    matrix["cuda:2"] = {"cuda:2", {}, {"nic1"}};
    auto map = RdmaTransport::buildLocalNicMap(matrix, {"nic1", "nic0"});
    ASSERT_EQ(map.size(), 1u);
    EXPECT_EQ(map.at("cuda:0"), (std::vector<size_t>{1}));
    EXPECT_TRUE(RdmaTransport::buildLocalNicMap(matrix, {}).empty());
}

class RdmaNicSelectionEnvTest : public ::testing::Test {
   protected:
    void SetUp() override {
        const char *value = std::getenv("MC_RDMA_NIC_SELECTION");
        had_value_ = value != nullptr;
        if (value) saved_ = value;
        ::unsetenv("MC_RDMA_NIC_SELECTION");
    }
    void TearDown() override {
        if (had_value_)
            ::setenv("MC_RDMA_NIC_SELECTION", saved_.c_str(), 1);
        else
            ::unsetenv("MC_RDMA_NIC_SELECTION");
    }
    bool had_value_ = false;
    std::string saved_;
};

TEST_F(RdmaNicSelectionEnvTest, DefaultsToAll) {
    GlobalConfig config;
    loadGlobalConfig(config);
    EXPECT_EQ(config.rdma_nic_selection, RdmaNicSelection::ALL);
}

TEST_F(RdmaNicSelectionEnvTest, LocalAndExplicitAllAreCaseInsensitive) {
    GlobalConfig config;
    ASSERT_EQ(::setenv("MC_RDMA_NIC_SELECTION", "LoCaL", 1), 0);
    loadGlobalConfig(config);
    EXPECT_EQ(config.rdma_nic_selection, RdmaNicSelection::LOCAL);
    ASSERT_EQ(::setenv("MC_RDMA_NIC_SELECTION", "ALL", 1), 0);
    loadGlobalConfig(config);
    EXPECT_EQ(config.rdma_nic_selection, RdmaNicSelection::ALL);
}

TEST_F(RdmaNicSelectionEnvTest, InvalidValuesPreserveCurrentPolicy) {
    GlobalConfig config;
    config.rdma_nic_selection = RdmaNicSelection::LOCAL;
    for (const auto *value : {"", "unsupported", "1"}) {
        ASSERT_EQ(::setenv("MC_RDMA_NIC_SELECTION", value, 1), 0);
        loadGlobalConfig(config);
        EXPECT_EQ(config.rdma_nic_selection, RdmaNicSelection::LOCAL);
    }
}

class RdmaRegisteredNicTest : public ::testing::Test {
   protected:
    void SetUp() override {
        ASSERT_EQ(desc.topology.parse(R"({"cuda:0":[["nic0"],["nic1","nic2"]],
                   "cpu:0":[["nic0"],["nic1","nic2"]]})"),
                  0);
        for (const auto &name : desc.topology.getHcaList())
            desc.devices.push_back({name, 0, "", ""});
        RdmaTransport::BufferDesc buffer{};
        buffer.addr = 0x10000;
        buffer.length = 0x10000;
        buffer.name = "cuda:0";
        buffer.rkey.resize(desc.devices.size(), 0);
        buffer.lkey.resize(desc.devices.size(), 0);
        desc.buffers.push_back(buffer);
    }
    int index(const std::string &name) const {
        for (size_t i = 0; i < desc.devices.size(); ++i)
            if (desc.devices[i].name == name) return i;
        return -1;
    }
    void registerOn(const std::string &name, size_t buffer = 0) {
        int id = index(name);
        ASSERT_GE(id, 0);
        desc.buffers[buffer].rkey[id] = 123;
        desc.buffers[buffer].lkey[id] = 456;
    }
    RdmaTransport::SegmentDesc desc;
};

TEST_F(RdmaRegisteredNicTest, SparsePreferredSurvivesEveryRetry) {
    registerOn("nic0");
    for (int retry = 0; retry < 30; ++retry) {
        int buffer = -1, device = -1;
        ASSERT_EQ(RdmaTransport::selectDevice(&desc, 0x10000, 1024, buffer,
                                              device, retry),
                  0);
        EXPECT_EQ(device, index("nic0"));
    }
}

TEST_F(RdmaRegisteredNicTest, UnregisteredPreferredFallsBackToValidMr) {
    registerOn("nic2");
    for (int retry = 0; retry < 10; ++retry) {
        int buffer = -1, device = -1;
        ASSERT_EQ(RdmaTransport::selectDevice(&desc, 0x10000, 1024, buffer,
                                              device, retry),
                  0);
        EXPECT_EQ(device, index("nic2"));
    }
}

TEST_F(RdmaRegisteredNicTest, HintCannotForceUnregisteredNic) {
    registerOn("nic2");
    int buffer = -1, device = -1;
    ASSERT_EQ(RdmaTransport::selectDevice(&desc, 0x10000, 1024, "nic0", buffer,
                                          device),
              0);
    EXPECT_EQ(device, index("nic2"));
}

TEST_F(RdmaRegisteredNicTest, HcaAffinityCannotForceUnregisteredNic) {
    auto saved = globalConfig().nic_peer_affinity;
    globalConfig().nic_peer_affinity = {{"source", {"nic0"}}};
    // Affinity is resolved when the topology is parsed.
    auto topology_json = desc.topology.toString();
    ASSERT_EQ(desc.topology.parse(topology_json), 0);
    globalConfig().nic_peer_affinity = std::move(saved);
    registerOn("nic2");
    for (int retry = 0; retry < 10; ++retry) {
        int buffer = -1, device = -1;
        ASSERT_EQ(RdmaTransport::selectDeviceByLocalHca(
                      &desc, 0x10000, 1024, "source", buffer, device, retry),
                  0);
        EXPECT_EQ(device, index("nic2"));
    }
}

TEST_F(RdmaRegisteredNicTest, AllModeStillUsesRegisteredHint) {
    for (const auto &name : {"nic0", "nic1", "nic2"}) registerOn(name);
    int buffer = -1, device = -1;
    ASSERT_EQ(RdmaTransport::selectDevice(&desc, 0x10000, 1024, "nic1", buffer,
                                          device),
              0);
    EXPECT_EQ(device, index("nic1"));
}

TEST_F(RdmaRegisteredNicTest, SparseMrSurvivesTopologyCompactionAndRecovery) {
    const auto healthy = desc.devices.back().name;
    const auto first = desc.devices.front().name;
    const auto second = desc.devices[1].name;
    auto saved_affinity = globalConfig().nic_peer_affinity;
    globalConfig().nic_peer_affinity = {{"source", {healthy}}};
    const int ret = desc.topology.parse(desc.topology.toString());
    globalConfig().nic_peer_affinity = std::move(saved_affinity);
    ASSERT_EQ(ret, 0);
    const auto original_topology = desc.topology;
    registerOn(healthy);
    const auto original_keys = desc.buffers[0].rkey;

    auto expect_healthy = [&] {
        for (int retry = 0; retry < 8; ++retry) {
            int buffer = -1, device = -1;
            ASSERT_EQ(RdmaTransport::selectDevice(&desc, 0x10000, 1024, buffer,
                                                  device, retry),
                      0);
            EXPECT_EQ(device, index(healthy));
            for (const auto &hint : {healthy, first}) {
                ASSERT_EQ(
                    RdmaTransport::selectDevice(&desc, 0x10000, 1024, hint,
                                                buffer, device, retry),
                    0);
                EXPECT_EQ(device, index(healthy));
            }
            ASSERT_EQ(
                RdmaTransport::selectDeviceByLocalHca(
                    &desc, 0x10000, 1024, "source", buffer, device, retry),
                0);
            EXPECT_EQ(device, index(healthy));
        }
        EXPECT_EQ(desc.buffers[0].rkey, original_keys);
    };

    expect_healthy();
    for (int flap = 0; flap < 2; ++flap) {
        // Match refreshPublishedLocalTopology: keep devices/keys fixed and
        // remove inactive devices only from the published topology.
        ASSERT_EQ(desc.topology.disableDevice(first), 0);
        expect_healthy();
        ASSERT_EQ(desc.topology.disableDevice(second), 0);
        expect_healthy();
        desc.topology = original_topology;
        expect_healthy();
    }
}

TEST_F(RdmaRegisteredNicTest, AllModeHintKeepsDeviceIdentityAfterNicDown) {
    for (const auto &device : desc.devices) registerOn(device.name);
    const auto healthy = desc.devices.back().name;
    const auto down = desc.devices.front().name;
    ASSERT_EQ(desc.topology.disableDevice(down), 0);
    for (int retry = 0; retry < 8; ++retry) {
        int buffer = -1, device = -1;
        ASSERT_EQ(RdmaTransport::selectDevice(&desc, 0x10000, 1024, healthy,
                                              buffer, device, retry),
                  0);
        EXPECT_EQ(device, index(healthy));
        ASSERT_EQ(RdmaTransport::selectDevice(&desc, 0x10000, 1024, buffer,
                                              device, retry),
                  0);
        EXPECT_NE(device, index(down));
    }
}

TEST_F(RdmaRegisteredNicTest, DownNicKeysCannotBeUsedForAHealthyNic) {
    const auto down = desc.devices.front().name;
    registerOn(down);
    ASSERT_EQ(desc.topology.disableDevice(down), 0);
    for (int retry = 0; retry < 8; ++retry) {
        int buffer = -1, device = -1;
        EXPECT_EQ(RdmaTransport::selectDevice(&desc, 0x10000, 1024, buffer,
                                              device, retry),
                  ERR_ADDRESS_NOT_REGISTERED);
        EXPECT_EQ(RdmaTransport::selectDevice(&desc, 0x10000, 1024, down,
                                              buffer, device, retry),
                  ERR_ADDRESS_NOT_REGISTERED);
        EXPECT_EQ(RdmaTransport::selectDeviceByLocalHca(
                      &desc, 0x10000, 1024, "source", buffer, device, retry),
                  ERR_ADDRESS_NOT_REGISTERED);
    }

    // A 1:N registration can still use its surviving NIC, even though the
    // fixed key array retains a nonzero entry for the inactive NIC as well.
    const auto healthy = desc.devices.back().name;
    registerOn(healthy);
    for (int retry = 0; retry < 8; ++retry) {
        int buffer = -1, device = -1;
        ASSERT_EQ(RdmaTransport::selectDevice(&desc, 0x10000, 1024, down,
                                              buffer, device, retry),
                  0);
        EXPECT_EQ(device, index(healthy));
    }
}

TEST_F(RdmaRegisteredNicTest, ReparsedPeerTopologyPreservesDeviceIdentity) {
    const auto down = desc.devices.front().name;
    const auto healthy = desc.devices.back().name;
    registerOn(healthy);
    ASSERT_EQ(desc.topology.disableDevice(down), 0);
    // Wire metadata preserves devices/keys but reconstructs topology from its
    // surviving candidates, whose indices can differ from the fixed arrays.
    const auto serialized = desc.topology.toString();
    desc.topology.clear();
    ASSERT_EQ(desc.topology.parse(serialized), 0);
    for (int retry = 0; retry < 8; ++retry) {
        int buffer = -1, device = -1;
        ASSERT_EQ(RdmaTransport::selectDevice(&desc, 0x10000, 1024, buffer,
                                              device, retry),
                  0);
        EXPECT_EQ(device, index(healthy));
    }
}

TEST_F(RdmaRegisteredNicTest, LocalOnlyBufferUsesLkeyButCannotBeRemoteTarget) {
    registerOn("nic1");
    desc.buffers[0].rkey.clear();
    for (int retry = 0; retry < 8; ++retry) {
        int buffer = -1, device = -1;
        ASSERT_EQ(RdmaTransport::selectDevice(&desc, 0x10000, 1024, buffer,
                                              device, retry, -1, -1, false),
                  0);
        EXPECT_EQ(device, index("nic1"));
        EXPECT_EQ(RdmaTransport::selectDevice(&desc, 0x10000, 1024, buffer,
                                              device, retry),
                  ERR_ADDRESS_NOT_REGISTERED);
    }
}

TEST_F(RdmaRegisteredNicTest, CachedDeviceRejectsMissingKeysAndDisabledNic) {
    registerOn("nic0");
    registerOn("nic2");
    desc.rebuildBufferRangeIndex();
    desc.buffers[0].rkey[index("nic0")] = 0;
    int buffer = -1, device = -1;
    ASSERT_EQ(RdmaTransport::selectDevice(&desc, 0x10000, 1024, buffer, device,
                                          0, 0, index("nic0")),
              0);
    EXPECT_EQ(device, index("nic2"));
    desc.buffers[0].rkey[index("nic0")] = 123;
    ASSERT_EQ(desc.topology.disableDevice("nic0"), 0);
    ASSERT_EQ(RdmaTransport::selectDevice(&desc, 0x10000, 1024, buffer, device,
                                          0, 0, index("nic0")),
              0);
    EXPECT_EQ(device, index("nic2"));
    ASSERT_EQ(RdmaTransport::selectDeviceByLocalHca(&desc, 0x10000, 1024,
                                                    "source", buffer, device, 0,
                                                    0, index("nic0")),
              0);
    EXPECT_EQ(device, index("nic2"));
}

TEST_F(RdmaRegisteredNicTest, CachedLocalOnlyDeviceUsesLocalKey) {
    registerOn("nic1");
    desc.buffers[0].rkey.clear();
    desc.rebuildBufferRangeIndex();
    int buffer = -1, device = -1;
    ASSERT_EQ(RdmaTransport::selectDevice(&desc, 0x10000, 1024, buffer, device,
                                          0, 0, index("nic1"), false),
              0);
    EXPECT_EQ(device, index("nic1"));
    EXPECT_EQ(RdmaTransport::selectDevice(&desc, 0x10000, 1024, buffer, device,
                                          0, 0, index("nic1")),
              ERR_ADDRESS_NOT_REGISTERED);
}

TEST_F(RdmaRegisteredNicTest, RejectsMissingKeysAndMalformedIndices) {
    int buffer = -1, device = -1;
    EXPECT_EQ(RdmaTransport::selectDevice(&desc, 0x10000, 1024, buffer, device),
              ERR_ADDRESS_NOT_REGISTERED);
    EXPECT_FALSE(RdmaTransport::hasRegisteredKey(desc.buffers[0], -1));
    EXPECT_FALSE(RdmaTransport::hasRegisteredKey(desc.buffers[0], 999));
    desc.buffers[0].rkey.clear();
    EXPECT_EQ(RdmaTransport::selectDeviceByLocalHca(&desc, 0x10000, 1024,
                                                    "source", buffer, device),
              ERR_ADDRESS_NOT_REGISTERED);
    EXPECT_EQ(
        RdmaTransport::selectDevice(nullptr, 0x10000, 1024, buffer, device),
        ERR_ADDRESS_NOT_REGISTERED);
}

TEST_F(RdmaRegisteredNicTest,
       RejectsMissingTopologyWithoutSelectingAnEmptyRailSet) {
    registerOn("nic0");
    desc.topology.clear();
    int buffer = -1, device = -1;
    EXPECT_EQ(RdmaTransport::selectDevice(&desc, 0x10000, 1024, buffer, device),
              ERR_ADDRESS_NOT_REGISTERED);
}

TEST_F(RdmaRegisteredNicTest, ChunkBoundaryUsesEachChunksOwnKeys) {
    desc.buffers.push_back(desc.buffers[0]);
    desc.buffers[1].addr += desc.buffers[0].length;
    registerOn("nic0", 0);
    registerOn("nic2", 1);
    int buffer = -1, device = -1;
    EXPECT_EQ(RdmaTransport::selectDevice(&desc, 0x1fff0, 32, buffer, device),
              ERR_ADDRESS_NOT_REGISTERED);
    ASSERT_EQ(RdmaTransport::selectDevice(&desc, 0x20000, 32, buffer, device),
              0);
    EXPECT_EQ(buffer, 1);
    EXPECT_EQ(device, index("nic2"));
    EXPECT_EQ(
        RdmaTransport::selectDevice(&desc, UINT64_MAX - 15, 32, buffer, device),
        ERR_ADDRESS_NOT_REGISTERED);
}

TEST_F(RdmaRegisteredNicTest, HostSegmentsAndUnknownLocationUseValidKeys) {
    registerOn("nic2");
    for (const auto &name : {"segments:4096:0,1", "unknown-location"}) {
        desc.buffers[0].name = name;
        int buffer = -1, device = -1;
        ASSERT_EQ(
            RdmaTransport::selectDevice(&desc, 0x18000, 1024, buffer, device),
            0);
        EXPECT_EQ(device, index("nic2"));
    }
}

}  // namespace
}  // namespace mooncake
