#include <gtest/gtest.h>

#include <optional>
#include <string>
#include <vector>

#include "replica.h"

namespace mooncake {
namespace {

// ReplicateConfig's wire layout immediately before host_id was added.
struct LegacyReplicateConfig {
    size_t replica_num{1};
    size_t nof_replica_num{0};
    bool with_soft_pin{false};
    bool with_hard_pin{false};
    std::vector<std::string> preferred_segments{};
    std::string preferred_segment{};
    std::vector<std::string> preferred_nof_segments{};
    bool prefer_alloc_in_same_node{false};
    ObjectDataType data_type{ObjectDataType::UNKNOWN};
    std::optional<std::vector<std::string>> group_ids{};
};

TEST(ReplicateConfigCompatibilityTest, DeserializesLegacyPayload) {
    LegacyReplicateConfig legacy;
    legacy.replica_num = 2;
    legacy.with_hard_pin = true;
    legacy.preferred_segments = {"segment-0"};
    legacy.data_type = ObjectDataType::TENSOR;
    legacy.group_ids = std::vector<std::string>{"group-0"};

    auto payload = struct_pack::serialize(legacy);
    auto result = struct_pack::deserialize<ReplicateConfig>(payload);

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->replica_num, legacy.replica_num);
    EXPECT_EQ(result->with_hard_pin, legacy.with_hard_pin);
    EXPECT_EQ(result->preferred_segments, legacy.preferred_segments);
    EXPECT_EQ(result->data_type, legacy.data_type);
    EXPECT_EQ(result->group_ids, legacy.group_ids);
    EXPECT_FALSE(result->host_id.has_value());
}

TEST(ReplicateConfigCompatibilityTest, CurrentPayloadDeserializesAsLegacy) {
    ReplicateConfig current;
    current.replica_num = 2;
    current.with_hard_pin = true;
    current.preferred_segments = {"segment-0"};
    current.data_type = ObjectDataType::TENSOR;
    current.host_id = "host-0";
    current.group_ids = std::vector<std::string>{"group-0"};

    auto payload = struct_pack::serialize(current);
    auto result = struct_pack::deserialize<LegacyReplicateConfig>(payload);

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->replica_num, current.replica_num);
    EXPECT_EQ(result->with_hard_pin, current.with_hard_pin);
    EXPECT_EQ(result->preferred_segments, current.preferred_segments);
    EXPECT_EQ(result->data_type, current.data_type);
    EXPECT_EQ(result->group_ids, current.group_ids);
}

}  // namespace
}  // namespace mooncake
