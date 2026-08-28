#include <array>

constexpr std::size_t kFeatureCount = 4;
const std::array<std::string, kFeatureCount> feature_names = {
    "relevance_score",
    "risk_score",
    "freshness_score",
    "confidence_prior"
};

constexpr double kOneBillionParameters = 1000000000.0;

constexpr float kErrorTolerance = 0.01f;

struct QuantizationConfig {
    float real_min;
    float real_max;

    int quantized_min;
    int quantized_max;

    float scale;

    int zero_point;

    float tolerance;
};


struct QuantizedValueReport {
    std::string feature_name;

    float fp32_value;
    float clipped_value;

    std::int8_t int8_payload;

    float dequantized_value;
    float absolute_error;

    bool clipped;
};

struct Int4PackedPair {
    std::string high_feature_name;
    std::string low_feature_name;

    std::uint8_t high_code;
    std::unit8_t low_code;
    std::uint8_t packed_byte;

    float high_dequantized;
    float low_dequantized;
};

int main()
{
    std::cout << "Part 1\n" << std::endl;
    return 0;
}