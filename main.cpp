#include <array>
#include <iostream>
#include <cstdint>
#include <math.h>
#include <algorithm>

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
    std::uint8_t low_code;
    std::uint8_t packed_byte;

    float high_dequantized;
    float low_dequantized;
};

QuantizationConfig buildAffineInt8Config(float real_min,
                                         float real_max,
                                         float tolerance) {
    const int qmin = static_cast<int>(std::numeric_limits<std::int8_t>::min());
    const int qmax = static_cast<int>(std::numeric_limits<std::int8_t>::max());

    const float scale = (real_max - real_min) / static_cast<float>(qmax-qmin);

    const int zero_point =
        static_cast<int>(std::round(static_cast<float>(qmin) - (real_min / scale)));

    return {real_min, real_max, qmin, qmax, scale, zero_point, tolerance};
}

QuantizationConfig buildUsignedInt4Config(float real_min,
                                          float real_max,
                                          float tolerance) {
    const int qmin = 0;
    const int qmax = 15;

    const float scale = (real_max - real_min) / static_cast<float>(qmax - qmin);
    const int zero_point = static_cast<int>(std::round(static_cast<float>(qmin) - (real_min / scale)));

    return {real_min, real_max, qmin, qmax, scale, zero_point, tolerance};
}

float clipToRealRange(float value, const QuantizationConfig &config) {
    return std::clamp(value, config.real_min, config.real_max);
}

std::int8_t quantizeToInt8(float value, const QuantizationConfig &config) {
    const float clipped = clipToRealRange(value, config);

    const float raw_quantized = (clipped / config.scale) + static_cast<float>(config.zero_point);

    const int rounded = static_cast<int>(std::round(raw_quantized));

    const int clamped = 
        std::clamp(rounded, config.quantized_min, config.quantized_max);

    return static_cast<std::int8_t>(clamped);
}

float dequantizeFromInt8(std::int8_t payload, const QuantizationConfig &config) {
    const int code = static_cast<int>(payload);
    
    return (static_cast<float>(code - config.zero_point)) * config.scale;
}

int main()
{
    std::cout << "Part 1\n" << std::endl;
    return 0;
}